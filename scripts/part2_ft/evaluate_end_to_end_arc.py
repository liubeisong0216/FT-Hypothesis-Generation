from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template
from typing import Any

from arc_pipeline import (
    LLMConfig,
    evaluate_program,
    format_task_for_prompt,
    generate_program,
    load_arc_tasks,
)
from evaluate_hypothesis_model import _load_split_tasks


REQUIRED_PREFIXES = [
    "Describing the input grid:",
    "Describing the size of the output grid:",
    "Describing how to transform the grid:",
]
DEFAULT_HYPOTHESIS_PROMPT_TEMPLATE = str(Path(__file__).resolve().parents[2] / "prompts/hypothesis_generation.prompt")


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()).strip()


def _extract_structured_hypothesis(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").splitlines() if line.strip()]
    structured: list[str] = []
    for prefix in REQUIRED_PREFIXES:
        match = next((line for line in lines if line.startswith(prefix)), None)
        if match is None:
            return _normalize_text(text)
        structured.append(match)
    return _normalize_text("\n".join(structured))


def _load_hypothesis_model(
    *,
    model_name_or_path: str,
    adapter_path: str | None,
    load_in_4bit: bool,
    use_bf16: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch_dtype if not load_in_4bit else None,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def _generate_hypotheses(
    *,
    prompt: str,
    tokenizer: Any,
    model: Any,
    num_hypotheses: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    do_sample = temperature > 0.0
    generated = model.generate(
        **encoded,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_hypotheses,
        pad_token_id=tokenizer.eos_token_id,
    )

    hypotheses: list[str] = []
    seen = set()
    input_len = encoded["input_ids"].shape[1]
    for row in generated:
        new_tokens = row[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        normalized = _extract_structured_hypothesis(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        hypotheses.append(normalized)
    return hypotheses


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_hypothesis_prompt(*, task_text: str, num_hypotheses: int, prompt_template_path: Path) -> str:
    template_text = prompt_template_path.read_text(encoding="utf-8")
    return Template(template_text).safe_substitute(
        task_cases=task_text,
        num_hypotheses=str(num_hypotheses),
    ).strip()


def _load_task_list(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        tasks = [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]
    if not tasks:
        raise ValueError(f"Task list is empty: {path}")
    return tasks


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end ARC evaluation using a finetuned hypothesis model plus program generation."
    )
    parser.add_argument("--hypothesis-model", required=True, help="Base model name or local path.")
    parser.add_argument("--adapter-path", help="Optional PEFT adapter path.")
    parser.add_argument(
        "--csv-path",
        default="data/task_data/ARC_training_tasks.csv",
        help="ARC CSV with train/test examples.",
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/ft_ready/current/manifest.json",
        help="Optional manifest used to choose train/valid task splits.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "all"],
        default="valid",
        help="Which manifest split to evaluate.",
    )
    parser.add_argument("--task-name", help="Optional single task to evaluate.")
    parser.add_argument(
        "--task-list-path",
        help="Optional path to a newline-delimited task list file. If provided, only these tasks are evaluated.",
    )
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--num-hypotheses", type=int, default=4)
    parser.add_argument("--programs-per-hypothesis", type=int, default=2)
    parser.add_argument("--hypothesis-max-input-tokens", type=int, default=3072)
    parser.add_argument("--hypothesis-max-new-tokens", type=int, default=256)
    parser.add_argument("--hypothesis-temperature", type=float, default=0.7)
    parser.add_argument("--hypothesis-top-p", type=float, default=0.95)
    parser.add_argument(
        "--hypothesis-prompt-template",
        default=DEFAULT_HYPOTHESIS_PROMPT_TEMPLATE,
        help="Prompt template file used to generate hypotheses (supports $task_cases and $num_hypotheses).",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--program-model", default="gpt-5.4-mini")
    parser.add_argument("--program-temperature", type=float, default=0.2)
    parser.add_argument("--program-max-completion-tokens", type=int, default=1200)
    parser.add_argument(
        "--metric",
        choices=["train", "test"],
        default="test",
        help="Which split to use for success_rate (task solved if pass_all).",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/evals/end_to_end_arc_eval.json",
        help="Path to save the evaluation trace JSON.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    prompt_template_path = Path(args.hypothesis_prompt_template)
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Hypothesis prompt template not found: {prompt_template_path}")

    tasks = load_arc_tasks(args.csv_path)
    if args.task_name:
        task_names = [args.task_name]
    else:
        allowed = None
        manifest_path = Path(args.manifest_path)
        if args.split != "all" and manifest_path.exists():
            allowed = _load_split_tasks(manifest_path, args.split)
        task_names = sorted(tasks)
        if allowed is not None:
            task_names = [task_name for task_name in task_names if task_name in allowed]
        if args.task_list_path:
            listed_tasks = _load_task_list(Path(args.task_list_path))
            listed_set = set(listed_tasks)
            task_names = [task_name for task_name in listed_tasks if task_name in tasks and task_name in set(task_names)]
            missing = [task_name for task_name in listed_tasks if task_name not in tasks]
            if missing:
                print(f"Warning: {len(missing)} task(s) from task list were not found in CSV and will be skipped.")
            if not task_names:
                raise ValueError("No overlapping tasks between task list and selected dataset split.")
        if args.task_limit is not None:
            task_names = task_names[: args.task_limit]

    tokenizer, hypothesis_model = _load_hypothesis_model(
        model_name_or_path=args.hypothesis_model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        use_bf16=args.use_bf16,
    )
    program_config = LLMConfig(
        model=args.program_model,
        temperature=args.program_temperature,
        max_completion_tokens=args.program_max_completion_tokens,
    )

    traces: list[dict[str, Any]] = []
    solved_tasks_train = 0
    dataset_examples_train = 0
    solved_tasks_test = 0
    test_total_examples = 0
    test_correct_examples = 0

    for index, task_name in enumerate(task_names, start=1):
        task_examples = tasks[task_name]
        train_examples = task_examples.get("train", [])
        test_examples = task_examples.get("test", [])
        if not train_examples:
            traces.append(
                {
                    "task_name": task_name,
                    "status": "error",
                    "error": "Task has no train examples.",
                }
            )
            continue

        task_text = format_task_for_prompt(train_examples)
        prompt = _build_hypothesis_prompt(
            task_text=task_text,
            num_hypotheses=args.num_hypotheses,
            prompt_template_path=prompt_template_path,
        )
        hypotheses = _generate_hypotheses(
            prompt=prompt,
            tokenizer=tokenizer,
            model=hypothesis_model,
            num_hypotheses=args.num_hypotheses,
            max_input_tokens=args.hypothesis_max_input_tokens,
            max_new_tokens=args.hypothesis_max_new_tokens,
            temperature=args.hypothesis_temperature,
            top_p=args.hypothesis_top_p,
        )

        candidate_results: list[dict[str, Any]] = []
        best_candidate: dict[str, Any] | None = None

        for hypothesis in hypotheses:
            for attempt_index in range(args.programs_per_hypothesis):
                code_str = generate_program(
                    task_text,
                    hypothesis,
                    config=program_config,
                    attempt_index=attempt_index,
                )
                evaluation = evaluate_program(code_str, train_examples)
                candidate = {
                    "hypothesis": hypothesis,
                    "program": code_str,
                    "evaluation": evaluation,
                }
                candidate_results.append(candidate)

                if best_candidate is None:
                    best_candidate = candidate
                    continue

                current_eval = candidate["evaluation"]
                best_eval = best_candidate["evaluation"]
                current_score = (
                    current_eval["pass_all"],
                    current_eval["correct_count"],
                    -len(candidate["program"]),
                )
                best_score = (
                    best_eval["pass_all"],
                    best_eval["correct_count"],
                    -len(best_candidate["program"]),
                )
                if current_score > best_score:
                    best_candidate = candidate

        useful_hypotheses = sorted(
            {
                candidate["hypothesis"]
                for candidate in candidate_results
                if candidate["evaluation"]["pass_all"]
            }
        )
        solved_train = bool(best_candidate and best_candidate["evaluation"]["pass_all"])
        if solved_train:
            solved_tasks_train += 1
        dataset_examples_train += len(useful_hypotheses)

        # Evaluate generalization on test examples (exact match per example).
        solved_test = False
        best_test_evaluation: dict[str, Any] | None = None
        if best_candidate is not None and test_examples:
            # Some datasets may omit test outputs; those cannot be scored.
            test_examples_with_outputs = [ex for ex in test_examples if ex.output_grid is not None]
            if test_examples_with_outputs:
                best_test_evaluation = evaluate_program(best_candidate["program"], test_examples_with_outputs)
                solved_test = bool(best_test_evaluation.get("pass_all"))
                test_total_examples += int(best_test_evaluation.get("total_examples", 0) or 0)
                test_correct_examples += int(best_test_evaluation.get("correct_count", 0) or 0)
            else:
                # No scorable test outputs for this task.
                best_test_evaluation = None

        if solved_test:
            solved_tasks_test += 1

        traces.append(
            {
                "task_name": task_name,
                "status": "ok",
                "result": {
                    "task_text": task_text,
                    "hypotheses": hypotheses,
                    "candidate_results": candidate_results,
                    "best_program": best_candidate["program"] if best_candidate else None,
                    "best_hypothesis": best_candidate["hypothesis"] if best_candidate else None,
                    "best_evaluation": best_candidate["evaluation"] if best_candidate else None,
                    "solved_train": solved_train,
                    "useful_hypotheses": useful_hypotheses,
                    "solved_test": solved_test,
                    "best_test_evaluation": best_test_evaluation,
                },
            }
        )
        print(
            f"[{index}/{len(task_names)}] {task_name}: "
            f"generated={len(hypotheses)} | useful={len(useful_hypotheses)} | "
            f"best_correct={best_candidate['evaluation']['correct_count'] if best_candidate else 0}"
        )

    processed = len(task_names)
    success_rate_train = solved_tasks_train / processed if processed else 0.0
    success_rate_test = solved_tasks_test / processed if processed else 0.0
    test_accuracy = (test_correct_examples / test_total_examples) if test_total_examples else 0.0

    # Backward-compatible field names:
    # - `success_rate` is the primary metric requested by --metric (default: test).
    # - The train split results are preserved with *_train suffix.
    success_rate = success_rate_test if args.metric == "test" else success_rate_train
    summary = {
        "processed": processed,
        "successful_tasks_train": solved_tasks_train,
        "dataset_examples_train": dataset_examples_train,
        "success_rate_train": success_rate_train,
        "successful_tasks_test": solved_tasks_test,
        "dataset_examples_test": test_total_examples,
        "correct_test_examples": test_correct_examples,
        "test_accuracy": test_accuracy,
        "success_rate": success_rate,
    }
    payload = {
        "config": {
            "hypothesis_model": args.hypothesis_model,
            "adapter_path": args.adapter_path,
            "csv_path": args.csv_path,
            "manifest_path": args.manifest_path,
            "split": args.split,
            "task_name": args.task_name,
            "task_list_path": args.task_list_path,
            "task_limit": args.task_limit,
            "num_hypotheses": args.num_hypotheses,
            "programs_per_hypothesis": args.programs_per_hypothesis,
            "program_model": args.program_model,
            "metric": args.metric,
            "hypothesis_prompt_template": str(prompt_template_path),
        },
        "summary": summary,
        "traces": traces,
    }
    _write_json(Path(args.output_json), payload)

    print(
        "Summary: "
        f"processed={summary['processed']}, "
        f"success_rate={summary['success_rate']:.2%} "
        f"(train={summary['success_rate_train']:.2%}, test={summary['successful_tasks_test']}/{processed}), "
        f"test_accuracy={summary['test_accuracy']:.2%}"
    )
    print(f"Saved evaluation trace to {args.output_json}")


if __name__ == "__main__":
    main()
