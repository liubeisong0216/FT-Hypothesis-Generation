from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from arc_pipeline import (
    ARCExample,
    LLMConfig,
    _chat_completion,
    _run_program_on_test_examples,
    evaluate_program,
    load_arc_tasks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH_DEFAULT = PROJECT_ROOT / "prompts" / "baseline_program_generation.prompt"


def _grid_to_text(grid: np.ndarray) -> str:
    return np.array2string(grid, separator=" ", max_line_width=10_000)


def _format_train_cases(train_examples: list[ARCExample]) -> str:
    blocks: list[str] = []
    for index, example in enumerate(train_examples):
        if example.output_grid is None:
            continue
        blocks.append(
            "\n".join(
                [
                    f"Case {index}:",
                    "Input:",
                    _grid_to_text(example.input_grid),
                    "Output:",
                    _grid_to_text(example.output_grid),
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_prompt(
    *,
    prompt_path: Path,
    train_cases: str,
    num_programs: int,
) -> str:
    template = Template(prompt_path.read_text(encoding="utf-8"))
    return template.safe_substitute(
        train_cases=train_cases,
        num_programs=str(num_programs),
    ).strip()


def _extract_program_blocks(raw_text: str) -> list[str]:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = [block.strip() for block in blocks if block.strip()]
    if cleaned:
        return cleaned
    text = raw_text.strip()
    return [text] if text else []


def _render_prompt_for_single_program(prompt: str, num_programs: int) -> str:
    rewritten = re.sub(
        rf"Generate exactly\s+{re.escape(str(num_programs))}\s+candidate Python programs",
        "Generate exactly 1 candidate Python program",
        prompt,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"Return exactly\s+\d+\s+separate code blocks",
        "Return exactly 1 separate code block",
        rewritten,
        flags=re.IGNORECASE,
    )
    return rewritten


def _load_task_list(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        tasks = [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]
    if not tasks:
        raise ValueError(f"Task list is empty: {path}")
    return tasks


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ARC baseline: directly generate candidate programs, validate on train, test best valid."
    )
    parser.add_argument("--csv-path", default="data/task_data/ARC_training_tasks.csv")
    parser.add_argument("--task-name")
    parser.add_argument("--task-list-path")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--num-programs", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=4000)
    parser.add_argument("--prompt-path", default=str(PROMPT_PATH_DEFAULT))
    parser.add_argument("--output-json", default="outputs/baseline_direct_programs/trace.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.num_programs <= 0:
        raise ValueError("--num-programs must be > 0")

    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    tasks = load_arc_tasks(args.csv_path)
    if args.task_name:
        if args.task_name not in tasks:
            raise ValueError(f"Task {args.task_name!r} not found in CSV.")
        task_names = [args.task_name]
    elif args.task_list_path:
        listed = _load_task_list(Path(args.task_list_path))
        task_names = [task_name for task_name in listed if task_name in tasks]
        missing = [task_name for task_name in listed if task_name not in tasks]
        if missing:
            print(f"Warning: {len(missing)} task(s) from list not found in CSV and skipped.")
        if not task_names:
            raise ValueError("No overlapping tasks between task list and CSV.")
    else:
        task_names = sorted(tasks)
    if args.task_limit is not None:
        task_names = task_names[: args.task_limit]

    config = LLMConfig(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    traces: list[dict[str, Any]] = []
    parse_shortfall_tasks = 0
    fallback_triggered_tasks = 0
    fallback_added_programs_total = 0
    fallback_fully_recovered_tasks = 0
    top1_test_examples_total = 0
    top1_test_examples_correct = 0
    top1_test_tasks_solved = 0
    top6_test_examples_total = 0
    top6_test_examples_correct = 0
    top6_test_tasks_solved = 0

    for index, task_name in enumerate(task_names, start=1):
        task = tasks[task_name]
        train_examples = task.get("train", [])
        test_examples = task.get("test", [])

        train_cases_text = _format_train_cases(train_examples)
        if not train_cases_text:
            traces.append(
                {
                    "task_name": task_name,
                    "status": "error",
                    "error": "Task has no train examples with outputs.",
                }
            )
            continue

        prompt = _render_prompt(
            prompt_path=prompt_path,
            train_cases=train_cases_text,
            num_programs=args.num_programs,
        )
        raw_output = _chat_completion(
            messages=[{"role": "user", "content": prompt}],
            config=config,
        )
        programs = _extract_program_blocks(raw_output)
        raw_model_outputs = [raw_output]
        fallback_used = False
        if len(programs) < args.num_programs:
            parse_shortfall_tasks += 1
            fallback_used = True
            fallback_triggered_tasks += 1
            missing = args.num_programs - len(programs)
            single_program_prompt = _render_prompt_for_single_program(prompt, args.num_programs)
            for _ in range(missing):
                single_output = _chat_completion(
                    messages=[{"role": "user", "content": single_program_prompt}],
                    config=config,
                )
                raw_model_outputs.append(single_output)
                extra_programs = _extract_program_blocks(single_output)
                if extra_programs:
                    programs.append(extra_programs[0])
                    fallback_added_programs_total += 1
        if len(programs) > args.num_programs:
            programs = programs[: args.num_programs]
        if fallback_used and len(programs) >= args.num_programs:
            fallback_fully_recovered_tasks += 1

        candidate_results: list[dict[str, Any]] = []
        test_examples_with_outputs = [example for example in test_examples if example.output_grid is not None]
        per_program_test_predictions: list[list[dict[str, Any]] | None] = []
        for program in programs:
            train_evaluation = evaluate_program(program, train_examples)
            test_evaluation: dict[str, Any] | None = None
            test_predictions_with_matches: list[dict[str, Any]] | None = None
            if test_examples_with_outputs:
                test_evaluation = evaluate_program(program, test_examples_with_outputs)
                try:
                    test_predictions_with_matches = _run_program_on_test_examples(program, test_examples_with_outputs)
                except Exception as exc:  # noqa: BLE001
                    test_predictions_with_matches = [{"error": f"{type(exc).__name__}: {exc}"}]
            per_program_test_predictions.append(test_predictions_with_matches)
            candidate_results.append(
                {
                    "program": program,
                    "train_evaluation": train_evaluation,
                    "test_evaluation": test_evaluation,
                }
            )

        selected_program_index = 0
        selected_program: str | None = programs[0] if programs else None
        selected_train_evaluation: dict[str, Any] | None = None
        selected_test_evaluation: dict[str, Any] | None = None
        selected_test_predictions: list[dict[str, Any]] = []
        if candidate_results:
            selected_program_index = max(
                range(len(candidate_results)),
                key=lambda idx: candidate_results[idx]["train_evaluation"].get("correct_count", 0),
            )
            selected_program = candidate_results[selected_program_index]["program"]
            selected_train_evaluation = candidate_results[selected_program_index]["train_evaluation"]
            selected_test_evaluation = candidate_results[selected_program_index]["test_evaluation"]
            selected_test_predictions = per_program_test_predictions[selected_program_index] or []

        if test_examples_with_outputs:
            top1_test_examples_total += len(test_examples_with_outputs)
            if selected_test_evaluation is not None:
                top1_test_examples_correct += int(selected_test_evaluation.get("correct_count", 0) or 0)
                if bool(selected_test_evaluation.get("pass_all")):
                    top1_test_tasks_solved += 1

            example_numbers = [example.example_number for example in test_examples_with_outputs]
            top6_match_by_example = {example_number: False for example_number in example_numbers}
            for predictions in per_program_test_predictions:
                if not predictions:
                    continue
                for item in predictions:
                    if not isinstance(item, dict):
                        continue
                    example_number = item.get("example_number")
                    if example_number in top6_match_by_example and "matches_known_output" in item:
                        top6_match_by_example[example_number] = (
                            top6_match_by_example[example_number] or bool(item["matches_known_output"])
                        )

            top6_test_examples_total += len(test_examples_with_outputs)
            top6_hits_for_task = sum(1 for matched in top6_match_by_example.values() if matched)
            top6_test_examples_correct += top6_hits_for_task
            if top6_hits_for_task == len(test_examples_with_outputs):
                top6_test_tasks_solved += 1

        traces.append(
            {
                "task_name": task_name,
                "status": "ok",
                "result": {
                    "num_programs_requested": args.num_programs,
                    "num_programs_parsed": len(programs),
                    "fallback_used": fallback_used,
                    "raw_model_output": raw_output,
                    "raw_model_outputs": raw_model_outputs,
                    "candidate_results": candidate_results,
                    "selected_program_index": selected_program_index,
                    "selected_train_evaluation": selected_train_evaluation,
                    "selected_test_evaluation": selected_test_evaluation,
                    "selected_program": selected_program,
                    "selected_test_predictions": selected_test_predictions,
                    "top6_example_hits": (
                        top6_hits_for_task if test_examples_with_outputs else None
                    ),
                    "top6_example_total": (
                        len(test_examples_with_outputs) if test_examples_with_outputs else None
                    ),
                },
            }
        )
        print(
            f"[{index}/{len(task_names)}] {task_name}: "
            f"parsed={len(programs)}, "
            f"best_train_correct={selected_train_evaluation.get('correct_count', 0) if selected_train_evaluation else 0}"
        )

    summary = {
        "processed_tasks": len(task_names),
        "num_programs_requested": args.num_programs,
        "tasks_with_parse_shortfall": parse_shortfall_tasks,
        "fallback_triggered_tasks": fallback_triggered_tasks,
        "fallback_added_programs_total": fallback_added_programs_total,
        "fallback_fully_recovered_tasks": fallback_fully_recovered_tasks,
        "top1_test_examples_total": top1_test_examples_total,
        "top1_test_examples_correct": top1_test_examples_correct,
        "top1_test_example_accuracy_unconditional": (
            top1_test_examples_correct / top1_test_examples_total if top1_test_examples_total else 0.0
        ),
        "top1_test_tasks_solved": top1_test_tasks_solved,
        "top1_test_task_success_rate_unconditional": (
            top1_test_tasks_solved / len(task_names) if task_names else 0.0
        ),
        "top6_test_examples_total": top6_test_examples_total,
        "top6_test_examples_correct": top6_test_examples_correct,
        "top6_test_example_accuracy_unconditional": (
            top6_test_examples_correct / top6_test_examples_total if top6_test_examples_total else 0.0
        ),
        "top6_test_tasks_solved": top6_test_tasks_solved,
        "top6_test_task_success_rate_unconditional": (
            top6_test_tasks_solved / len(task_names) if task_names else 0.0
        ),
    }

    payload = {
        "config": {
            "csv_path": args.csv_path,
            "task_name": args.task_name,
            "task_list_path": args.task_list_path,
            "task_limit": args.task_limit,
            "model": args.model,
            "num_programs": args.num_programs,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_completion_tokens,
            "prompt_path": str(prompt_path),
        },
        "summary": summary,
        "traces": traces,
    }
    output_path = Path(args.output_json)
    _write_json(output_path, payload)

    print(
        "Summary: "
        f"processed_tasks={summary['processed_tasks']}, "
        f"top1_test_example_accuracy_unconditional={summary['top1_test_example_accuracy_unconditional']:.2%}, "
        f"top1_test_task_success_rate_unconditional={summary['top1_test_task_success_rate_unconditional']:.2%}, "
        f"top6_test_example_accuracy_unconditional={summary['top6_test_example_accuracy_unconditional']:.2%}, "
        f"top6_test_task_success_rate_unconditional={summary['top6_test_task_success_rate_unconditional']:.2%}"
    )
    print(f"Saved trace to {output_path}")


if __name__ == "__main__":
    main()
