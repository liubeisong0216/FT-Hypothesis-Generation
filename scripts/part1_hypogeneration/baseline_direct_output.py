from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Template
from typing import Any

import numpy as np

from arc_pipeline import ARCExample, LLMConfig, _chat_completion, load_arc_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH_DEFAULT = PROJECT_ROOT / "prompts" / "baseline_direct_output.prompt"


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
    test_input: np.ndarray,
    num_candidates: int,
) -> str:
    template = Template(prompt_path.read_text(encoding="utf-8"))
    return template.safe_substitute(
        train_cases=train_cases,
        test_input=_grid_to_text(test_input),
        num_candidates=str(num_candidates),
    ).strip()


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _normalize_grid(candidate: Any) -> list[list[int]]:
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("Candidate grid must be a non-empty 2D list.")
    normalized: list[list[int]] = []
    width: int | None = None
    for row in candidate:
        if not isinstance(row, list) or not row:
            raise ValueError("Each candidate grid row must be a non-empty list.")
        normalized_row: list[int] = []
        for cell in row:
            if not isinstance(cell, int):
                raise ValueError("Grid cells must be integers.")
            if cell < 0 or cell > 9:
                raise ValueError("Grid cells must be in [0, 9].")
            normalized_row.append(cell)
        if width is None:
            width = len(normalized_row)
        elif len(normalized_row) != width:
            raise ValueError("All rows in a candidate grid must have the same width.")
        normalized.append(normalized_row)
    return normalized


def _parse_candidates(raw_text: str, num_candidates: int) -> tuple[list[list[list[int]]], str | None]:
    try:
        payload = json.loads(_extract_json_text(raw_text))
    except Exception as exc:  # noqa: BLE001
        return [], f"JSONParseError: {type(exc).__name__}: {exc}"

    candidates_raw = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates_raw, list):
        return [], "InvalidFormat: Missing list field `candidates`."

    parsed: list[list[list[int]]] = []
    errors: list[str] = []
    for index, candidate in enumerate(candidates_raw):
        try:
            parsed.append(_normalize_grid(candidate))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"candidate_{index + 1}: {type(exc).__name__}: {exc}")

    if len(parsed) > num_candidates:
        parsed = parsed[:num_candidates]
    if len(parsed) < num_candidates:
        errors.append(f"expected_{num_candidates}_candidates_but_got_{len(parsed)}")
    if errors:
        return parsed, "; ".join(errors)
    return parsed, None


def _call_model(prompt: str, config: LLMConfig) -> str:
    return _chat_completion(
        messages=[{"role": "user", "content": prompt}],
        config=config,
    )


def _generate_candidates_single_call(
    *,
    prompt: str,
    num_candidates: int,
    config: LLMConfig,
) -> tuple[list[list[list[int]]], str | None, str]:
    raw_output = _call_model(prompt, config)
    candidates, parse_error = _parse_candidates(raw_output, num_candidates)
    return candidates, parse_error, raw_output


def _generate_candidates_multi_call(
    *,
    prompt: str,
    num_candidates: int,
    config: LLMConfig,
) -> tuple[list[list[list[int]]], list[str], list[str]]:
    candidates: list[list[list[int]]] = []
    parse_errors: list[str] = []
    raw_outputs: list[str] = []
    for _ in range(num_candidates):
        # In multi-call mode, each request should generate exactly one candidate
        # to reduce output length and avoid truncation.
        single_candidate_prompt = re.sub(
            rf"exactly\s+{re.escape(str(num_candidates))}\s+candidate output grids",
            "exactly 1 candidate output grids",
            prompt,
            flags=re.IGNORECASE,
        )
        single_candidate_prompt = re.sub(
            r'"candidates"\s+must\s+be\s+a\s+list\s+of\s+exactly\s+\d+\s+grids',
            '"candidates" must be a list of exactly 1 grids',
            single_candidate_prompt,
            flags=re.IGNORECASE,
        )
        raw_output = _call_model(single_candidate_prompt, config)
        raw_outputs.append(raw_output)
        parsed_one, parse_error = _parse_candidates(raw_output, 1)
        if parse_error:
            parse_errors.append(parse_error)
            continue
        if parsed_one:
            candidates.append(parsed_one[0])
    if len(candidates) > num_candidates:
        candidates = candidates[:num_candidates]
    return candidates, parse_errors, raw_outputs


def _equals_grid(candidate: list[list[int]], target: np.ndarray) -> bool:
    return np.array_equal(np.array(candidate, dtype=int), target)


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
    parser = argparse.ArgumentParser(description="ARC baseline: directly generate candidate output grids.")
    parser.add_argument("--csv-path", default="data/task_data/ARC_training_tasks.csv")
    parser.add_argument("--task-name")
    parser.add_argument("--task-list-path")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument(
        "--generation-mode",
        choices=["single_call", "multi_call"],
        default="single_call",
        help="single_call: one request returns all candidates; multi_call: one candidate per request.",
    )
    parser.add_argument(
        "--fallback-on-parse-error",
        action="store_true",
        help=(
            "When generation-mode=single_call, retry with multi_call only if parsing fails "
            "or returns fewer than num-candidates."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int, default=6000)
    parser.add_argument("--prompt-path", default=str(PROMPT_PATH_DEFAULT))
    parser.add_argument("--output-json", default="outputs/baseline_direct_output/trace.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be > 0")

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
    scored_examples = 0
    top1_example_hits = 0
    topk_example_hits = 0
    scored_tasks = 0
    top1_task_hits = 0
    topk_task_hits = 0
    single_call_parse_errors = 0
    fallback_triggered = 0
    fallback_recovered = 0

    for task_index, task_name in enumerate(task_names, start=1):
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

        test_results: list[dict[str, Any]] = []
        task_scored = False
        task_top1_all = True
        task_topk_all = True

        for test_example in test_examples:
            prompt = _render_prompt(
                prompt_path=prompt_path,
                train_cases=train_cases_text,
                test_input=test_example.input_grid,
                num_candidates=args.num_candidates,
            )
            generation_path = args.generation_mode
            raw_outputs: list[str] = []
            parse_errors: list[str] = []

            if args.generation_mode == "multi_call":
                candidates, multi_errors, multi_raw_outputs = _generate_candidates_multi_call(
                    prompt=prompt,
                    num_candidates=args.num_candidates,
                    config=config,
                )
                raw_outputs.extend(multi_raw_outputs)
                parse_errors.extend(multi_errors)
            else:
                candidates, parse_error, raw_output = _generate_candidates_single_call(
                    prompt=prompt,
                    num_candidates=args.num_candidates,
                    config=config,
                )
                raw_outputs.append(raw_output)
                if parse_error:
                    parse_errors.append(parse_error)
                    single_call_parse_errors += 1

                needs_fallback = bool(parse_error) or len(candidates) < args.num_candidates
                if args.fallback_on_parse_error and needs_fallback:
                    fallback_triggered += 1
                    generation_path = "fallback_multi_call"
                    candidates, multi_errors, multi_raw_outputs = _generate_candidates_multi_call(
                        prompt=prompt,
                        num_candidates=args.num_candidates,
                        config=config,
                    )
                    raw_outputs.extend(multi_raw_outputs)
                    parse_errors.extend(multi_errors)
                    if len(candidates) >= args.num_candidates and not multi_errors:
                        fallback_recovered += 1

            parse_error = "; ".join(parse_errors) if parse_errors else None

            item: dict[str, Any] = {
                "example_number": test_example.example_number,
                "input_grid": test_example.input_grid.tolist(),
                "candidates": candidates,
                "parse_error": parse_error,
                "generation_path": generation_path,
                "raw_model_output": raw_outputs[0] if raw_outputs else "",
                "raw_model_outputs": raw_outputs,
            }

            if test_example.output_grid is not None:
                task_scored = True
                scored_examples += 1
                top1_hit = bool(candidates and _equals_grid(candidates[0], test_example.output_grid))
                topk_hit = any(_equals_grid(candidate, test_example.output_grid) for candidate in candidates)
                top1_example_hits += int(top1_hit)
                topk_example_hits += int(topk_hit)
                task_top1_all = task_top1_all and top1_hit
                task_topk_all = task_topk_all and topk_hit
                item["known_output"] = test_example.output_grid.tolist()
                item["top1_hit"] = top1_hit
                item["topk_hit"] = topk_hit

            test_results.append(item)

        if task_scored:
            scored_tasks += 1
            top1_task_hits += int(task_top1_all)
            topk_task_hits += int(task_topk_all)

        traces.append(
            {
                "task_name": task_name,
                "status": "ok",
                "result": {
                    "num_train_examples": len(train_examples),
                    "num_test_examples": len(test_examples),
                    "test_results": test_results,
                    "task_scored": task_scored,
                    "task_top1_hit": task_top1_all if task_scored else None,
                    "task_topk_hit": task_topk_all if task_scored else None,
                },
            }
        )
        print(f"[{task_index}/{len(task_names)}] {task_name}: test_examples={len(test_examples)}")

    summary = {
        "processed_tasks": len(task_names),
        "num_candidates": args.num_candidates,
        "generation_mode": args.generation_mode,
        "fallback_on_parse_error": bool(args.fallback_on_parse_error),
        "single_call_parse_errors": single_call_parse_errors,
        "fallback_triggered": fallback_triggered,
        "fallback_recovered": fallback_recovered,
        "scored_examples": scored_examples,
        "top1_example_hits": top1_example_hits,
        "topk_example_hits": topk_example_hits,
        "top1_example_accuracy": (top1_example_hits / scored_examples) if scored_examples else 0.0,
        "topk_example_accuracy": (topk_example_hits / scored_examples) if scored_examples else 0.0,
        "scored_tasks": scored_tasks,
        "top1_task_hits": top1_task_hits,
        "topk_task_hits": topk_task_hits,
        "top1_task_success_rate": (top1_task_hits / scored_tasks) if scored_tasks else 0.0,
        "topk_task_success_rate": (topk_task_hits / scored_tasks) if scored_tasks else 0.0,
    }

    payload = {
        "config": {
            "csv_path": args.csv_path,
            "task_name": args.task_name,
            "task_list_path": args.task_list_path,
            "task_limit": args.task_limit,
            "model": args.model,
            "num_candidates": args.num_candidates,
            "generation_mode": args.generation_mode,
            "fallback_on_parse_error": bool(args.fallback_on_parse_error),
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
        f"top1_example_accuracy={summary['top1_example_accuracy']:.2%}, "
        f"topk_example_accuracy={summary['topk_example_accuracy']:.2%}, "
        f"top1_task_success_rate={summary['top1_task_success_rate']:.2%}, "
        f"topk_task_success_rate={summary['topk_task_success_rate']:.2%}"
    )
    print(f"Saved trace to {output_path}")


if __name__ == "__main__":
    main()
