from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def load_trace(trace_path: str | Path) -> dict[str, Any]:
    with open(trace_path, encoding="utf-8") as handle:
        return json.load(handle)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    evaluation = candidate.get("evaluation", {})
    return (
        evaluation.get("pass_all", False),
        evaluation.get("correct_count", 0),
        evaluation.get("accuracy", 0.0),
        -len(candidate.get("program", "")),
    )


def _format_evaluation(evaluation: dict[str, Any]) -> str:
    return (
        f"pass_all={evaluation.get('pass_all')} | "
        f"correct_count={evaluation.get('correct_count')}/{evaluation.get('total_examples')} | "
        f"accuracy={evaluation.get('accuracy', 0.0):.2%} | "
        f"error={evaluation.get('error')}"
    )


def _print_summary(data: dict[str, Any]) -> None:
    summary = data.get("summary", {})
    print("Summary")
    print(
        f"processed={summary.get('processed')} | "
        f"successful_tasks={summary.get('successful_tasks')} | "
        f"dataset_examples={summary.get('dataset_examples')} | "
        f"success_rate={summary.get('success_rate', 0.0):.2%}"
    )
    print()


def _print_task_table(data: dict[str, Any]) -> None:
    print("Tasks")
    for item in data.get("traces", []):
        task_name = item.get("task_name")
        status = item.get("status")
        if status != "ok":
            print(f"{task_name} | status={status} | error={item.get('error')}")
            continue

        result = item["result"]
        best_evaluation = result.get("best_evaluation") or {}
        print(
            f"{task_name} | solved_train={result.get('solved_train')} | "
            f"useful_hypotheses={len(result.get('useful_hypotheses', []))} | "
            f"best_correct={best_evaluation.get('correct_count', 0)}/{best_evaluation.get('total_examples', 0)}"
        )
    print()


def _find_task(data: dict[str, Any], task_name: str) -> dict[str, Any]:
    for item in data.get("traces", []):
        if item.get("task_name") == task_name:
            return item
    raise ValueError(f"Task {task_name!r} not found in trace file.")


def _print_task_detail(
    item: dict[str, Any],
    *,
    show_programs: bool,
    show_task_text: bool,
    max_candidates: int | None,
) -> None:
    print(f"Task: {item.get('task_name')}")
    print(f"Status: {item.get('status')}")
    if item.get("status") != "ok":
        print(f"Error: {item.get('error')}")
        return

    result = item["result"]
    print(f"Solved train: {result.get('solved_train')}")
    print(f"Useful hypotheses: {len(result.get('useful_hypotheses', []))}")
    print(f"Best evaluation: {_format_evaluation(result.get('best_evaluation') or {})}")
    print()

    if show_task_text:
        print("Task Text")
        print(result.get("task_text", ""))
        print()

    print("Raw Hypotheses")
    for index, hypothesis in enumerate(result.get("hypotheses", []), start=1):
        print(f"[{index}]")
        print(hypothesis)
        print()

    candidates = sorted(result.get("candidate_results", []), key=_candidate_sort_key, reverse=True)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    print("Candidate Results")
    for index, candidate in enumerate(candidates, start=1):
        print(f"Candidate {index}")
        print("Hypothesis:")
        print(candidate.get("hypothesis", ""))
        print("Evaluation:")
        print(_format_evaluation(candidate.get("evaluation", {})))
        if show_programs:
            print("Program:")
            print(candidate.get("program", ""))
        print()

    if result.get("test_predictions"):
        print("Test Predictions")
        print(json.dumps(result["test_predictions"], ensure_ascii=False, indent=2))


def _print_all_task_details(
    data: dict[str, Any],
    *,
    show_programs: bool,
    show_task_text: bool,
    max_candidates: int | None,
) -> None:
    traces = data.get("traces", [])
    for index, item in enumerate(traces, start=1):
        print("=" * 80)
        print(f"Task {index}/{len(traces)}")
        print("=" * 80)
        _print_task_detail(
            item,
            show_programs=show_programs,
            show_task_text=show_task_text,
            max_candidates=max_candidates,
        )
        print()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect ARC batch trace output.")
    parser.add_argument(
        "--trace-path",
        default="outputs/train3_gpt54mini_trace.json",
        help="Path to the trace JSON file.",
    )
    parser.add_argument(
        "--task-name",
        help="If provided, show detailed results for one task.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all tasks and their high-level results.",
    )
    parser.add_argument(
        "--all-task-details",
        action="store_true",
        help="Show detailed results for every task in the trace file.",
    )
    parser.add_argument(
        "--hide-programs",
        action="store_true",
        help="Hide full generated program text in detailed task view.",
    )
    parser.add_argument(
        "--show-task-text",
        action="store_true",
        help="Include the full formatted task prompt text in detailed task view.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on the number of candidate results to print in detailed task view.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to save the rendered analysis text, for example outputs/007bbfb7_analysis.txt",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    data = load_trace(args.trace_path)
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        _print_summary(data)

        if args.list_tasks or (not args.task_name and not args.all_task_details):
            _print_task_table(data)

        if args.task_name:
            item = _find_task(data, args.task_name)
            _print_task_detail(
                item,
                show_programs=not args.hide_programs,
                show_task_text=args.show_task_text,
                max_candidates=args.max_candidates,
            )
        elif args.all_task_details:
            _print_all_task_details(
                data,
                show_programs=not args.hide_programs,
                show_task_text=args.show_task_text,
                max_candidates=args.max_candidates,
            )

    rendered = buffer.getvalue()
    print(rendered, end="")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()
