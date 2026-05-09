from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Trace payload must be a JSON object.")
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _compute_from_traces(payload: dict[str, Any]) -> dict[str, int]:
    traces = payload.get("traces") or []

    processed_tasks = 0
    top1_examples_total = 0
    top1_examples_correct = 0
    top1_tasks_solved = 0
    top6_examples_total = 0
    top6_examples_correct = 0
    top6_tasks_solved = 0

    for item in traces:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "ok":
            continue
        processed_tasks += 1
        result = item.get("result") or {}
        candidate_results = result.get("candidate_results") or []
        selected_program_index = _safe_int(result.get("selected_program_index"), 0)

        if candidate_results:
            best_train_correct = max(
                _safe_int((candidate.get("train_evaluation") or {}).get("correct_count"), 0)
                for candidate in candidate_results
                if isinstance(candidate, dict)
            )
            selected_train_correct = _safe_int(
                (
                    (candidate_results[selected_program_index].get("train_evaluation") or {})
                    if 0 <= selected_program_index < len(candidate_results)
                    else {}
                ).get("correct_count"),
                0,
            )
            if selected_train_correct != best_train_correct:
                raise ValueError(
                    f"selected_program_index mismatch in task {item.get('task_name')}: "
                    f"selected_train_correct={selected_train_correct}, best_train_correct={best_train_correct}"
                )

        selected_test_eval = result.get("selected_test_evaluation") or {}
        total_top1 = _safe_int(selected_test_eval.get("total_examples"), 0)
        correct_top1 = _safe_int(selected_test_eval.get("correct_count"), 0)
        top1_examples_total += total_top1
        top1_examples_correct += correct_top1
        if total_top1 > 0 and bool(selected_test_eval.get("pass_all")):
            top1_tasks_solved += 1

        top6_total = _safe_int(result.get("top6_example_total"), 0)
        top6_hits = _safe_int(result.get("top6_example_hits"), 0)
        top6_examples_total += top6_total
        top6_examples_correct += top6_hits
        if top6_total > 0 and top6_hits == top6_total:
            top6_tasks_solved += 1

    return {
        "processed_tasks": processed_tasks,
        "top1_test_examples_total": top1_examples_total,
        "top1_test_examples_correct": top1_examples_correct,
        "top1_test_tasks_solved": top1_tasks_solved,
        "top6_test_examples_total": top6_examples_total,
        "top6_test_examples_correct": top6_examples_correct,
        "top6_test_tasks_solved": top6_tasks_solved,
    }


def _check_summary(payload: dict[str, Any], computed: dict[str, int]) -> list[str]:
    summary = payload.get("summary") or {}
    mismatches: list[str] = []
    for key, value in computed.items():
        summary_value = _safe_int(summary.get(key), -999999)
        if summary_value != value:
            mismatches.append(f"{key}: summary={summary_value}, computed={value}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency of baseline_direct_programs trace metrics.")
    parser.add_argument("--trace-path", required=True)
    args = parser.parse_args()

    payload = _load_json(Path(args.trace_path))
    computed = _compute_from_traces(payload)
    mismatches = _check_summary(payload, computed)

    if mismatches:
        print("Consistency check FAILED")
        for item in mismatches:
            print(f"- {item}")
        raise SystemExit(1)

    print("Consistency check PASSED")
    print(json.dumps(computed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
