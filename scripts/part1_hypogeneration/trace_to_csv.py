from __future__ import annotations

# flake8: noqa: E501

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from arc_pipeline import load_arc_tasks


def _task_id(task_name: str) -> str:
    return task_name.removesuffix(".json")


def _task_name(task_id: str) -> str:
    task_id = task_id.strip()
    return task_id if task_id.endswith(".json") else f"{task_id}.json"


def _grid_json(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_task_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    metadata: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {field.strip(): field for field in reader.fieldnames or []}
        if "task_id" not in fieldnames:
            raise ValueError("Metadata CSV must contain a `task_id` column.")
        task_id_field = fieldnames["task_id"]
        hint_field = fieldnames.get("hint")
        difficulty_field = fieldnames.get("difficulty")
        for row in reader:
            task_id = (row.get(task_id_field) or "").strip()
            if not task_id:
                continue
            metadata[_task_name(task_id)] = {
                "hint": (row.get(hint_field) or "").strip() if hint_field else "",
                "difficulty": (row.get(difficulty_field) or "").strip() if difficulty_field else "",
            }
    return metadata


def _load_test_examples(csv_path: Path | None) -> dict[str, dict[int, dict[str, str]]]:
    if csv_path is None:
        return {}

    tasks = load_arc_tasks(csv_path)
    examples: dict[str, dict[int, dict[str, str]]] = {}
    for task_name, splits in tasks.items():
        by_example: dict[int, dict[str, str]] = {}
        for example in splits.get("test", []):
            by_example[example.example_number] = {
                "input_grid": _grid_json(example.input_grid),
                "output_grid": _grid_json(example.output_grid),
            }
        examples[task_name] = by_example
    return examples


def _detect_trace_type(data: dict[str, Any]) -> str:
    config = data.get("config") or {}
    if "num_candidates" in config:
        return "direct_output"
    if "num_programs" in config:
        return "direct_program"

    for item in data.get("traces", []):
        result = item.get("result") or {}
        candidates = result.get("candidate_results") or []
        if candidates and "hypothesis" in candidates[0]:
            return "hypothesis_program"
    return "unknown"


def _base_row(task_name: str, metadata: dict[str, dict[str, str]]) -> dict[str, Any]:
    task_metadata = metadata.get(task_name, {})
    return {
        "task_id": _task_id(task_name),
        "difficulty": task_metadata.get("difficulty", ""),
        "hint": task_metadata.get("hint", ""),
    }


def _evaluation_fields(prefix: str, evaluation: dict[str, Any] | None) -> dict[str, Any]:
    evaluation = evaluation or {}
    return {
        f"{prefix}_pass_all": evaluation.get("pass_all", ""),
        f"{prefix}_correct_count": evaluation.get("correct_count", ""),
        f"{prefix}_total_examples": evaluation.get("total_examples", ""),
        f"{prefix}_accuracy": evaluation.get("accuracy", ""),
        f"{prefix}_error": evaluation.get("error", ""),
    }


def _convert_direct_output_test_rows(
    data: dict[str, Any],
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for item in data.get("traces", []):
        task_name = item.get("task_name", "")
        base = _base_row(task_name, metadata)
        base["status"] = item.get("status", "")
        if item.get("status") != "ok":
            rows.append({**base, "error": item.get("error", "")})
            continue

        result = item.get("result") or {}
        for test_result in result.get("test_results", []):
            candidates = test_result.get("candidates") or []
            known_output = test_result.get("known_output")
            if not candidates:
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "example_number": test_result.get("example_number", ""),
                        "candidate_index": "",
                        "is_top1": "",
                        "input_grid": _grid_json(test_result.get("input_grid")),
                        "output_grid": _grid_json(known_output),
                        "pred_output_grid": "",
                        "correct": False,
                        "parse_error": test_result.get("parse_error", ""),
                        "num_candidates_returned": 0,
                        "task_top1_hit": result.get("task_top1_hit", ""),
                        "task_topk_hit": result.get("task_topk_hit", ""),
                    }
                )
                continue

            for index, candidate in enumerate(candidates, start=1):
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "example_number": test_result.get("example_number", ""),
                        "candidate_index": index,
                        "is_top1": index == 1,
                        "input_grid": _grid_json(test_result.get("input_grid")),
                        "output_grid": _grid_json(known_output),
                        "pred_output_grid": _grid_json(candidate),
                        "correct": candidate == known_output,
                        "parse_error": test_result.get("parse_error", ""),
                        "num_candidates_returned": len(candidates),
                        "task_top1_hit": result.get("task_top1_hit", ""),
                        "task_topk_hit": result.get("task_topk_hit", ""),
                    }
                )

    fieldnames = [
        "task_id",
        "difficulty",
        "hint",
        "status",
        "example_number",
        "candidate_index",
        "is_top1",
        "input_grid",
        "output_grid",
        "pred_output_grid",
        "correct",
        "parse_error",
        "num_candidates_returned",
        "task_top1_hit",
        "task_topk_hit",
        "error",
    ]
    return rows, fieldnames


def _convert_direct_program_candidate_rows(
    data: dict[str, Any],
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for item in data.get("traces", []):
        task_name = item.get("task_name", "")
        base = _base_row(task_name, metadata)
        base["status"] = item.get("status", "")
        if item.get("status") != "ok":
            rows.append({**base, "error": item.get("error", "")})
            continue

        result = item.get("result") or {}
        selected_index = result.get("selected_program_index")
        for index, candidate in enumerate(result.get("candidate_results", [])):
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "candidate_index": index + 1,
                    "is_best": index == selected_index,
                    "program": candidate.get("program", ""),
                    **_evaluation_fields("train", candidate.get("train_evaluation")),
                    **_evaluation_fields("test", candidate.get("test_evaluation")),
                    "num_programs_requested": result.get("num_programs_requested", ""),
                    "num_programs_parsed": result.get("num_programs_parsed", ""),
                    "fallback_used": result.get("fallback_used", ""),
                }
            )

    fieldnames = [
        "task_id",
        "difficulty",
        "hint",
        "status",
        "candidate_index",
        "is_best",
        "program",
        "train_pass_all",
        "train_correct_count",
        "train_total_examples",
        "train_accuracy",
        "train_error",
        "test_pass_all",
        "test_correct_count",
        "test_total_examples",
        "test_accuracy",
        "test_error",
        "num_programs_requested",
        "num_programs_parsed",
        "fallback_used",
        "error",
    ]
    return rows, fieldnames


def _convert_hypothesis_program_candidate_rows(
    data: dict[str, Any],
    metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for item in data.get("traces", []):
        task_name = item.get("task_name", "")
        base = _base_row(task_name, metadata)
        base["status"] = item.get("status", "")
        if item.get("status") != "ok":
            rows.append({**base, "error": item.get("error", "")})
            continue

        result = item.get("result") or {}
        if result.get("task_hint"):
            base["hint"] = result.get("task_hint")
        hypotheses = result.get("hypotheses") or []
        attempt_counts: dict[str, int] = {}
        for index, candidate in enumerate(result.get("candidate_results", []), start=1):
            hypothesis = candidate.get("hypothesis", "")
            attempt_counts[hypothesis] = attempt_counts.get(hypothesis, 0) + 1
            evaluation = candidate.get("evaluation") or {}
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "candidate_index": index,
                    "hypothesis_index": hypotheses.index(hypothesis) + 1 if hypothesis in hypotheses else "",
                    "program_attempt_for_hypothesis": attempt_counts[hypothesis],
                    "is_best": (
                        hypothesis == result.get("best_hypothesis")
                        and candidate.get("program") == result.get("best_program")
                    ),
                    "is_useful_hypothesis": bool(evaluation.get("pass_all")),
                    "solved_train": result.get("solved_train", ""),
                    "hypothesis": hypothesis,
                    "program": candidate.get("program", ""),
                    **_evaluation_fields("train", evaluation),
                }
            )

    fieldnames = [
        "task_id",
        "difficulty",
        "hint",
        "status",
        "candidate_index",
        "hypothesis_index",
        "program_attempt_for_hypothesis",
        "is_best",
        "is_useful_hypothesis",
        "solved_train",
        "hypothesis",
        "program",
        "train_pass_all",
        "train_correct_count",
        "train_total_examples",
        "train_accuracy",
        "train_error",
        "error",
    ]
    return rows, fieldnames


def _convert_selected_test_rows(
    data: dict[str, Any],
    metadata: dict[str, dict[str, str]],
    test_examples: dict[str, dict[int, dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    trace_type = _detect_trace_type(data)
    if trace_type == "direct_output":
        return _convert_direct_output_test_rows(data, metadata)

    for item in data.get("traces", []):
        task_name = item.get("task_name", "")
        base = _base_row(task_name, metadata)
        base["status"] = item.get("status", "")
        if item.get("status") != "ok":
            rows.append({**base, "error": item.get("error", "")})
            continue

        result = item.get("result") or {}
        if result.get("task_hint"):
            base["hint"] = result.get("task_hint")

        predictions = result.get("selected_test_predictions")
        if predictions is None:
            predictions = result.get("test_predictions") or []

        for fallback_index, prediction in enumerate(predictions, start=1):
            example_number = prediction.get("example_number", fallback_index)
            example = test_examples.get(task_name, {}).get(example_number, {})
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "example_number": example_number,
                    "input_grid": example.get("input_grid", ""),
                    "output_grid": example.get("output_grid", ""),
                    "pred_output_grid": _grid_json(prediction.get("prediction")),
                    "correct": prediction.get("matches_known_output", ""),
                    "prediction_error": prediction.get("error", ""),
                    "selected_program_index": result.get("selected_program_index", ""),
                    "best_hypothesis": result.get("best_hypothesis", ""),
                    "best_program": result.get("best_program") or result.get("selected_program", ""),
                    "solved_train": result.get("solved_train", ""),
                }
            )

    fieldnames = [
        "task_id",
        "difficulty",
        "hint",
        "status",
        "example_number",
        "input_grid",
        "output_grid",
        "pred_output_grid",
        "correct",
        "prediction_error",
        "selected_program_index",
        "best_hypothesis",
        "best_program",
        "solved_train",
        "error",
    ]
    return rows, fieldnames


def convert_trace(
    data: dict[str, Any],
    *,
    view: str,
    metadata: dict[str, dict[str, str]],
    test_examples: dict[str, dict[int, dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[str], str]:
    trace_type = _detect_trace_type(data)
    if view == "auto":
        view = "test" if trace_type == "direct_output" else "candidates"

    if view == "test":
        rows, fieldnames = _convert_selected_test_rows(data, metadata, test_examples)
    elif trace_type == "direct_output":
        rows, fieldnames = _convert_direct_output_test_rows(data, metadata)
    elif trace_type == "direct_program":
        rows, fieldnames = _convert_direct_program_candidate_rows(data, metadata)
    elif trace_type == "hypothesis_program":
        rows, fieldnames = _convert_hypothesis_program_candidate_rows(data, metadata)
    else:
        raise ValueError("Could not detect trace type. Expected direct output, direct program, or hypothesis program trace.")

    return rows, fieldnames, trace_type


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Part 1 ARC trace JSON files to analysis-friendly CSV.")
    parser.add_argument("--trace-path", required=True, help="Input trace JSON path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--view",
        choices=["auto", "candidates", "test"],
        default="auto",
        help=(
            "auto: direct-output traces export test rows; program traces export candidate rows. "
            "candidates: export generated program candidates. test: export selected/best test predictions."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/task_data/train_hints_with_difficulty.csv",
        help="Optional CSV with task_id, hint, and difficulty columns.",
    )
    parser.add_argument(
        "--csv-path",
        default="data/task_data/ARC_training_tasks.csv",
        help="ARC CSV used to fill input_grid/output_grid for --view test on program traces.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    metadata_path = Path(args.metadata_csv) if args.metadata_csv else None
    csv_path = Path(args.csv_path) if args.csv_path else None

    data = _load_json(Path(args.trace_path))
    metadata = _load_task_metadata(metadata_path)
    test_examples = _load_test_examples(csv_path) if args.view == "test" else {}
    rows, fieldnames, trace_type = convert_trace(
        data,
        view=args.view,
        metadata=metadata,
        test_examples=test_examples,
    )
    _write_csv(Path(args.output_csv), rows, fieldnames)
    print(f"Detected trace_type={trace_type}")
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
