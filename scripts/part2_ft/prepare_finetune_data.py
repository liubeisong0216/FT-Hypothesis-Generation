from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_training_dataset import DEFAULT_USER_PROMPT_TEMPLATE


def _load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format in {path}")


def _prompt_from_record(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("role") == "user" and first.get("content"):
            return str(first["content"]).strip()

    task_text = record.get("input")
    if not task_text:
        raise ValueError("Record is missing both messages and input fields.")
    return DEFAULT_USER_PROMPT_TEMPLATE.format(task_text=task_text).strip()


def _completion_from_record(record: dict[str, Any]) -> str:
    completion = record.get("output")
    if completion:
        return str(completion).strip()

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and last.get("content"):
            return str(last["content"]).strip()

    raise ValueError("Record is missing both output and assistant message content.")


def build_splits(
    records: list[dict[str, Any]],
    *,
    valid_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0.0, 1.0).")

    by_task: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        task_name = record.get("task_name")
        if not task_name:
            raise ValueError("Each record must include task_name.")
        by_task[str(task_name)].append(
            {
                "prompt": _prompt_from_record(record),
                "completion": _completion_from_record(record),
            }
        )

    task_names = list(by_task)
    rng = random.Random(seed)
    rng.shuffle(task_names)

    num_valid_tasks = int(round(len(task_names) * valid_ratio))
    if valid_ratio > 0.0 and len(task_names) > 1:
        num_valid_tasks = max(1, min(num_valid_tasks, len(task_names) - 1))
    else:
        num_valid_tasks = 0

    valid_task_names = set(task_names[:num_valid_tasks])
    train_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []

    for task_name in task_names:
        target = valid_rows if task_name in valid_task_names else train_rows
        target.extend(by_task[task_name])

    summary = {
        "seed": seed,
        "valid_ratio": valid_ratio,
        "num_total_examples": len(records),
        "num_total_tasks": len(task_names),
        "num_train_examples": len(train_rows),
        "num_valid_examples": len(valid_rows),
        "num_train_tasks": len(task_names) - len(valid_task_names),
        "num_valid_tasks": len(valid_task_names),
        "train_tasks": sorted(set(task_names) - valid_task_names),
        "valid_tasks": sorted(valid_task_names),
    }
    return train_rows, valid_rows, summary


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _write_text_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            text = f"{row['prompt']}\n{row['completion']}".strip()
            json.dump({"text": text}, handle, ensure_ascii=False)
            handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare prompt-completion train/valid JSONL files for finetuning."
    )
    parser.add_argument(
        "--input-json",
        default="outputs/ft_dataset/current_verified_hypotheses.json",
        help="Merged dataset JSON produced by build_training_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ft_ready/current",
        help="Directory where train.jsonl, valid.jsonl, and manifest.json will be written.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of tasks to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle tasks before splitting.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)

    records = _load_records(input_json)
    train_rows, valid_rows, summary = build_splits(
        records,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_text_jsonl(output_dir / "train_text.jsonl", train_rows)
    if valid_rows:
        _write_jsonl(output_dir / "valid.jsonl", valid_rows)
        _write_text_jsonl(output_dir / "valid_text.jsonl", valid_rows)
    _write_json(output_dir / "manifest.json", summary)

    print(f"Loaded {len(records)} merged example(s) from {input_json}")
    print(f"Wrote {len(train_rows)} train example(s) to {output_dir / 'train.jsonl'}")
    if valid_rows:
        print(f"Wrote {len(valid_rows)} valid example(s) to {output_dir / 'valid.jsonl'}")
    else:
        print("No validation split created")
    print(f"Wrote text-format train data to {output_dir / 'train_text.jsonl'}")
    print(f"Wrote split manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
