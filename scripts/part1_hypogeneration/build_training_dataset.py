from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_USER_PROMPT_TEMPLATE = """You will be given ARC input-output training examples. Infer the transformation rule.
Return exactly three lines in this format:
Describing the input grid: ...
Describing the size of the output grid: ...
Describing how to transform the grid: ...

Task examples:
{task_text}"""


def _collect_trace_paths(items: list[str]) -> list[Path]:
    trace_paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            trace_paths.extend(sorted(path.glob("*trace.json")))
        elif path.is_file():
            trace_paths.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

    unique_paths: list[Path] = []
    seen = set()
    for path in trace_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_user_prompt(task_text: str, template: str) -> str:
    return template.format(task_text=task_text).strip()


def build_dataset(
    trace_paths: list[Path],
    *,
    user_prompt_template: str,
    max_per_task: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    per_task_counts: defaultdict[str, int] = defaultdict(int)
    source_stats: dict[str, int] = {}

    for trace_path in trace_paths:
        data = _load_json(trace_path)
        added_from_source = 0
        for item in data.get("traces", []):
            if item.get("status") != "ok":
                continue

            result = item.get("result", {})
            task_name = item.get("task_name")
            task_text = result.get("task_text")
            useful_hypotheses = result.get("useful_hypotheses", [])
            if not task_name or not task_text or not useful_hypotheses:
                continue

            for hypothesis in useful_hypotheses:
                key = (task_name, hypothesis)
                if key in seen_pairs:
                    continue
                if max_per_task is not None and per_task_counts[task_name] >= max_per_task:
                    continue

                seen_pairs.add(key)
                per_task_counts[task_name] += 1
                added_from_source += 1
                records.append(
                    {
                        "task_name": task_name,
                        "input": task_text,
                        "output": hypothesis,
                        "messages": [
                            {
                                "role": "user",
                                "content": _build_user_prompt(task_text, user_prompt_template),
                            },
                            {
                                "role": "assistant",
                                "content": hypothesis,
                            },
                        ],
                        "source_trace": str(trace_path),
                    }
                )
        source_stats[str(trace_path)] = added_from_source

    summary = {
        "trace_files": [str(path) for path in trace_paths],
        "num_trace_files": len(trace_paths),
        "num_examples": len(records),
        "num_tasks": len(per_task_counts),
        "max_per_task": max_per_task,
        "examples_per_task": dict(sorted(per_task_counts.items())),
        "examples_per_source": source_stats,
    }
    return records, summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump({"messages": record["messages"]}, handle, ensure_ascii=False)
            handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a merged verified ARC hypothesis training dataset from one or more trace files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Trace JSON files or directories containing *trace.json files.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/ft_dataset/verified_hypotheses.json",
        help="Path to save the merged dataset JSON with metadata.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/ft_dataset/verified_hypotheses_chat.jsonl",
        help="Path to save chat-style JSONL for fine-tuning.",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/ft_dataset/verified_hypotheses_summary.json",
        help="Path to save dataset summary JSON.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Optional cap on the number of kept verified hypotheses per task after deduplication.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    trace_paths = _collect_trace_paths(args.inputs)
    records, summary = build_dataset(
        trace_paths,
        user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
        max_per_task=args.max_per_task,
    )

    output_json = Path(args.output_json)
    output_jsonl = Path(args.output_jsonl)
    summary_output = Path(args.summary_output)

    _write_json(
        output_json,
        {
            "summary": summary,
            "records": records,
        },
    )
    _write_json(summary_output, summary)
    _write_jsonl(output_jsonl, records)

    print(f"Loaded {len(trace_paths)} trace file(s)")
    print(f"Kept {summary['num_examples']} verified hypothesis example(s)")
    print(f"Covered {summary['num_tasks']} unique task(s)")
    print(f"Saved JSON dataset to {output_json}")
    print(f"Saved JSONL dataset to {output_jsonl}")
    print(f"Saved summary to {summary_output}")


if __name__ == "__main__":
    main()
