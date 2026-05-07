from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_PREFIXES = [
    "Describing the input grid:",
    "Describing the size of the output grid:",
    "Describing how to transform the grid:",
]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_dataset_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format in {path}")


def _load_split_tasks(manifest_path: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    manifest = _load_json(manifest_path)
    key = f"{split}_tasks"
    tasks = manifest.get(key)
    if not isinstance(tasks, list):
        raise ValueError(f"Could not find task list {key!r} in {manifest_path}")
    return {str(task) for task in tasks}


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_structured_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    structured: list[str] = []
    for prefix in REQUIRED_PREFIXES:
        match = next((line for line in lines if line.startswith(prefix)), None)
        if match is None:
            return []
        structured.append(match)
    return structured


def _has_valid_format(text: str) -> bool:
    lines = _extract_structured_lines(text)
    return len(lines) == 3


def _canonicalize(text: str) -> str:
    lines = _extract_structured_lines(text)
    if not lines:
        return _normalize_text(text)
    return _normalize_text("\n".join(lines))


def _tokenize_for_score(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_for_score(prediction)
    ref_tokens = _tokenize_for_score(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _group_examples(
    records: list[dict[str, Any]],
    *,
    allowed_tasks: set[str] | None,
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"task_name": None, "prompt": None, "golds": []}
    )
    for record in records:
        task_name = str(record["task_name"])
        if allowed_tasks is not None and task_name not in allowed_tasks:
            continue

        entry = grouped[task_name]
        entry["task_name"] = task_name
        if entry["prompt"] is None:
            messages = record.get("messages") or []
            if messages and isinstance(messages[0], dict):
                entry["prompt"] = str(messages[0]["content"]).strip()
            else:
                entry["prompt"] = str(record["input"]).strip()
        entry["golds"].append(str(record["output"]).strip())

    return sorted(grouped.values(), key=lambda item: item["task_name"])


def _load_model(
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


def _generate_one(
    *,
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_model(
    examples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    max_input_tokens: int,
    max_new_tokens: int,
    max_tasks: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_tasks is not None:
        examples = examples[:max_tasks]

    results: list[dict[str, Any]] = []
    valid_format = 0
    raw_exact = 0
    normalized_exact = 0
    best_f1_total = 0.0

    for example in examples:
        prediction = _generate_one(
            prompt=example["prompt"],
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )
        prediction_normalized = _canonicalize(prediction)
        golds = example["golds"]
        raw_exact_match = prediction in golds
        normalized_golds = [_canonicalize(gold) for gold in golds]
        normalized_exact_match = prediction_normalized in normalized_golds
        best_f1 = max(_token_f1(prediction_normalized, gold) for gold in normalized_golds)
        is_valid_format = _has_valid_format(prediction)

        valid_format += int(is_valid_format)
        raw_exact += int(raw_exact_match)
        normalized_exact += int(normalized_exact_match)
        best_f1_total += best_f1

        results.append(
            {
                "task_name": example["task_name"],
                "prediction": prediction,
                "prediction_normalized": prediction_normalized,
                "valid_format": is_valid_format,
                "raw_exact_match": raw_exact_match,
                "normalized_exact_match": normalized_exact_match,
                "best_token_f1": best_f1,
                "golds": golds,
            }
        )

    total = len(results)
    summary = {
        "num_tasks_evaluated": total,
        "valid_format_rate": valid_format / total if total else math.nan,
        "raw_exact_match_rate": raw_exact / total if total else math.nan,
        "normalized_exact_match_rate": normalized_exact / total if total else math.nan,
        "average_best_token_f1": best_f1_total / total if total else math.nan,
    }
    return results, summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a hypothesis-generation model against ARC gold hypotheses."
    )
    parser.add_argument("--model", required=True, help="Base model name or local path.")
    parser.add_argument(
        "--adapter-path",
        help="Optional PEFT adapter path for a finetuned model.",
    )
    parser.add_argument(
        "--dataset-json",
        default="outputs/ft_dataset/current_verified_hypotheses.json",
        help="Merged verified hypothesis dataset JSON.",
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/ft_ready/current/manifest.json",
        help="Train/valid split manifest from prepare_finetune_data.py.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "all"],
        default="valid",
        help="Which task split to evaluate on.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/evals/hypothesis_eval_summary.json",
        help="Where to save summary JSON.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default="outputs/evals/hypothesis_eval_predictions.jsonl",
        help="Where to save per-task predictions JSONL.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    records = _load_dataset_records(Path(args.dataset_json))
    allowed_tasks = _load_split_tasks(Path(args.manifest_path), args.split)
    grouped_examples = _group_examples(records, allowed_tasks=allowed_tasks)
    tokenizer, model = _load_model(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        use_bf16=args.use_bf16,
    )
    results, summary = evaluate_model(
        grouped_examples,
        tokenizer=tokenizer,
        model=model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
    )

    payload = {
        "config": {
            "model": args.model,
            "adapter_path": args.adapter_path,
            "dataset_json": args.dataset_json,
            "manifest_path": args.manifest_path,
            "split": args.split,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_tasks": args.max_tasks,
            "load_in_4bit": args.load_in_4bit,
            "use_bf16": args.use_bf16,
        },
        "summary": summary,
    }
    _write_json(Path(args.output_json), payload)
    _write_jsonl(Path(args.predictions_jsonl), results)

    print(f"Evaluated {summary['num_tasks_evaluated']} task(s)")
    print(f"Valid format rate: {summary['valid_format_rate']:.2%}")
    print(f"Raw exact match rate: {summary['raw_exact_match_rate']:.2%}")
    print(f"Normalized exact match rate: {summary['normalized_exact_match_rate']:.2%}")
    print(f"Average best token F1: {summary['average_best_token_f1']:.4f}")
    print(f"Saved summary to {args.output_json}")
    print(f"Saved predictions to {args.predictions_jsonl}")


if __name__ == "__main__":
    main()
