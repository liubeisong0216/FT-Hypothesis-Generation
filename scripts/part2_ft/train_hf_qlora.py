from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _parse_target_modules(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a QLoRA adapter with TRL + PEFT on prompt-completion ARC hypothesis data."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", default="outputs/ft_ready/current/train.jsonl")
    parser.add_argument("--valid-file", default="outputs/ft_ready/current/valid.jsonl")
    parser.add_argument("--output-dir", default="outputs/ft_runs/hf_qwen25_3b_qlora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        help="Force bf16 training if your GPU supports it.",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Transformers/TRL report_to value, e.g. none, wandb.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    train_file = Path(args.train_file)
    valid_file = Path(args.valid_file)
    if not train_file.exists():
        raise SystemExit(f"Training file not found: {train_file}")

    data_files = {"train": str(train_file)}
    has_valid = valid_file.exists()
    if has_valid:
        data_files["validation"] = str(valid_file)

    dataset = load_dataset("json", data_files=data_files)

    compute_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_parse_target_modules(args.target_modules),
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if has_valid else "no",
        eval_steps=args.eval_steps if has_valid else None,
        bf16=args.use_bf16,
        fp16=not args.use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_valid else None,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
