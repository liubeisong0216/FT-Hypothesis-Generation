from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def _build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(Path(args.data_dir)),
        "--adapter-path",
        args.adapter_path,
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
    ]

    if args.num_layers is not None:
        cmd.extend(["--num-layers", str(args.num_layers)])
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    if args.resume_adapter_file:
        cmd.extend(["--resume-adapter-file", args.resume_adapter_file])
    return cmd


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local MLX LoRA/QLoRA finetuning on Apple Silicon."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help=(
            "Model repo or local MLX model path. If this points to a quantized MLX model, "
            "mlx-lm will use QLoRA automatically."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/ft_ready/current",
        help="Directory containing MLX finetune data files.",
    )
    parser.add_argument(
        "--adapter-path",
        default="outputs/ft_runs/mlx_qwen25_3b",
        help="Directory where MLX adapters will be saved.",
    )
    parser.add_argument("--iters", type=int, default=200, help="Number of optimizer iterations.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-step batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of transformer layers to adapt locally. Lower is lighter.",
    )
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory use.",
    )
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Train on completion tokens only for prompt-completion datasets.",
    )
    parser.add_argument(
        "--use-text-data",
        action="store_true",
        help="Use text-format JSONL files to avoid chat-template processing on local smoke tests.",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=None,
        help="Optional cap on the number of shortest local text examples to keep for smoke tests.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=None,
        help="Optional maximum text length in characters for local text-mode smoke tests.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        help="Optional path to an existing MLX adapter safetensors file.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved mlx-lm command without executing it.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / ("train_text.jsonl" if args.use_text_data else "train.jsonl")
    if not train_file.exists():
        raise SystemExit(f"Expected training file at {train_file}")

    if args.use_text_data:
        text_dir = data_dir / "mlx_text_data"
        text_dir.mkdir(parents=True, exist_ok=True)

        def _load_text_rows(path: Path) -> list[dict]:
            rows = []
            if not path.exists():
                return rows
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        def _filter_rows(rows: list[dict]) -> list[dict]:
            if args.max_text_chars is not None:
                rows = [row for row in rows if len(row.get("text", "")) <= args.max_text_chars]
            rows = sorted(rows, key=lambda row: len(row.get("text", "")))
            if args.smoke_size is not None:
                rows = rows[: args.smoke_size]
            return rows

        train_rows = _filter_rows(_load_text_rows(data_dir / "train_text.jsonl"))
        valid_rows = _filter_rows(_load_text_rows(data_dir / "valid_text.jsonl"))
        if not train_rows:
            raise SystemExit(
                "No local text-mode training rows left after filtering. "
                "Relax --max-text-chars or increase --smoke-size."
            )

        for dst_name, rows in [("train.jsonl", train_rows), ("valid.jsonl", valid_rows)]:
            dst = text_dir / dst_name
            with dst.open("w", encoding="utf-8") as handle:
                for row in rows:
                    json.dump(row, handle, ensure_ascii=False)
                    handle.write("\n")
        args.data_dir = str(text_dir)

    command = _build_command(args)
    print("Resolved MLX command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.print_only:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
