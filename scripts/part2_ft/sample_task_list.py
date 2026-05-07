from __future__ import annotations

import argparse
import random
from pathlib import Path

from arc_pipeline import load_arc_tasks


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a fixed list of ARC task names and save to a text file."
    )
    parser.add_argument(
        "--csv-path",
        default="data/task_data/ARC_evaluation_tasks.csv",
        help="Path to ARC CSV file to sample from.",
    )
    parser.add_argument(
        "--output-path",
        default="data/task_data/heldout_50_tasks.txt",
        help="Output text file path for sampled task names.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of unique task names to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tasks = sorted(load_arc_tasks(args.csv_path).keys())
    if args.num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0")
    if args.num_tasks > len(tasks):
        raise ValueError(f"Requested {args.num_tasks} tasks but only {len(tasks)} are available.")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(tasks, args.num_tasks))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")

    print(f"Sampled {len(sampled)} tasks from {args.csv_path}")
    print(f"Seed: {args.seed}")
    print(f"Saved task list to {output_path}")


if __name__ == "__main__":
    main()
