from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from arc_pipeline import load_arc_tasks


ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 teal
    "#8B4513",  # 9 brown
]


def _build_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(ARC_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)
    return cmap, norm


def _plot_grid(ax: plt.Axes, grid: np.ndarray, title: str, cmap: ListedColormap, norm: BoundaryNorm) -> None:
    height, width = grid.shape
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="#666666", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)


def visualize_task(task_name: str, csv_path: str | Path, save_path: str | Path | None = None, show: bool = True) -> None:
    tasks = load_arc_tasks(csv_path)
    if task_name not in tasks:
        available = ", ".join(sorted(tasks)[:10])
        raise ValueError(f"Task {task_name!r} not found in {csv_path}. Sample tasks: {available}")

    task = tasks[task_name]
    examples = task.get("train", []) + task.get("test", [])
    if not examples:
        raise ValueError(f"Task {task_name!r} has no examples.")

    cmap, norm = _build_cmap()
    num_rows = len(examples)
    fig, axes = plt.subplots(num_rows, 2, figsize=(8, max(3 * num_rows, 4)))
    if num_rows == 1:
        axes = np.array([axes])

    train_count = len(task.get("train", []))

    for row_index, example in enumerate(examples):
        split = "train" if row_index < train_count else "test"
        _plot_grid(
            axes[row_index, 0],
            example.input_grid,
            f"{split} #{example.example_number} input",
            cmap,
            norm,
        )
        if example.output_grid is not None:
            _plot_grid(
                axes[row_index, 1],
                example.output_grid,
                f"{split} #{example.example_number} output",
                cmap,
                norm,
            )
        else:
            axes[row_index, 1].axis("off")
            axes[row_index, 1].set_title(f"{split} #{example.example_number} output unavailable", fontsize=10, pad=8)

    fig.suptitle(task_name, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize all input/output examples for one ARC task.")
    parser.add_argument("--task-name", required=True, help="ARC task_name such as 00d62c1b.json")
    parser.add_argument(
        "--csv-path",
        default="data/task_data/ARC_training_tasks.csv",
        help="Path to the ARC CSV file.",
    )
    parser.add_argument(
        "--save-path",
        help="Optional path to save the rendered figure, for example outputs/00d62c1b.png",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful when only saving an image.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    visualize_task(
        task_name=args.task_name,
        csv_path=args.csv_path,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
