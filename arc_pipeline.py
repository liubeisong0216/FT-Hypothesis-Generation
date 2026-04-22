from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import signal
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Iterable, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
PROMPTS_DIR = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class ARCExample:
    input_grid: np.ndarray
    output_grid: Optional[np.ndarray]
    example_type: str
    example_number: int


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_completion_tokens: int = 900


def parse_grid(grid_str: str) -> np.ndarray:
    """
    Convert a serialized grid into a numpy array.

    Supports either:
    - ARC pipe format: "|077|777|077|"
    - Python list format: "[[0, 7, 7], [7, 7, 7], [0, 7, 7]]"
    """
    if grid_str is None:
        raise ValueError("grid_str cannot be None")

    grid_str = grid_str.strip()
    if not grid_str:
        raise ValueError("grid_str cannot be empty")

    if grid_str.startswith("|") and grid_str.endswith("|"):
        rows = [row for row in grid_str.strip("|").split("|") if row]
        parsed = [[int(cell) for cell in row] for row in rows]
        return np.array(parsed, dtype=int)

    parsed = ast.literal_eval(grid_str)
    return np.array(parsed, dtype=int)


def load_arc_tasks(csv_path: str | Path) -> dict[str, dict[str, list[ARCExample]]]:
    """
    Load ARC tasks from a CSV file and group them by task name.

    Returns:
        dict:
            key = task_name
            value = {
                "train": [ARCExample, ...],
                "test": [ARCExample, ...],
            }
    """
    grouped: dict[str, dict[str, list[tuple[int, ARCExample]]]] = defaultdict(
        lambda: {"train": [], "test": []}
    )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            example_type = row["example_type"].strip().lower()
            if example_type not in {"train", "test"}:
                continue

            output_grid = parse_grid(row["output_grid"]) if row.get("output_grid") else None
            example = ARCExample(
                input_grid=parse_grid(row["input_grid"]),
                output_grid=output_grid,
                example_type=example_type,
                example_number=int(row["example_number"]),
            )
            grouped[row["task_name"]][example_type].append((example.example_number, example))

    tasks: dict[str, dict[str, list[ARCExample]]] = {}
    for task_name, splits in grouped.items():
        tasks[task_name] = {
            split_name: [example for _, example in sorted(split_examples, key=lambda item: item[0])]
            for split_name, split_examples in splits.items()
        }

    return tasks


def _grid_to_text(grid: np.ndarray) -> str:
    return np.array2string(grid, separator=" ", max_line_width=10_000)


def format_task_for_prompt(
    task_examples: Iterable[ARCExample | tuple[np.ndarray, np.ndarray]],
    *,
    label_prefix: str = "Case",
) -> str:
    """
    Convert training examples into a prompt-friendly text block.
    """
    blocks: list[str] = []
    for index, example in enumerate(task_examples):
        if isinstance(example, ARCExample):
            input_grid = example.input_grid
            output_grid = example.output_grid
        else:
            input_grid, output_grid = example

        if output_grid is None:
            raise ValueError("Prompt formatting requires output grids for all examples.")

        blocks.append(
            "\n".join(
                [
                    f"{label_prefix} {index}:",
                    "Input:",
                    _grid_to_text(input_grid),
                    "Output:",
                    _grid_to_text(output_grid),
                ]
            )
        )
    return "\n\n".join(blocks)


def _load_prompt_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _render_prompt_template(path: str | Path, **kwargs: Any) -> str:
    template = Template(_load_prompt_file(path))
    return template.substitute(**kwargs).strip()


def _parse_prompt_messages(path: str | Path, **kwargs: Any) -> list[dict[str, str]]:
    raw_text = _render_prompt_template(path, **kwargs)
    pattern = re.compile(r"^\[(system|user|assistant)\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(raw_text))
    if not matches:
        raise ValueError(f"Prompt file {path} does not contain any [system]/[user]/[assistant] sections.")

    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        content = raw_text[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _get_openai_client(client: Optional[Any] = None) -> Any:
    if client is not None:
        return client
    if OpenAI is None:
        raise RuntimeError("openai is not installed. Install it or pass a custom client.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def _chat_completion(
    *,
    messages: list[dict[str, str]],
    config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
) -> str:
    llm_config = config or LLMConfig()
    llm_client = _get_openai_client(client)

    response = llm_client.chat.completions.create(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_completion_tokens=llm_config.max_completion_tokens,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _normalize_hypothesis_block(block: str) -> str:
    cleaned_lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
    if not cleaned_lines:
        return ""

    first_line = re.sub(r"^\s*(?:Candidate|Hypothesis)\s*\d+\s*:\s*", "", cleaned_lines[0], flags=re.IGNORECASE)
    cleaned_lines[0] = first_line
    return "\n".join(cleaned_lines).strip()


def _parse_hypothesis_blocks(raw_text: str, num_hypotheses: int) -> list[str]:
    blocks = [block for block in re.split(r"\n\s*\n", raw_text.strip()) if block.strip()]
    normalized_blocks = [_normalize_hypothesis_block(block) for block in blocks]

    hypotheses = [block for block in normalized_blocks if block]
    if not hypotheses:
        hypotheses = [_normalize_hypothesis_block(raw_text)]

    unique_hypotheses: list[str] = []
    seen = set()
    for hypothesis in hypotheses:
        if hypothesis not in seen:
            seen.add(hypothesis)
            unique_hypotheses.append(hypothesis)
        if len(unique_hypotheses) >= num_hypotheses:
            break

    return unique_hypotheses


def generate_hypotheses(
    task_text: str,
    num_hypotheses: int = 4,
    *,
    config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
) -> list[str]:
    """
    Call an LLM to generate candidate natural-language hypotheses.
    """
    messages = _parse_prompt_messages(
        PROMPTS_DIR / "hypothesis_generation.prompt",
        task_cases=task_text,
        num_hypotheses=num_hypotheses,
    )
    raw_text = _chat_completion(
        messages=messages,
        config=config,
        client=client,
    )
    return _parse_hypothesis_blocks(raw_text, num_hypotheses)


def _extract_code(raw_text: str) -> str:
    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return raw_text.strip()


def generate_program(
    task_text: str,
    hypothesis: str,
    *,
    config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
    attempt_index: int = 0,
) -> str:
    """
    Generate Python code implementing transform_grid for a given hypothesis.
    """
    attempt_guidance = ""
    if attempt_index > 0:
        attempt_guidance = (
            f"This is implementation attempt {attempt_index + 1}. "
            "Try a different implementation strategy while keeping the same underlying rule."
        )

    messages = _parse_prompt_messages(
        PROMPTS_DIR / "program_generation.prompt",
        task_examples=task_text.replace("Case ", "Example "),
        hypothesis=hypothesis,
        attempt_guidance=attempt_guidance,
    )
    raw_text = _chat_completion(
        messages=messages,
        config=config or LLMConfig(temperature=0.2, max_completion_tokens=1200),
        client=client,
    )
    return _extract_code(raw_text)


_BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "help",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
}


def _validate_program_ast(code_str: str) -> None:
    tree = ast.parse(code_str, mode="exec")
    transform_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transform_grid"]
    if not transform_defs:
        raise ValueError("Generated code must define transform_grid.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name for alias in node.names]
            else:
                module_names = [node.module or ""]

            for module_name in module_names:
                if module_name != "numpy":
                    raise ValueError(f"Import not allowed: {module_name}")

        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ValueError(f"Use of banned name: {node.id}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")


def _limited_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if name != "numpy":
        raise ImportError(f"Import of {name!r} is not allowed.")
    return __import__(name, globals_, locals_, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    names = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "reversed",
        "set",
        "slice",
        "sorted",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "ValueError",
        "TypeError",
        "IndexError",
    ]
    builtins_source = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    allowed = {name: builtins_source[name] for name in names}
    allowed["__import__"] = _limited_import
    return allowed


@contextmanager
def _time_limit(seconds: int) -> Iterable[None]:
    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Execution exceeded {seconds} seconds.")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _normalize_example(
    example: ARCExample | tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(example, ARCExample):
        return example.input_grid, example.output_grid
    return example


def evaluate_program(
    code_str: str,
    examples: Iterable[ARCExample | tuple[np.ndarray, np.ndarray]],
    *,
    timeout_seconds: int = 2,
) -> dict[str, Any]:
    """
    Execute generated code in a restricted namespace and evaluate it on examples.
    """
    result: dict[str, Any] = {
        "pass_all": False,
        "correct_count": 0,
        "total_examples": 0,
        "accuracy": 0.0,
        "error": None,
    }

    try:
        _validate_program_ast(code_str)
        compiled = compile(code_str, "<generated_program>", "exec")
        globals_dict = {"__builtins__": _safe_builtins(), "np": np, "numpy": np}
        locals_dict: dict[str, Any] = {}
        exec(compiled, globals_dict, locals_dict)
        transform_grid = locals_dict.get("transform_grid") or globals_dict.get("transform_grid")
        if not callable(transform_grid):
            raise ValueError("transform_grid is missing or not callable.")

        normalized_examples = list(examples)
        result["total_examples"] = len(normalized_examples)

        for example in normalized_examples:
            input_grid, output_grid = _normalize_example(example)
            if output_grid is None:
                raise ValueError("Evaluation examples must include output grids.")

            with _time_limit(timeout_seconds):
                predicted = transform_grid(np.array(input_grid, copy=True))

            predicted = np.array(predicted, dtype=int)
            if np.array_equal(predicted, output_grid):
                result["correct_count"] += 1

        if result["total_examples"] > 0:
            result["accuracy"] = result["correct_count"] / result["total_examples"]
        result["pass_all"] = result["correct_count"] == result["total_examples"]
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _run_program_on_test_examples(
    code_str: str,
    test_examples: Iterable[ARCExample],
    *,
    timeout_seconds: int = 2,
) -> list[dict[str, Any]]:
    _validate_program_ast(code_str)
    compiled = compile(code_str, "<generated_program>", "exec")
    globals_dict = {"__builtins__": _safe_builtins(), "np": np, "numpy": np}
    locals_dict: dict[str, Any] = {}
    exec(compiled, globals_dict, locals_dict)
    transform_grid = locals_dict.get("transform_grid") or globals_dict.get("transform_grid")
    if not callable(transform_grid):
        raise ValueError("transform_grid is missing or not callable.")

    outputs: list[dict[str, Any]] = []
    for example in test_examples:
        with _time_limit(timeout_seconds):
            predicted = transform_grid(np.array(example.input_grid, copy=True))
        predicted_array = np.array(predicted, dtype=int)
        item = {
            "example_number": example.example_number,
            "prediction": predicted_array.tolist(),
        }
        if example.output_grid is not None:
            item["matches_known_output"] = bool(np.array_equal(predicted_array, example.output_grid))
        outputs.append(item)
    return outputs


def solve_task(
    task_examples: dict[str, list[ARCExample]],
    *,
    num_hypotheses: int = 4,
    programs_per_hypothesis: int = 1,
    hypothesis_config: Optional[LLMConfig] = None,
    program_config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Run the full hypothesis -> program -> execution pipeline for one task.
    """
    train_examples = task_examples.get("train", [])
    test_examples = task_examples.get("test", [])
    if not train_examples:
        raise ValueError("Task must contain at least one training example.")

    task_text = format_task_for_prompt(train_examples)
    hypotheses = generate_hypotheses(
        task_text,
        num_hypotheses=num_hypotheses,
        config=hypothesis_config,
        client=client,
    )

    candidate_results: list[dict[str, Any]] = []
    best_candidate: Optional[dict[str, Any]] = None

    for hypothesis in hypotheses:
        for attempt_index in range(programs_per_hypothesis):
            code_str = generate_program(
                task_text,
                hypothesis,
                config=program_config,
                client=client,
                attempt_index=attempt_index,
            )
            evaluation = evaluate_program(code_str, train_examples)
            candidate = {
                "hypothesis": hypothesis,
                "program": code_str,
                "evaluation": evaluation,
            }
            candidate_results.append(candidate)

            if best_candidate is None:
                best_candidate = candidate
                continue

            current_eval = candidate["evaluation"]
            best_eval = best_candidate["evaluation"]
            current_score = (current_eval["pass_all"], current_eval["correct_count"], -len(candidate["program"]))
            best_score = (best_eval["pass_all"], best_eval["correct_count"], -len(best_candidate["program"]))
            if current_score > best_score:
                best_candidate = candidate

    test_predictions: list[dict[str, Any]] = []
    if best_candidate is not None and test_examples:
        try:
            test_predictions = _run_program_on_test_examples(best_candidate["program"], test_examples)
        except Exception as exc:  # noqa: BLE001
            test_predictions = [{"error": f"{type(exc).__name__}: {exc}"}]

    useful_hypotheses = sorted(
        {
            candidate["hypothesis"]
            for candidate in candidate_results
            if candidate["evaluation"]["pass_all"]
        }
    )

    return {
        "task_text": task_text,
        "hypotheses": hypotheses,
        "candidate_results": candidate_results,
        "best_program": best_candidate["program"] if best_candidate else None,
        "best_hypothesis": best_candidate["hypothesis"] if best_candidate else None,
        "best_evaluation": best_candidate["evaluation"] if best_candidate else None,
        "solved_train": bool(best_candidate and best_candidate["evaluation"]["pass_all"]),
        "useful_hypotheses": useful_hypotheses,
        "test_predictions": test_predictions,
    }


def run_batch_pipeline(
    tasks: dict[str, dict[str, list[ARCExample]]],
    *,
    num_hypotheses: int = 4,
    programs_per_hypothesis: int = 1,
    hypothesis_config: Optional[LLMConfig] = None,
    program_config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
    task_limit: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> dict[str, Any]:
    """
    Run the batch pipeline and return both the final dataset and full task traces.
    """
    dataset: list[dict[str, str]] = []
    traces: list[dict[str, Any]] = []
    processed = 0
    solved_tasks = 0

    for task_name in sorted(tasks):
        if task_limit is not None and processed >= task_limit:
            break

        processed += 1
        try:
            result = solve_task(
                tasks[task_name],
                num_hypotheses=num_hypotheses,
                programs_per_hypothesis=programs_per_hypothesis,
                hypothesis_config=hypothesis_config,
                program_config=program_config,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            traces.append(
                {
                    "task_name": task_name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if logger:
                logger(f"[{processed}] {task_name}: failed with {type(exc).__name__}: {exc}")
            continue

        traces.append(
            {
                "task_name": task_name,
                "status": "ok",
                "result": result,
            }
        )

        if result["useful_hypotheses"]:
            solved_tasks += 1
            for hypothesis in result["useful_hypotheses"]:
                dataset.append(
                    {
                        "task_name": task_name,
                        "input": result["task_text"],
                        "output": hypothesis,
                    }
                )

        if logger:
            logger(
                f"[{processed}] {task_name}: "
                f"{len(result['useful_hypotheses'])} useful hypotheses, "
                f"best_correct={result['best_evaluation']['correct_count'] if result['best_evaluation'] else 0}"
            )

    success_rate = solved_tasks / processed if processed else 0.0
    summary = {
        "processed": processed,
        "successful_tasks": solved_tasks,
        "dataset_examples": len(dataset),
        "success_rate": success_rate,
    }

    if logger:
        logger(
            "Summary: "
            f"processed={processed}, "
            f"successful_tasks={solved_tasks}, "
            f"dataset_examples={len(dataset)}, "
            f"success_rate={success_rate:.2%}"
        )

    return {
        "dataset": dataset,
        "traces": traces,
        "summary": summary,
    }


def build_hypothesis_dataset(
    tasks: dict[str, dict[str, list[ARCExample]]],
    *,
    num_hypotheses: int = 4,
    programs_per_hypothesis: int = 1,
    hypothesis_config: Optional[LLMConfig] = None,
    program_config: Optional[LLMConfig] = None,
    client: Optional[Any] = None,
    task_limit: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> list[dict[str, str]]:
    """
    Build a dataset of (task_text, useful_hypothesis) pairs.
    """
    batch_result = run_batch_pipeline(
        tasks,
        num_hypotheses=num_hypotheses,
        programs_per_hypothesis=programs_per_hypothesis,
        hypothesis_config=hypothesis_config,
        program_config=program_config,
        client=client,
        task_limit=task_limit,
        logger=logger,
    )
    return batch_result["dataset"]


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARC hypothesis -> program -> execution pipeline")
    parser.add_argument(
        "--csv-path",
        default="data/task_data/ARC_training_tasks.csv",
        help="Path to ARC CSV data.",
    )
    parser.add_argument(
        "--task-name",
        help="If provided, solve a single task instead of building a dataset.",
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to process when building a dataset.",
    )
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=4,
        help="Number of hypotheses to generate per task.",
    )
    parser.add_argument(
        "--programs-per-hypothesis",
        type=int,
        default=1,
        help="Number of program attempts to generate per hypothesis.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model to use for both hypothesis and program generation.",
    )
    parser.add_argument(
        "--output-path",
        help="Optional path to save JSON results.",
    )
    parser.add_argument(
        "--trace-output-path",
        help="Optional path to save full task traces, including all hypotheses, programs, evaluations, and errors.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    tasks = load_arc_tasks(args.csv_path)
    shared_config = LLMConfig(model=args.model)

    if args.task_name:
        if args.task_name not in tasks:
            raise SystemExit(f"Task {args.task_name!r} not found in {args.csv_path}")

        result = solve_task(
            tasks[args.task_name],
            num_hypotheses=args.num_hypotheses,
            programs_per_hypothesis=args.programs_per_hypothesis,
            hypothesis_config=shared_config,
            program_config=LLMConfig(model=args.model, temperature=0.2, max_completion_tokens=1200),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output_path:
            _write_json(args.output_path, result)
        return

    batch_result = run_batch_pipeline(
        tasks,
        num_hypotheses=args.num_hypotheses,
        programs_per_hypothesis=args.programs_per_hypothesis,
        hypothesis_config=shared_config,
        program_config=LLMConfig(model=args.model, temperature=0.2, max_completion_tokens=1200),
        task_limit=args.task_limit,
    )
    dataset = batch_result["dataset"]
    print(json.dumps(dataset[:3], ensure_ascii=False, indent=2))
    if args.output_path:
        _write_json(args.output_path, dataset)
    if args.trace_output_path:
        _write_json(args.trace_output_path, batch_result)


if __name__ == "__main__":
    main()
