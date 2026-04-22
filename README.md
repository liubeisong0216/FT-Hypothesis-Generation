# FT-Hypothesis-Generation

Minimal ARC hypothesis -> program -> execution pipeline.

## Files

- `arc_pipeline.py`: generate hypotheses, generate programs, evaluate on training examples, build dataset, save traces
- `visualize_task.py`: visualize all input/output examples for one `task_name`
- `analyze_trace.py`: inspect saved trace files and export readable analysis text

## Data

Main CSV files:

- `data/task_data/ARC_training_tasks.csv`
- `data/task_data/ARC_evaluation_tasks.csv`

Each task is grouped by `task_name`. The code uses only training examples for hypothesis/program generation and evaluation.

## Requirements

- Python 3.10+
- `numpy`
- `openai`
- `matplotlib` for visualization
- `OPENAI_API_KEY` set in your environment

Optional:

- `OPENAI_MODEL` if you want a default model other than `gpt-4o`

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Run The Pipeline

Run one task:

```bash
python arc_pipeline.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-name 00d62c1b.json \
  --num-hypotheses 2 \
  --programs-per-hypothesis 1 \
  --model gpt-5.4-mini \
  --output-path outputs/00d62c1b_full.json
```

Run a small batch and save both dataset and trace:

```bash
python arc_pipeline.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-limit 3 \
  --num-hypotheses 2 \
  --programs-per-hypothesis 1 \
  --model gpt-5.4-mini \
  --output-path outputs/train3_dataset.json \
  --trace-output-path outputs/train3_trace.json
```

### `arc_pipeline.py` parameters

- `--csv-path`: path to ARC CSV file
- `--task-name`: solve one specific task; if omitted, run batch mode
- `--task-limit`: max number of tasks in batch mode
- `--num-hypotheses`: number of candidate hypotheses generated per task
- `--programs-per-hypothesis`: number of code candidates generated per hypothesis
- `--model`: OpenAI model name, for example `gpt-5.4-mini`
- `--output-path`: save JSON result
- `--trace-output-path`: in batch mode, save full trace including hypotheses, programs, evaluations, and errors

### Output files

- Single-task mode:
  - full result JSON with `hypotheses`, `candidate_results`, `best_program`, `best_hypothesis`, `best_evaluation`
- Batch mode:
  - dataset JSON with `(task_name, input, output)`
  - trace JSON with all candidate hypotheses/programs and evaluation results

## 2. Visualize A Task

Show all examples for one task:

```bash
python visualize_task.py \
  --task-name 00d62c1b.json \
  --csv-path data/task_data/ARC_training_tasks.csv
```

Save visualization to file:

```bash
python visualize_task.py \
  --task-name 00d62c1b.json \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --save-path outputs/00d62c1b.png \
  --no-show
```

### `visualize_task.py` parameters

- `--task-name`: task to visualize
- `--csv-path`: path to ARC CSV file
- `--save-path`: optional image output path
- `--no-show`: save only, do not open an interactive window

## 3. Analyze Trace Files

List all tasks in a trace:

```bash
python analyze_trace.py \
  --trace-path outputs/train3_trace.json \
  --list-tasks
```

Analyze one task and save to txt:

```bash
python analyze_trace.py \
  --trace-path outputs/train3_trace.json \
  --task-name 007bbfb7.json \
  --output-path outputs/007bbfb7_analysis.txt
```

Export all task details to one txt:

```bash
python analyze_trace.py \
  --trace-path outputs/train3_trace.json \
  --all-task-details \
  --hide-programs \
  --max-candidates 2 \
  --output-path outputs/train3_all_tasks_analysis.txt
```

### `analyze_trace.py` parameters

- `--trace-path`: path to trace JSON file
- `--task-name`: show details for one task
- `--list-tasks`: show summary table for all tasks
- `--all-task-details`: show detailed analysis for every task
- `--hide-programs`: do not print generated code
- `--show-task-text`: include the full formatted task prompt
- `--max-candidates`: limit number of candidate results shown
- `--output-path`: save analysis text to file

## Suggested Workflow

1. Run a small batch with `arc_pipeline.py`
2. Save `--trace-output-path`
3. Inspect failures with `analyze_trace.py`
4. Visualize interesting tasks with `visualize_task.py`
