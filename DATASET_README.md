# ARC Hypothesis Generation Datasets

This Google Drive folder contains two dataset directories used for our ARC hypothesis generation experiments:

- `raw_data/`: original ARC task data in CSV format.
- `ft_dataset/`: verified hypothesis datasets prepared for fine-tuning.

## Directory Structure

```text
.
├── raw_data/
│   ├── ARC_training_tasks.csv
│   └── ARC_evaluation_tasks.csv
└── ft_dataset/
    ├── all400_verified_hypotheses.csv
    ├── all400_verified_hypotheses.json
    ├── all400_verified_hypotheses_chat.jsonl
    ├── all400_verified_hypotheses_summary.json
    ├── all400_verified_hypotheses_one_per_task.json
    ├── all400_verified_hypotheses_one_per_task_chat.jsonl
    └── all400_verified_hypotheses_one_per_task_summary.json
```

## `raw_data/`

This folder contains the original ARC task data used as input to the hypothesis generation pipeline.

### `ARC_training_tasks.csv`

Original ARC training tasks.

Each row corresponds to one input-output example from an ARC task. Important columns include:

- `task_name`: ARC task id, for example `00d62c1b.json`.
- `example_type`: whether the row is a train or test example.
- `example_number`: index of the example within the task.
- `input_height`, `input_width`: input grid dimensions.
- `output_height`, `output_width`: output grid dimensions.
- `input_grid`: serialized ARC input grid.
- `output_grid`: serialized ARC output grid.

This file was used to run GPT-based hypothesis generation and verification.

### `ARC_evaluation_tasks.csv`

Original ARC evaluation tasks in the same CSV schema as `ARC_training_tasks.csv`.

This file is kept as raw source data. The current fine-tuning dataset in `ft_dataset/` was generated from verified hypotheses on the training-task runs, not from this evaluation CSV.

## `ft_dataset/`

This folder contains fine-tuning datasets built from GPT-generated ARC hypotheses that were verified by the pipeline.

A hypothesis is included here only if it was in `useful_hypotheses` in the trace output. In our pipeline, that means the hypothesis produced a program candidate that passed all available training examples for that ARC task.

Current coverage:

- 400 ARC training tasks were attempted.
- 119 tasks have at least one verified hypothesis.
- 323 total verified hypotheses were collected after deduplication.

The source runs were:

- `run_1_50_1`
- `run_1-50_2`
- `run_51_150`
- `run_151_250`
- `run_251_350`
- `run_351_400`

### `all400_verified_hypotheses.json`

Main full dataset.

Contains:

- `summary`: metadata and counts.
- `records`: all verified hypothesis examples.

Each record has:

- `task_name`: ARC task id.
- `input`: formatted ARC task examples shown to the model.
- `output`: verified hypothesis text.
- `messages`: chat-format user/assistant pair for fine-tuning.
- `source_trace`: trace file where the hypothesis came from.

Use this file when you want the complete metadata and full records.

### `all400_verified_hypotheses_chat.jsonl`

Chat fine-tuning version of the full dataset.

Each line is:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

This is the most convenient file for chat-style fine-tuning.

Stats:

- 323 examples.
- 119 unique ARC tasks.
- Multiple verified hypotheses may appear for the same task.

### `all400_verified_hypotheses.csv`

Human-readable table for inspection.

Columns:

- `task_name`
- `hypothesis`
- `source_trace`

Use this for quick browsing, filtering, or manual review in Google Sheets.

### `all400_verified_hypotheses_summary.json`

Summary metadata for the full dataset.

Includes:

- number of source trace files.
- total verified hypotheses.
- number of covered tasks.
- number of hypotheses per task.
- number of examples contributed by each source trace.

### `all400_verified_hypotheses_one_per_task.json`

Reduced dataset with at most one verified hypothesis per ARC task.

Stats:

- 119 examples.
- 119 unique ARC tasks.
- `max_per_task = 1`.

Use this version if you want to avoid over-weighting tasks that happened to have many verified paraphrases.

### `all400_verified_hypotheses_one_per_task_chat.jsonl`

Chat fine-tuning version of the one-hypothesis-per-task dataset.

Each line contains one `messages` training example.

Use this for a balanced fine-tuning baseline where each solved ARC task contributes equally.

### `all400_verified_hypotheses_one_per_task_summary.json`

Summary metadata for the one-hypothesis-per-task dataset.

## Recommended Usage

For fine-tuning experiments:

1. Start with `all400_verified_hypotheses_chat.jsonl` if you want the largest verified dataset.
2. Also try `all400_verified_hypotheses_one_per_task_chat.jsonl` as a balanced baseline.
3. Compare validation behavior carefully, because only 119 out of 400 attempted tasks currently have verified hypotheses.

For data inspection:

1. Use `all400_verified_hypotheses.csv` for quick manual review.
2. Use `all400_verified_hypotheses_summary.json` to check per-task counts.

## Important Notes

- The fine-tuning dataset is not raw ARC ground truth. It is derived data: ARC task examples paired with verified natural-language hypotheses.
- "Verified" means the hypothesis had at least one generated program that passed all training examples for that task.
- A verified hypothesis can still be imperfect as a natural-language explanation, because verification is mediated through generated code.
- Tasks with no verified hypothesis are not included in `ft_dataset/`.
