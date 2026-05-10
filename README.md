# Part 1 Experiment Overview

This repository currently focuses on Part 1: ARC-style hypothesis generation, program generation, and execution-based evaluation. The fine-tuning pipeline in `scripts/part2_ft/` is not used for these experiments.

All experiments use the same 100-task random sample:

```text
data/task_data/train_random100_seed42.txt
```

The sampled tasks come from:

```text
data/task_data/ARC_training_tasks.csv
```

Human-written hints and difficulty labels are stored in:

```text
data/task_data/train_hints_with_difficulty.csv
```

## Setup

Run commands from the repository root:

```bash
cd /Users/liubeisong/Desktop/2026_Spring/CCM/FT-Hypothesis-Generation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Part 1 dependencies:

```bash
pip install -r requirements.txt
```

Set the API key:

```bash
export OPENAI_API_KEY=...
```

The baseline scripts import helpers from `arc_pipeline.py`, so set `PYTHONPATH` before running them:

```bash
export PYTHONPATH="$PWD/scripts/part1_hypogeneration:$PYTHONPATH"
```

## Experiment 0: Direct Output Grid

This baseline asks `gpt-5.4-mini` to directly generate candidate output grids for each test input. It does not generate hypotheses or programs.

Configuration:

- Hypothesis generation: none
- Program generation: none
- Output generation model: `gpt-5.4-mini`
- Candidates per test example: `6`
- Final output directory: `final_outputs/0_output_grid/`

Run:

```bash
python scripts/part1_hypogeneration/baseline_direct_output.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --model gpt-5.4-mini \
  --num-candidates 6 \
  --output-json outputs/random100_baseline/trace.json
```

## Experiment 1: Program Only

This baseline asks `gpt-5.4-mini` to directly generate candidate Python programs. Each program defines a grid transformation, is evaluated on the training examples, and the best training-set program is selected for test evaluation.

Configuration:

- Hypothesis generation: none
- Program generation model: `gpt-5.4-mini`
- Programs per task: `6`
- Selection criterion: best training-set execution result
- Final output directory: `final_outputs/1_program_only/`

Run:

```bash
python scripts/part1_hypogeneration/baseline_direct_programs.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --model gpt-5.4-mini \
  --num-programs 6 \
  --output-json outputs/random100_program/trace.json
```

## Experiment 2: Hypothesis + Program

This experiment first asks `gpt-5.4-mini` to generate natural-language hypotheses. For each hypothesis, `gpt-5.4-mini` generates candidate programs. All programs are evaluated on the training examples, and the best program is selected for test evaluation.

Configuration:

- Hypothesis generation model: `gpt-5.4-mini`
- Hypotheses per task: `3`
- Program generation model: `gpt-5.4-mini`
- Programs per hypothesis: `2`
- Total program attempts per task: up to `6`
- Selection criterion: best training-set execution result
- Final output directory: `final_outputs/2_hypo_program/`

Run:

```bash
python scripts/part1_hypogeneration/arc_pipeline.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --num-hypotheses 3 \
  --programs-per-hypothesis 2 \
  --model gpt-5.4-mini \
  --output-path outputs/random100_hypo_program/dataset.json \
  --trace-output-path outputs/random100_hypo_program/trace.json
```

## Experiment 3: Hint + Hypothesis + Program

This experiment is the same as Experiment 2, except each task includes a human-written hint in the hypothesis-generation prompt. The hint is intended to guide the model toward the core transformation rule.

Configuration:

- Hint source: `data/task_data/train_hints_with_difficulty.csv`
- Hypothesis prompt: `prompts/hypothesis_generation_with_hint.prompt`
- Hypothesis generation model: `gpt-5.4-mini`
- Hypotheses per task: `3`
- Program generation model: `gpt-5.4-mini`
- Programs per hypothesis: `2`
- Total program attempts per task: up to `6`
- Selection criterion: best training-set execution result
- Final output directory: `final_outputs/3_hint_hypo_program/`

Run:

```bash
python scripts/part1_hypogeneration/arc_pipeline.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --hint-csv-path data/task_data/train_hints_with_difficulty.csv \
  --hypothesis-prompt-path prompts/hypothesis_generation_with_hint.prompt \
  --num-hypotheses 3 \
  --programs-per-hypothesis 2 \
  --model gpt-5.4-mini \
  --output-path outputs/random100_hint_hypo_program/dataset.json \
  --trace-output-path outputs/random100_hint_hypo_program/trace.json
```

## Experiment 4: High-Quality Hypothesis + Program

This experiment uses `gpt-5.5` to generate higher-quality hypotheses, then still uses `gpt-5.4-mini` for program generation. The rest of the pipeline matches Experiment 2: generate programs, evaluate on training examples, select the best program, and evaluate on test examples.

Configuration:

- Hypothesis generation model: `gpt-5.5`
- Hypotheses per task: `3`
- Program generation model: `gpt-5.4-mini`
- Programs per hypothesis: `2`
- Total program attempts per task: up to `6`
- Selection criterion: best training-set execution result
- Final output directory: `final_outputs/4_good_hypo_program/`

Run:

```bash
python scripts/part1_hypogeneration/arc_pipeline.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --num-hypotheses 3 \
  --programs-per-hypothesis 2 \
  --hypothesis-model gpt-5.5 \
  --program-model gpt-5.4-mini \
  --output-path outputs/random100_goodhypo_program/dataset.json \
  --trace-output-path outputs/random100_goodhypo_program/trace.json
```

## Inspecting And Exporting Results

Each final experiment folder contains a local `readme.md`, `trace.json`, and `results.csv`:

```text
final_outputs/
  0_output_grid/
  1_program_only/
  2_hypo_program/
  3_hint_hypo_program/
  4_good_hypo_program/
```

Use the unified trace-to-CSV script to regenerate CSV views:

```bash
python scripts/part1_hypogeneration/trace_to_csv.py \
  --trace-path final_outputs/4_good_hypo_program/trace.json \
  --output-csv final_outputs/4_good_hypo_program/results.csv \
  --view test
```

Use `--view candidates` to inspect generated hypotheses and candidate programs.
