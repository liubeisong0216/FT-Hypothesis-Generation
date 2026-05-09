# Part 1 Experiment Summary

This document summarizes the current Part 1 ARC experiments. All four experiments use the same 100-task random sample:

```text
data/task_data/train_random100_seed42.txt
```

which is sampled from the original ARC training task CSV:

```text
data/task_data/ARC_training_tasks.csv
```

Part 1 is only evaluating ARC behavior through prompting, program generation, and execution. The fine-tuning pipeline in `scripts/part2_ft/` is not used for these experiments.

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

Set the OpenAI API key:

```bash
export OPENAI_API_KEY=...
```

The baseline scripts import helpers from `arc_pipeline.py`, so set `PYTHONPATH` before running:

```bash
export PYTHONPATH="$PWD/scripts/part1_hypogeneration:$PYTHONPATH"
```


## Experiment 1: Direct Output Grid Baseline

Goal: ask `gpt-5.4-mini` to directly generate candidate output grids for each test input. This experiment does not generate hypotheses or programs.

Configuration:

- Hypothesis generation: none
- Program generation: none
- Output generation model: `gpt-5.4-mini`
- Candidates per test example: `6`
- Output directory: `outputs/random100_baseline/`

Run:

```bash
python scripts/part1_hypogeneration/baseline_direct_output.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --model gpt-5.4-mini \
  --num-candidates 6 \
  --output-json outputs/random100_baseline/trace.json
```

Saved result summary:

- Processed tasks: `100`
- Scored test examples: `105`
- Top-1 test example accuracy: `12 / 105 = 11.43%`
- Top-6 test example accuracy: `16 / 105 = 15.24%`
- Top-1 task success rate: `8 / 100 = 8.00%`
- Top-6 task success rate: `12 / 100 = 12.00%`

## Experiment 2: Direct Program Baseline

Goal: ask `gpt-5.4-mini` to directly generate six candidate Python programs per task, evaluate those programs on the training examples, select the best program by train performance, then score that selected program on the test examples.

Configuration:

- Hypothesis generation: none
- Program generation model: `gpt-5.4-mini`
- Programs per task: `6`
- Selection criterion: best training-set execution result
- Test metric: selected program evaluated on known test outputs
- Output directory: `outputs/random100_program/`

Run:

```bash
python scripts/part1_hypogeneration/baseline_direct_programs.py \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --task-list-path data/task_data/train_random100_seed42.txt \
  --model gpt-5.4-mini \
  --num-programs 6 \
  --output-json outputs/random100_program/trace.json
```

Saved result summary:

- Processed tasks: `100`
- Requested programs per task: `6`
- Test examples: `105`
- Selected-program test example accuracy: `34 / 105 = 32.38%`
- Selected-program task success rate: `30 / 100 = 30.00%`
- Top-6 and selected-program metrics are the same in the saved summary for this run.

## Experiment 3: Hypothesis Plus Program

Goal: ask `gpt-5.4-mini` to generate three natural-language hypotheses per task. For each hypothesis, ask `gpt-5.4-mini` to generate two candidate programs. Evaluate all programs on the training examples, select the best one, and score it on the test examples.

Configuration:

- Hypothesis generation model: `gpt-5.4-mini`
- Hypotheses per task: `3`
- Program generation model: `gpt-5.4-mini`
- Programs per hypothesis: `2`
- Total program attempts per task: up to `6`
- Selection criterion: best training-set execution result
- Test metric: selected best program evaluated on known test outputs
- Output directory: `outputs/random100_hypo_program/`

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

Saved result summary:

- Processed tasks: `100`
- Tasks solved on train: `26 / 100 = 26.00%`
- Verified hypothesis examples harvested: `52`
- Test examples: `105`
- Test example accuracy: `31 / 105 = 29.52%`
- Test task success rate: `27 / 100 = 27.00%`

There is also a second saved run at `outputs/random100_hypo_program2/` with similar settings:

- Tasks solved on train: `24 / 100 = 24.00%`
- Verified hypothesis examples harvested: `46`
- Test example accuracy: `27 / 103 = 26.21%`
- Test task success rate: `25 / 98 = 25.51%`

## Experiment 4: High-Quality Hypothesis Plus Program

Goal: use `gpt-5.5` to generate higher-quality hypotheses, then still use `gpt-5.4-mini` for program generation. As in Experiment 3, each task gets three hypotheses and two program attempts per hypothesis. The best program is selected by training-set execution and scored on test examples.

Configuration:

- Hypothesis generation model: `gpt-5.5`
- Hypotheses per task: `3`
- Program generation model: `gpt-5.4-mini`
- Programs per hypothesis: `2`
- Total program attempts per task: up to `6`
- Selection criterion: best training-set execution result
- Test metric: selected best program evaluated on known test outputs
- Output directory: `outputs/random100_goodhypo_program/`

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

Saved result summary:

- Processed tasks: `100`
- Tasks solved on train: `85 / 100 = 85.00%`
- Verified hypothesis examples harvested: `224`
- Test examples: `104`
- Test example accuracy: `87 / 104 = 83.65%`
- Test task success rate: `82 / 99 = 82.83%`

## Inspecting Results

For Experiment 3 and Experiment 4 traces, use `analyze_trace.py`:

```bash
python scripts/part1_hypogeneration/analyze_trace.py \
  --trace-path outputs/random100_goodhypo_program/trace.json \
  --list-tasks
```

Inspect one task in detail:

```bash
python scripts/part1_hypogeneration/analyze_trace.py \
  --trace-path outputs/random100_goodhypo_program/trace.json \
  --task-name 00d62c1b.json \
  --hide-programs
```

Export a compact text report:

```bash
python scripts/part1_hypogeneration/analyze_trace.py \
  --trace-path outputs/random100_goodhypo_program/trace.json \
  --all-task-details \
  --hide-programs \
  --max-candidates 2 \
  --output-path outputs/random100_goodhypo_program/analysis.txt
```

The direct-output and direct-program baselines use separate trace schemas, so inspect their JSON files directly or use the summaries in their output directories:

```text
outputs/random100_baseline/readme.md
outputs/random100_program/readme.md
```

## Main Takeaway

On the same 100-task sample, direct output generation is weakest, direct program generation improves substantially, same-model hypothesis-plus-program is similar or slightly worse than direct program generation, and high-quality `gpt-5.5` hypotheses paired with `gpt-5.4-mini` program generation gives the strongest result by a large margin.
