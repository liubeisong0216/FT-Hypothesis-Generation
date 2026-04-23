# FT-Hypothesis-Generation Project Handoff

## Overview

This project builds a verified training dataset for **ARC hypothesis generation**, then uses that dataset to fine-tune a language model.

The high-level pipeline is:

1. Take ARC training tasks from CSV.
2. Ask an LLM to generate multiple natural-language hypotheses for each task.
3. Ask an LLM to turn each hypothesis into Python code implementing `transform_grid(...)`.
4. Execute the generated code on ARC training examples in a restricted sandbox.
5. Keep only hypotheses whose generated programs **pass all train examples**.
6. Aggregate those verified `(task_text, hypothesis)` pairs into a fine-tuning dataset.
7. Fine-tune a smaller instruction model on that dataset.

The eventual goal is:

- Input: ARC task examples
- Output: a structured, natural-language transformation hypothesis in a consistent 3-line format


## Main Research Goal

This project is **not primarily an ARC solver benchmark**.
It is a **data generation pipeline** for creating supervised training examples for a hypothesis-generation model.

So the key output is not just solve rate, but:

- how many **verified hypotheses** can be harvested
- how clean those hypotheses are
- how many ARC tasks are covered by verified examples


## Repository Structure

### Core pipeline

- `arc_pipeline.py`
  Main hypothesis -> program -> execution pipeline.

- `prompts/hypothesis_generation.prompt`
  Prompt template for generating natural-language hypotheses.

- `prompts/program_generation.prompt`
  Prompt template for generating Python `transform_grid(...)` code from a hypothesis.

### Analysis / utilities

- `analyze_trace.py`
  Reads saved trace JSON files and summarizes candidates / results.

- `visualize_task.py`
  Visualizes ARC task examples.

- `build_training_dataset.py`
  Merges multiple trace files/directories and extracts only verified hypotheses.

- `prepare_finetune_data.py`
  Converts merged verified data into train/valid JSONL files for fine-tuning.

### Fine-tuning

- `train_mlx_lora.py`
  Local Apple Silicon training entrypoint using `mlx-lm`.

- `train_hf_qlora.py`
  GPU training script using `transformers + trl + peft + bitsandbytes`.

- `requirements-finetune-gpu.txt`
  Python dependencies for GPU QLoRA training.

- `FINETUNE.md`
  Short fine-tuning instructions.


## ARC Data

Main task files:

- `data/task_data/ARC_training_tasks.csv`
- `data/task_data/ARC_evaluation_tasks.csv`

Current dataset-building work uses the **training CSV only**.

Important:

- `ARC_training_tasks.csv` is for generating fine-tune data.
- `ARC_evaluation_tasks.csv` should be reserved for later evaluation/generalization.


## How `arc_pipeline.py` Works

For each ARC task:

1. Load train examples.
2. Format them into plain text:
   - `Case 0`
   - `Input`
   - `Output`
3. Generate `N` candidate hypotheses.
4. For each hypothesis, generate `K` candidate programs.
5. Evaluate each generated program against all train examples.
6. If a program passes all train examples, mark that hypothesis as useful.

Saved outputs include:

- candidate hypotheses
- candidate programs
- evaluation results
- best program / best hypothesis
- list of `useful_hypotheses`


## Important Fixes Already Made

### 1. Execution sandbox bug fix

Originally, many generated programs failed with:

- `ImportError: Import of 'numpy._core._methods' is not allowed.`

This was not a reasoning failure.
It was an execution sandbox issue.

We fixed `arc_pipeline.py` so that:

- `numpy` internal submodule imports are allowed
- a few safe Python builtins are also allowed:
  - `next`
  - `round`
  - `map`
  - `filter`
  - `isinstance`

This materially improved verified candidate yield.

### 2. Fine-tune data pipeline added

We added scripts to:

- merge verified hypotheses from multiple runs
- split into train/validation
- support both:
  - local MLX smoke tests on Mac
  - GPU QLoRA training on server


## Current Output Runs

The following run directories are currently important:

- `outputs/run1_1-50`
- `outputs/run2_1-50`
- `outputs/run_51_150`

Each contains:

- `trainXX_dataset.json`
- `trainXX_trace.json`

The trace files are the important source of truth because they contain:

- `task_text`
- `candidate_results`
- `useful_hypotheses`


## Current Merged Verified Dataset

The merged verified dataset has already been built from:

- `outputs/run1_1-50`
- `outputs/run2_1-50`
- `outputs/run_51_150`

Output files:

- `outputs/ft_dataset/current_verified_hypotheses.json`
- `outputs/ft_dataset/current_verified_hypotheses_chat.jsonl`
- `outputs/ft_dataset/current_verified_hypotheses_summary.json`

Current summary:

- 143 verified hypothesis examples
- 50 unique ARC tasks covered

Source contribution:

- `run1_1-50`: 30 examples
- `run2_1-50`: 44 examples
- `run_51_150`: 69 examples


## How the Fine-Tune Dataset Was Built

Command used:

```bash
python build_training_dataset.py \
  outputs/run_51_150 \
  outputs/run1_1-50 \
  outputs/run2_1-50 \
  --output-json outputs/ft_dataset/current_verified_hypotheses.json \
  --output-jsonl outputs/ft_dataset/current_verified_hypotheses_chat.jsonl \
  --summary-output outputs/ft_dataset/current_verified_hypotheses_summary.json
```

This script:

- reads one or more trace directories/files
- extracts `useful_hypotheses`
- deduplicates by `(task_name, hypothesis)`
- emits merged records with metadata


## Fine-Tune Split Files

The merged verified dataset was then split into train/valid files.

Command used:

```bash
python prepare_finetune_data.py \
  --input-json outputs/ft_dataset/current_verified_hypotheses.json \
  --output-dir outputs/ft_ready/current \
  --valid-ratio 0.1 \
  --seed 42
```

Output files:

- `outputs/ft_ready/current/train.jsonl`
- `outputs/ft_ready/current/valid.jsonl`
- `outputs/ft_ready/current/train_text.jsonl`
- `outputs/ft_ready/current/valid_text.jsonl`
- `outputs/ft_ready/current/manifest.json`

Current split summary:

- 130 train examples
- 13 valid examples
- split by task, not by individual row

Why there are two formats:

- `train.jsonl` / `valid.jsonl`
  - prompt-completion format
  - intended for GPU training with TRL/PEFT

- `train_text.jsonl` / `valid_text.jsonl`
  - plain text format
  - intended for safer local MLX smoke tests on Mac


## Local Mac Fine-Tune Smoke Test

The local machine is:

- Mac
- Apple M2 Pro

The goal locally was only to **verify the training path**, not to do full training.

### What happened

Initial local MLX training failed for two reasons:

1. chat-template incompatibility with the local data format
2. Metal resource issues caused by very long samples

Both were handled by:

- switching local MLX runs to text-format data
- adding filtering options for:
  - max text length
  - number of shortest samples used for local smoke tests

### Smoke test that succeeded

Successful local command:

```bash
python train_mlx_lora.py \
  --data-dir outputs/ft_ready/current \
  --use-text-data \
  --smoke-size 16 \
  --max-text-chars 2200 \
  --adapter-path outputs/ft_runs/mlx_qwen25_3b \
  --iters 10 \
  --batch-size 1 \
  --num-layers 4 \
  --grad-checkpoint
```

Successful result:

- training started normally
- validation loss decreased from `0.901` to `0.623`
- final adapter saved to:
  - `outputs/ft_runs/mlx_qwen25_3b/adapters.safetensors`

This means:

- the fine-tuning data is valid
- the local training code path works
- the project is ready to move to GPU training on server


## Recommended Server Training Path

The main server-side training script is:

- `train_hf_qlora.py`

It uses:

- `transformers`
- `trl`
- `peft`
- `bitsandbytes`

This is the intended path for real GPU fine-tuning.

### Install dependencies

```bash
pip install -r requirements-finetune-gpu.txt
```

### Main GPU training command

```bash
python train_hf_qlora.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file outputs/ft_ready/current/train.jsonl \
  --valid-file outputs/ft_ready/current/valid.jsonl \
  --output-dir outputs/ft_runs/hf_qwen25_3b_qlora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 4096 \
  --gradient-checkpointing
```

If the GPU supports BF16:

```bash
--use-bf16
```


## Why Qwen2.5-3B-Instruct

The current chosen fine-tune base model is:

- `Qwen/Qwen2.5-3B-Instruct`

Reasons:

- small enough to be practical
- strong instruction-following behavior
- good for structured natural-language outputs
- suitable for this task:
  - ARC task examples in
  - 3-line formatted hypothesis out

This project is training a **hypothesis generator**, not a code model.


## What Is Already Done

- ARC pipeline implemented
- execution sandbox improved
- verified hypotheses harvested from current runs
- merged fine-tune dataset created
- train/valid split created
- local fine-tune smoke test on Apple Silicon completed successfully
- GPU QLoRA training script added


## What Is Not Yet Done

### 1. Full GPU fine-tune run

The next main step is to run `train_hf_qlora.py` on the GPU server.

### 2. Fine-tuned model evaluation

There is now a dedicated offline evaluation script:

- `evaluate_hypothesis_model.py`

This script:

- loads a base model plus optional PEFT adapter
- evaluates on train / valid / all task splits
- generates one hypothesis per task prompt
- compares that hypothesis to the gold verified hypotheses for the task

Current metrics:

- valid-format rate
- raw exact-match rate
- normalized exact-match rate
- average best token-F1

What still needs to be added later if desired:

- deeper evaluation variants, for example using different program-generation models or larger task batches


## Current Open Questions / Next Steps

1. Run the full GPU fine-tune on the server.
2. Decide whether to keep all 143 verified hypotheses or create a cleaner subset:
   - for example `max 1-2 hypotheses per task`
3. Continue generating more verified hypotheses from later ARC task batches.
4. Run both offline and end-to-end evaluation scripts for the fine-tuned model.
5. Possibly compare:
   - base model vs fine-tuned model
   - exact hypothesis quality
   - downstream verified hypothesis yield


## Useful Commands Recap

### Merge verified hypotheses

```bash
python build_training_dataset.py \
  outputs/run_51_150 \
  outputs/run1_1-50 \
  outputs/run2_1-50 \
  --output-json outputs/ft_dataset/current_verified_hypotheses.json \
  --output-jsonl outputs/ft_dataset/current_verified_hypotheses_chat.jsonl \
  --summary-output outputs/ft_dataset/current_verified_hypotheses_summary.json
```

### Prepare train/valid split

```bash
python prepare_finetune_data.py \
  --input-json outputs/ft_dataset/current_verified_hypotheses.json \
  --output-dir outputs/ft_ready/current \
  --valid-ratio 0.1 \
  --seed 42
```

### Local Mac smoke test

```bash
python train_mlx_lora.py \
  --data-dir outputs/ft_ready/current \
  --use-text-data \
  --smoke-size 16 \
  --max-text-chars 2200 \
  --adapter-path outputs/ft_runs/mlx_qwen25_3b \
  --iters 10 \
  --batch-size 1 \
  --num-layers 4 \
  --grad-checkpoint
```

### GPU QLoRA training

```bash
python train_hf_qlora.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file outputs/ft_ready/current/train.jsonl \
  --valid-file outputs/ft_ready/current/valid.jsonl \
  --output-dir outputs/ft_runs/hf_qwen25_3b_qlora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 4096 \
  --gradient-checkpointing
```

### Evaluate the fine-tuned model on held-out validation tasks

```bash
python evaluate_hypothesis_model.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path outputs/ft_runs/hf_qwen25_3b_qlora \
  --dataset-json outputs/ft_dataset/current_verified_hypotheses.json \
  --manifest-path outputs/ft_ready/current/manifest.json \
  --split valid \
  --load-in-4bit \
  --output-json outputs/evals/hf_qwen25_3b_valid_summary.json \
  --predictions-jsonl outputs/evals/hf_qwen25_3b_valid_predictions.jsonl
```

### End-to-end ARC evaluation with the fine-tuned hypothesis model

```bash
python evaluate_end_to_end_arc.py \
  --hypothesis-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path outputs/ft_runs/hf_qwen25_3b_qlora \
  --csv-path data/task_data/ARC_training_tasks.csv \
  --manifest-path outputs/ft_ready/current/manifest.json \
  --split valid \
  --num-hypotheses 4 \
  --programs-per-hypothesis 2 \
  --program-model gpt-5.4-mini \
  --load-in-4bit \
  --output-json outputs/evals/end_to_end_valid_arc_eval.json
```


## One-Sentence Summary

This project automatically generates **verified ARC transformation hypotheses**, builds a supervised fine-tuning dataset from them, and is now ready to move from local smoke-tested training to full GPU QLoRA fine-tuning on `Qwen2.5-3B-Instruct`.
