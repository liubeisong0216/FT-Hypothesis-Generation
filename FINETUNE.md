# Fine-Tuning

This repo now includes a minimal finetuning path for:

- local smoke tests on Apple Silicon with `mlx-lm`
- GPU QLoRA training with `transformers` + `trl` + `peft`

## 1. Prepare the Dataset

First build the merged verified hypothesis dataset, then split it into train/valid files:

```bash
python prepare_finetune_data.py \
  --input-json outputs/ft_dataset/current_verified_hypotheses.json \
  --output-dir outputs/ft_ready/current \
  --valid-ratio 0.1 \
  --seed 42
```

This creates:

- `outputs/ft_ready/current/train.jsonl`
- `outputs/ft_ready/current/valid.jsonl`
- `outputs/ft_ready/current/manifest.json`

The JSONL files use prompt-completion format:

```json
{"prompt": "...", "completion": "..."}
```

This format works for both MLX-LM and TRL SFT.

## 2. Local Apple Silicon Smoke Test

Install MLX-LM training dependencies:

```bash
pip install "mlx-lm[train]"
```

Print the resolved command first:

```bash
python train_mlx_lora.py --print-only
```

Then run a short local training smoke test:

```bash
python train_mlx_lora.py \
  --data-dir outputs/ft_ready/current \
  --use-text-data \
  --adapter-path outputs/ft_runs/mlx_qwen25_3b \
  --iters 50 \
  --batch-size 1 \
  --num-layers 8 \
  --grad-checkpoint
```

Notes:

- This local path is meant to verify the data and training loop.
- For Apple Silicon smoke tests, `--use-text-data` is the safest option because it avoids chat-template issues.
- If Metal runs out of resources, shrink the run with `--smoke-size`, `--max-text-chars`, fewer layers, and fewer iterations.
- If `--model` points to a quantized MLX model, `mlx-lm` will use QLoRA automatically.
- If `--model` points to a normal model repo, this will run regular LoRA.

## 3. GPU QLoRA Training

Install GPU training dependencies:

```bash
pip install -r requirements-finetune-gpu.txt
```

Run training:

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

If your GPU supports BF16, add:

```bash
--use-bf16
```

## 4. Recommended Workflow

1. Rebuild the merged verified dataset after each new ARC run.
2. Refresh the train/valid split with `prepare_finetune_data.py`.
3. Run a tiny local MLX smoke test on Mac.
4. Move the same split files to your GPU server.
5. Run `train_hf_qlora.py` there.

## 5. Evaluate the Fine-Tuned Model

After training on GPU, evaluate the adapter on the held-out validation tasks:

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

This script reports:

- valid-format rate
- raw exact-match rate
- normalized exact-match rate
- average best token-F1 against the gold hypotheses for each task

## 6. End-to-End ARC Evaluation

To evaluate the finetuned hypothesis model inside the full ARC loop:

1. generate hypotheses with the finetuned model
2. generate programs with an OpenAI model
3. execute those programs on ARC train examples

Example command:

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

This produces a trace JSON with:

- generated hypotheses
- generated programs
- train-set execution results
- `solved_train`
- `useful_hypotheses`

This is the closest evaluation to the original dataset-generation pipeline.
