# LoRA Fine-Tuning + 4-bit Quantization

A script to fine-tune **TinyLlama-1.1B-Chat** on 1,000 instruction examples
using **LoRA** (Low-Rank Adaptation), with the base model loaded in **4-bit**
(QLoRA setup) — plus a CPU script that verifies the pipeline mechanics.

> **Status:** the training script (`train_lora.py`) is written and its
> mechanics are verified on CPU (see `RESULTS.md`), but the full
> TinyLlama-1.1B GPU fine-tune has **not** been run yet — it needs a CUDA
> GPU. The loss/eval numbers below describe what the script *does*, not
> completed results.

## Why this project

- **Parameter-efficient fine-tuning:** only ~0.5–1% of parameters are trained
  (the LoRA adapters); the 1.1B base weights stay frozen.
- **Quantization for deployment:** 4-bit NF4 base + tiny adapter = a fine-tuned
  model that ships ~100x smaller than a full fp16 copy — directly relevant to
  on-device/edge ML.
- **Evaluation built in:** the script prints trainable-param %, compares base vs.
  fine-tuned generations, and reports adapter-vs-full-model size (when run on
  a GPU — see `RESULTS.md` for what has actually been verified so far).

## Run it (Google Colab, free T4 GPU)

1. Runtime → Change runtime type → **T4 GPU**.
2. `pip install -q "transformers>=4.40" "peft>=0.10" "trl>=0.8" datasets bitsandbytes accelerate`
3. Upload `train_lora.py` and run it. Takes ~15–25 minutes.

## How LoRA works (30-second version)

Instead of updating the full weight matrix `W`, LoRA learns a low-rank update
`ΔW = A·B` where `A` and `B` are tiny matrices (rank 16 here). At inference the
adapter merges back into `W`, so there is **zero latency overhead**. You get
task specialization while shipping only megabytes of adapter weights.

## Interview talking points

- Why LoRA over full fine-tuning? (Memory: optimizer states for 1.1B params don't
  fit on small GPUs; storage: one base model + many task adapters.)
- Why 4-bit NF4 + float16 compute? (Memory footprint ÷4 with minimal accuracy loss;
  compute stays in fp16 for stability.)
- What would you check before deploying? (Eval on held-out prompts, quantization
  accuracy delta, latency on target hardware.)
