# Verification Results — LoRA Fine-Tuning

**Date:** 2026-10-01
**Script:** `verify_mechanics_cpu.py` (committed in this repo)
**Result:** MECHANICS CHECK PASSED in 25.6s

## Environment

- Linux VM, Python 3.12, CPU-only (no GPU)
- Isolated venv with: `torch` (CPU), `transformers`, `peft`, `datasets`,
  `accelerate`, `trl`

## Command

```bash
~/workspace/ml-projects/venv/bin/python verify_mechanics_cpu.py
```

## What was verified (pipeline mechanics on CPU)

| Check | Result |
|---|---|
| `train_lora.py` imports cleanly | yes (including `trl.SFTTrainer`) |
| LoRA attach with the project's exact config (r=16, alpha=32, dropout=0.05) | OK — 32,768 trainable / 3,414,848 total = **0.960%** of params (tiny random GPT-2 stand-in on CPU) |
| Real optimizer steps | 4 steps in 0.7s, final loss 10.5115 |
| Adapter save | `cpu-sanity-adapter/` written |
| `format_example()` on a **real** `databricks/databricks-dolly-15k` sample | OK — `closed_qa` example → 767 chars, `<|user|>`/`<|assistant|>` template correct |

## What was NOT run (important)

- The **real TinyLlama-1.1B QLoRA fine-tune was NOT run**. It needs a CUDA
  GPU with 4-bit quantization (bitsandbytes); this machine is CPU-only.
- No base-vs-fine-tuned generation comparison and no 1,000-example training
  run were performed. Any claim of completed TinyLlama training results would
  be inaccurate — the training script is provided to run on a GPU (e.g. a
  free Colab T4), and its expected outputs are described in the README as
  *what the script does*, not as completed results.
