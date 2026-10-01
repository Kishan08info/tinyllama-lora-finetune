"""CPU sanity check of the LoRA pipeline MECHANICS (not the real fine-tune).

The real fine-tune (train_lora.py) needs a CUDA GPU + 4-bit quantization and was
NOT run here. This script verifies, on CPU with a tiny random model, that the
pipeline's moving parts work: LoRA attach -> trainable-param count -> a few
training steps -> adapter save -> reload. It also validates format_example()
against a real databricks-dolly-15k sample.

Run: pip install -q "transformers>=4.40" "peft>=0.10" datasets accelerate
     python verify_mechanics_cpu.py
"""
import io
import sys
import time
import random
import string

import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ---- Reuse the exact helpers from train_lora.py ----
sys.path.insert(0, ".")
from train_lora import LoraConfig as _  # noqa: F401  (confirms train_lora.py imports cleanly)
import train_lora as tl

t0 = time.time()

# 1. LoRA attach with the project's EXACT config, on a tiny CPU model
cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=50257)
base = GPT2LMHeadModel(cfg)
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
# GPT2 uses c_attn/c_proj/c_fc/c_mlp naming; map the project's target list to
# GPT2's actual module names so the mechanics (rank-16 adapter injection) are identical.
lora_cfg_gpt2 = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["c_attn", "c_proj", "c_fc"],
                           task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg_gpt2)
train, total = tl.count_trainable(model)
assert 0 < train < total, "LoRA should add a small trainable fraction"
print(f"[lora-attach] OK — {100 * train / total:.3f}% of params trainable")

# 2. A few real optimizer steps on toy data (proves the training loop works)
torch.manual_seed(0)
tok_ids = torch.randint(0, 50257, (8, 32))
dataset = [{"input_ids": tok_ids[i], "labels": tok_ids[i]} for i in range(8)]
args = TrainingArguments(output_dir="./cpu-sanity-out", per_device_train_batch_size=2,
                         max_steps=4, learning_rate=2e-4, logging_steps=1,
                         report_to="none", save_strategy="no")
trainer = Trainer(model=model, args=args, train_dataset=dataset)
t1 = time.time()
metrics = trainer.train()
t2 = time.time()
print(f"[train] OK — 4 steps on CPU in {t2 - t1:.1f}s, final loss "
      f"{metrics.metrics['train_loss']:.4f}")
model.save_pretrained("./cpu-sanity-adapter")
print("[save] OK — adapter saved to ./cpu-sanity-adapter/")

# 3. format_example() on a REAL databricks-dolly-15k example (with retries;
#    falls back to a synthetic dolly-shaped sample if the Hub is unreachable)
from datasets import load_dataset
ex = None
for attempt in range(3):
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train[:1]")
        ex = ds[0]
        src = f"real dolly example (category={ex['category']})"
        break
    except Exception as e:
        print(f"[format] download attempt {attempt + 1} failed: "
              f"{type(e).__name__}: {str(e)[:100]}")
        time.sleep(5)
if ex is None:
    ex = {"instruction": "Explain LoRA fine-tuning in one sentence.",
          "response": "LoRA fine-tuning adapts a model by training small "
                      "low-rank matrices instead of all weights.",
          "category": "open_qa (synthetic fallback — hub unreachable)"}
    src = "synthetic dolly-shaped fallback (hub unreachable after 3 tries)"
formatted = tl.format_example(ex)
assert "<|user|>" in formatted and "<|assistant|>" in formatted
assert ex["instruction"] in formatted and ex["response"] in formatted
print(f"[format] OK — {src} -> {len(formatted)} chars")
print("  preview:", formatted[:160].replace("\n", "\\n"))

print(f"\nMECHANICS CHECK PASSED in {time.time() - t0:.1f}s total")
print("NOTE: this verifies pipeline mechanics only. The real TinyLlama-1.1B "
      "QLoRA fine-tune was NOT run (no GPU on this machine).")
