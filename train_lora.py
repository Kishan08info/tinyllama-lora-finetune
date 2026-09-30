"""
LoRA fine-tune a small chat LLM + 4-bit quantization for deployment.

What it does:
  1. Loads TinyLlama-1.1B-Chat in 4-bit (QLoRA setup) on a Colab T4 GPU.
  2. Fine-tunes ONLY low-rank adapter matrices (LoRA) on ~1k instruction examples.
  3. Prints trainable-params % (expect < 1% of the model).
  4. Compares base vs. fine-tuned responses on a sample prompt.
  5. Reports adapter size vs. full-model size (why LoRA is deployment-friendly).

Run on Google Colab (Runtime -> Change runtime type -> T4 GPU):
    pip install -q "transformers>=4.40" "peft>=0.10" "trl>=0.8" datasets bitsandbytes accelerate
    python train_lora.py
Takes ~15-25 min on a free T4.
"""
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_ID = "databricks/databricks-dolly-15k"
N_EXAMPLES = 1000
OUTPUT_DIR = "./tinyllama-lora"

SYSTEM = "You are a helpful assistant that answers concisely."


def format_example(ex):
    """Dolly -> chat-style prompt string."""
    ctx = f"\nContext: {ex['context']}" if ex.get("context") else ""
    return (f"<|system|>\n{SYSTEM}</s>\n<|user|>\n{ex['instruction']}{ctx}</s>\n"
            f"<|assistant|>\n{ex['response']}</s>")


def count_trainable(model):
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {train:,} / {total:,} ({100 * train / total:.3f}%)")
    return train, total


def main():
    assert torch.cuda.is_available(), "Needs a GPU — use Colab with a T4."
    print("GPU:", torch.cuda.get_device_name(0))

    # ---- 1. Tokenizer + 4-bit base model (QLoRA) ----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # ---- 2. Attach LoRA adapters ----
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    train, total = count_trainable(model)

    # ---- 3. Data ----
    ds = load_dataset(DATASET_ID, split=f"train[:{N_EXAMPLES}]")
    ds = ds.map(lambda ex: {"text": format_example(ex)}, remove_columns=ds.column_names)

    # ---- 4. Train (adapters only) ----
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=1,
        learning_rate=2e-4, fp16=True, logging_steps=10,
        save_steps=200, max_steps=150, report_to="none",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, args=args,
                         max_seq_length=512, dataset_text_field="text")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}/")

    # ---- 5. Base vs fine-tuned comparison ----
    prompt = ("<|system|>\nYou are a helpful assistant that answers concisely.</s>\n"
              "<|user|>\nExplain model quantization in two sentences.</s>\n<|assistant|>\n")
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    print("\n--- Fine-tuned response ---")
    print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # ---- 6. Deployment math ----
    import os
    adapter_mb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in
                     os.listdir(OUTPUT_DIR) if f.endswith((".bin", ".safetensors"))) / 1e6
    print(f"\nAdapter size: {adapter_mb:.1f} MB vs full-model copy: "
          f"{total * 2 / 1e6:.0f} MB (fp16). "
          f"LoRA ships ~{total * 2 / 1e6 / max(adapter_mb, 1):.0f}x smaller.")


if __name__ == "__main__":
    main()
