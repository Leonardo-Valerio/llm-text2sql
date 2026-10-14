"""Fine-tuning com LoRA do Qwen2.5-0.5B-Instruct para Text-to-SQL (na GPU).

Uso (um treino isolado):
    python train.py --lr 2e-4 --r 8 --epochs 1 --name meu_teste
Para rodar todos os experimentos de uma vez, use run_experiments.py.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from common import (SEED, build_messages, load_base_model, load_splits, load_tokenizer,
                    require_gpu, use_bf16)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_dataset(ds, tok, max_len: int = 512):
    """Tokeniza no formato de chat. A loss só é calculada na resposta (SQL), não no prompt."""
    def fn(ex):
        prompt = tok.apply_chat_template(build_messages(ex["context"], ex["question"]),
                                         tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(build_messages(ex["context"], ex["question"], ex["answer"]),
                                       tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        labels = ([-100] * len(p_ids) + f_ids[len(p_ids):])[:len(f_ids)]
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}
    return ds.map(fn, remove_columns=ds.column_names)


def train_lora(name: str, lr: float, r: int, epochs: int, train_ds, tok,
               batch_size: int = 8, grad_accum: int = 2, out_root: str = "outputs"):
    """Treina um adapter LoRA e devolve (modelo_treinado, métricas_de_treino)."""
    require_gpu()
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()

    model = load_base_model(for_training=True)
    lora_cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                          target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    tok.padding_side = "right"
    tokenized = tokenize_dataset(train_ds, tok)
    out_dir = Path(out_root) / name
    bf16 = use_bf16()
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=bf16,
        fp16=not bf16,
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    print(f"\n=== Treinando {name} | lr={lr} r={r} epochs={epochs} | "
          f"{trainable:,} params treináveis de {total:,} ({100 * trainable / total:.2f}%) ===")
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    loss_hist = [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]
    metrics = {
        "name": name, "lr": lr, "r": r, "lora_alpha": 2 * r, "epochs": epochs,
        "n_train": len(train_ds), "batch_efetivo": batch_size * grad_accum,
        "trainable_params": trainable, "train_loss": round(result.training_loss, 4),
        "train_time_s": round(train_time, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "precision": "bf16" if bf16 else "fp16 (AMP)",
        "loss_history": loss_hist, "adapter_dir": str(out_dir),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Treino concluído em {train_time:.0f}s | loss={result.training_loss:.4f} | "
          f"VRAM pico={metrics['peak_vram_gb']} GB | adapter salvo em {out_dir}")
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="manual")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    require_gpu()
    train_ds, _ = load_splits(args.n_train)
    tok = load_tokenizer()
    train_lora(args.name, args.lr, args.r, args.epochs, train_ds, tok, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
