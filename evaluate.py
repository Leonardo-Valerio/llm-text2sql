"""Avaliação: mede Exact Match e similaridade do SQL gerado contra a resposta correta.

Uso:
    python evaluate.py                         # avalia o modelo base (sem fine-tuning)
    python evaluate.py --adapter outputs/A_...  # avalia um modelo treinado (adapter LoRA)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import (build_messages, exact_match, generate, load_base_model, load_splits,
                    load_tokenizer, require_gpu, similarity)


def evaluate_model(model, tok, test_ds, batch_size: int = 16, max_new_tokens: int = 128, desc: str = ""):
    model.eval()
    rows, t0 = [], time.time()
    for i in range(0, len(test_ds), batch_size):
        batch = test_ds.select(range(i, min(i + batch_size, len(test_ds))))
        msgs = [build_messages(c, q) for c, q in zip(batch["context"], batch["question"])]
        preds = generate(model, tok, msgs, max_new_tokens=max_new_tokens)
        for ex, pred in zip(batch, preds):
            rows.append({
                "question": ex["question"],
                "context": ex["context"],
                "gold": ex["answer"],
                "pred": pred,
                "exact_match": exact_match(pred, ex["answer"]),
                "similarity": round(similarity(pred, ex["answer"]), 4),
            })
        print(f"  [{desc}] {len(rows)}/{len(test_ds)} avaliados", flush=True)
    elapsed = time.time() - t0
    n = len(rows)
    metrics = {
        "exact_match": round(sum(r["exact_match"] for r in rows) / n, 4),
        "similarity": round(sum(r["similarity"] for r in rows) / n, 4),
        "eval_time_s": round(elapsed, 1),
        "ms_por_exemplo": round(1000 * elapsed / n, 1),
        "n_test": n,
    }
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="Pasta do adapter LoRA (omitir = modelo base)")
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    require_gpu()
    _, test_ds = load_splits(n_train=0, n_test=args.n_test)
    tok = load_tokenizer()
    model = load_base_model()
    name = "base"
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        name = Path(args.adapter).name

    metrics, rows = evaluate_model(model, tok, test_ds, args.batch_size, desc=name)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out = Path("results") / f"eval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detalhes salvos em {out}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
