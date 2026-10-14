"""Roda TODO o pipeline: avalia o modelo base, treina 4 configurações de parâmetros,
avalia cada uma, gera tabelas, gráfico de loss e atualiza o README automaticamente.

Uso:
    python run_experiments.py           # completo (~15-40 min dependendo da GPU)
    python run_experiments.py --quick   # teste rápido (~3 min) só para ver se tudo funciona
"""
import argparse
import gc
import json
import re
from pathlib import Path

import torch

from common import (MODEL_ID, gpu_name, load_base_model, load_splits, load_tokenizer, low_vram,
                    precision, require_gpu, vram_gb)
from evaluate import evaluate_model
from train import train_lora

# Cada experimento muda UM parâmetro em relação ao A (baseline), para comparar o efeito isolado.
CONFIGS = [
    dict(name="A_lr2e-4_r8_ep1", lr=2e-4, r=8, epochs=1),   # referência
    dict(name="B_lr5e-5_r8_ep1", lr=5e-5, r=8, epochs=1),   # learning rate menor
    dict(name="C_lr2e-4_r16_ep1", lr=2e-4, r=16, epochs=1),  # LoRA rank maior
    dict(name="D_lr2e-4_r8_ep2", lr=2e-4, r=8, epochs=2),   # mais épocas
]

RESULTS = Path("results")


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def plot_losses(train_metrics):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado, pulando gráfico")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in train_metrics:
        h = m["loss_history"]
        ax.plot([x["step"] for x in h], [x["loss"] for x in h], label=m["name"], linewidth=2)
    ax.set_xlabel("Passo de treino")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de loss por experimento")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)
    plt.close(fig)


def write_reports(base_metrics, base_rows, experiments, best):
    lines = [
        f"**GPU:** {gpu_name()}  ",
        f"**Modelo:** `{MODEL_ID}`  ",
        f"**Exemplos de teste:** {base_metrics['n_test']} (nunca vistos no treino)",
        "",
        "| Experimento | LR | LoRA r | Épocas | Loss final | Tempo treino (s) | VRAM pico (GB) | Exact Match | Similaridade |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Modelo base (sem treino) | - | - | - | - | - | - | {base_metrics['exact_match']:.1%} | {base_metrics['similarity']:.1%} |",
    ]
    for e in experiments:
        t, ev = e["train"], e["eval"]
        star = " ⭐" if t["name"] == best["train"]["name"] else ""
        lines.append(
            f"| {t['name']}{star} | {t['lr']} | {t['r']} | {t['epochs']} | {t['train_loss']} | "
            f"{t['train_time_s']} | {t['peak_vram_gb']} | {ev['exact_match']:.1%} | {ev['similarity']:.1%} |"
        )
    lines += ["", f"⭐ Melhor configuração: **{best['train']['name']}** "
              f"(Exact Match {best['eval']['exact_match']:.1%} vs {base_metrics['exact_match']:.1%} do modelo base).",
              "", "![Curvas de loss](results/loss_curves.png)"]
    table = "\n".join(lines)
    (RESULTS / "resultados.md").write_text("# Resultados\n\n" + table + "\n", encoding="utf-8")

    # Exemplos lado a lado: base vs melhor modelo
    ex = ["| Pergunta | Esperado | Modelo base | Modelo treinado |", "|---|---|---|---|"]
    def cell(s):
        return s.replace("|", "\\|").replace("\n", " ")[:160]
    for b, f in list(zip(base_rows, best["rows"]))[:8]:
        ok_b = "✅" if b["exact_match"] else "❌"
        ok_f = "✅" if f["exact_match"] else "❌"
        ex.append(f"| {cell(b['question'])} | `{cell(b['gold'])}` | {ok_b} `{cell(b['pred'])}` | {ok_f} `{cell(f['pred'])}` |")
    examples = "\n".join(ex)
    (RESULTS / "exemplos.md").write_text("# Exemplos de respostas\n\n" + examples + "\n", encoding="utf-8")

    # Atualiza o README entre os marcadores
    readme = Path("README.md")
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        block = f"<!-- RESULTS_START -->\n{table}\n\n### Exemplos (base vs treinado)\n\n{examples}\n<!-- RESULTS_END -->"
        txt = re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->", lambda _: block, txt, flags=re.S)
        readme.write_text(txt, encoding="utf-8")
        print("README.md atualizado com os resultados.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="teste rápido com poucos dados")
    ap.add_argument("--n_train", type=int, default=None, help="padrão: 2000 (1000 em GPUs < 6 GB)")
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="padrão: 8 (2 em GPUs < 6 GB). Diminua se der 'CUDA out of memory'")
    args = ap.parse_args()

    require_gpu()
    small = low_vram()
    args.n_train = args.n_train or (1000 if small else 2000)
    args.batch_size = args.batch_size or (2 if small else 8)
    grad_accum = max(1, 16 // args.batch_size)  # mantém o batch efetivo em 16
    eval_bs = 8 if small else 16
    print(f"GPU detectada: {gpu_name()} ({vram_gb():.1f} GB) | precisão: {precision()} | "
          f"perfil {'econômico (GPU < 6 GB)' if small else 'padrão'} | n_train={args.n_train} | "
          f"batch={args.batch_size}x{grad_accum}")
    configs = CONFIGS
    if args.quick:
        args.n_train, args.n_test, configs = 200, 20, CONFIGS[:1]

    RESULTS.mkdir(exist_ok=True)
    train_ds, test_ds = load_splits(args.n_train, args.n_test)
    tok = load_tokenizer()

    # 1) Modelo base, sem treino (linha de comparação)
    print("\n=== Avaliando modelo base (antes do fine-tuning) ===")
    base = load_base_model()
    base_metrics, base_rows = evaluate_model(base, tok, test_ds, batch_size=eval_bs, desc="base")
    print(base_metrics)
    del base
    free_gpu()

    # 2) Experimentos
    experiments = []
    for cfg in configs:
        model, tmetrics = train_lora(cfg["name"], cfg["lr"], cfg["r"], cfg["epochs"], train_ds, tok,
                                     batch_size=args.batch_size, grad_accum=grad_accum)
        emetrics, rows = evaluate_model(model, tok, test_ds, batch_size=eval_bs, desc=cfg["name"])
        print(emetrics)
        experiments.append({"train": tmetrics, "eval": emetrics, "rows": rows})
        del model
        free_gpu()

    best = max(experiments, key=lambda e: (e["eval"]["exact_match"], e["eval"]["similarity"]))
    (RESULTS / "best_adapter.txt").write_text(best["train"]["adapter_dir"], encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps({
        "gpu": gpu_name(), "model": MODEL_ID,
        "base": {"eval": base_metrics, "rows": base_rows},
        "experiments": experiments,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_losses([e["train"] for e in experiments])
    write_reports(base_metrics, base_rows, experiments, best)
    print(f"\nPronto! Melhor: {best['train']['name']} | resultados em results/ | rode `python app.py`")


if __name__ == "__main__":
    main()
