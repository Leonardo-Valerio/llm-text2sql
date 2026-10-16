"""Gera os resultados a partir dos modelos JÁ TREINADOS em outputs/, sem treinar de novo.

Use se o run_experiments.py foi interrompido no meio (falta de tempo, bateria etc.).
Ele só reavalia o modelo base e cada experimento concluído (~1 min cada) e gera
a tabela, o gráfico de loss, os exemplos e atualiza o README.
Experimentos que não terminaram de treinar são ignorados.

Uso:
    python make_report.py
"""
import json
from pathlib import Path

from peft import PeftModel

from common import MODEL_ID, gpu_name, load_base_model, load_splits, load_tokenizer, low_vram, require_gpu
from evaluate import evaluate_model
from run_experiments import CONFIGS, RESULTS, free_gpu, plot_losses, write_reports


def main():
    require_gpu()
    eval_bs = 8 if low_vram() else 16
    _, test_ds = load_splits(n_train=0, n_test=100)
    tok = load_tokenizer()
    RESULTS.mkdir(exist_ok=True)

    print("\n=== Avaliando modelo base (sem treino) ===")
    base = load_base_model()
    base_metrics, base_rows = evaluate_model(base, tok, test_ds, batch_size=eval_bs, desc="base")
    print(base_metrics)
    del base
    free_gpu()

    experiments = []
    for cfg in CONFIGS:
        d = Path("outputs") / cfg["name"]
        mfile = d / "train_metrics.json"
        if not (mfile.exists() and (d / "adapter_config.json").exists()):
            print(f"[pulando] {cfg['name']}: treino não foi concluído")
            continue
        tmetrics = json.loads(mfile.read_text(encoding="utf-8"))
        model = PeftModel.from_pretrained(load_base_model(), str(d)).merge_and_unload()
        emetrics, rows = evaluate_model(model, tok, test_ds, batch_size=eval_bs, desc=cfg["name"])
        print(emetrics)
        experiments.append({"train": tmetrics, "eval": emetrics, "rows": rows})
        del model
        free_gpu()

    if not experiments:
        raise SystemExit("Nenhum experimento concluído em outputs/.")

    best = max(experiments, key=lambda e: (e["eval"]["exact_match"], e["eval"]["similarity"]))
    (RESULTS / "best_adapter.txt").write_text(best["train"]["adapter_dir"], encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps({
        "gpu": gpu_name(), "model": MODEL_ID,
        "base": {"eval": base_metrics, "rows": base_rows},
        "experiments": experiments,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_losses([e["train"] for e in experiments])
    write_reports(base_metrics, base_rows, experiments, best)
    print(f"\nPronto! {len(experiments)} experimento(s) | melhor: {best['train']['name']} | rode `python app.py`")


if __name__ == "__main__":
    main()
