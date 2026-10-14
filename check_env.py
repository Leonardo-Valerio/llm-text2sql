"""Verifica os pré-requisitos: Python, PyTorch com CUDA, GPU, VRAM e bibliotecas.
Rode no terminal do VSCode e tire o print da saída (junto com o GPU Monitor).

Uso:
    python check_env.py
"""
import importlib
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

lines = []


def log(msg=""):
    print(msg)
    lines.append(msg)


def ok(cond, msg_ok, msg_fail):
    log(("  [OK]   " + msg_ok) if cond else ("  [FALHA] " + msg_fail))
    return cond


log("=" * 64)
log(" VERIFICAÇÃO DO AMBIENTE  -  CP2/CP3 Generative AI for Engineering")
log("=" * 64)
log(f"Sistema: {platform.system()} {platform.release()} | Python {sys.version.split()[0]}")
ok(sys.version_info >= (3, 10), "Python >= 3.10", "Use Python 3.10, 3.11 ou 3.12")

log("\n[1] Driver NVIDIA (nvidia-smi)")
if shutil.which("nvidia-smi"):
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    log(smi)
else:
    ok(False, "", "nvidia-smi não encontrado: instale/atualize o driver da NVIDIA")

log("[2] CUDA Toolkit (nvcc) - opcional para este projeto, o PyTorch já traz o runtime CUDA")
if shutil.which("nvcc"):
    log("  " + subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])
else:
    log("  nvcc não encontrado (ok, mas o professor pede o CUDA Toolkit instalado: veja o README)")

log("\n[3] PyTorch + CUDA")
try:
    import torch
    log(f"  torch {torch.__version__} | CUDA do PyTorch: {torch.version.cuda}")
    cuda_ok = ok(torch.cuda.is_available(), "torch.cuda.is_available() = True",
                 "PyTorch NÃO enxerga a GPU. Reinstale com: "
                 "pip install torch --index-url https://download.pytorch.org/whl/cu126")
    if cuda_ok:
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1024**3
        log(f"  GPU: {p.name} | VRAM: {vram:.1f} GB | Compute capability: {p.major}.{p.minor}")
        log(f"  bf16 nativo: {p.major >= 8}")
        ok(vram >= 3.5, "VRAM suficiente (>= 4 GB)", "VRAM < 4 GB: use --batch_size 1 no run_experiments.py")
        sys.path.insert(0, str(Path(__file__).parent))
        from common import low_vram, precision
        log(f"  Precisão que será usada: {precision()} | perfil: "
            f"{'econômico (batch 2, gradient checkpointing, 1000 exemplos)' if low_vram() else 'padrão'}")
        # Teste rápido de desempenho
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            a @ a
        torch.cuda.synchronize()
        tflops = 20 * 2 * 4096**3 / (time.time() - t0) / 1e12
        log(f"  Teste de multiplicação de matrizes na GPU: {tflops:.1f} TFLOPS (fp16)")
except ImportError:
    ok(False, "", "PyTorch não instalado")

log("\n[4] Bibliotecas")
for pkg in ["transformers", "peft", "datasets", "accelerate", "gradio", "matplotlib"]:
    try:
        m = importlib.import_module(pkg)
        ok(True, f"{pkg} {getattr(m, '__version__', '')}", "")
    except ImportError:
        ok(False, "", f"{pkg} não instalado: pip install -r requirements.txt")

log("=" * 64)
Path("results").mkdir(exist_ok=True)
Path("results/gpu_info.txt").write_text("\n".join(lines), encoding="utf-8")
log("Saída salva em results/gpu_info.txt")
