# Resultados

**GPU:** NVIDIA GeForce GTX 1650  
**Modelo:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Exemplos de teste:** 100 (nunca vistos no treino)

| Experimento | LR | LoRA r | Épocas | Loss final | Tempo treino (s) | VRAM pico (GB) | Exact Match | Similaridade |
|---|---|---|---|---|---|---|---|---|
| Modelo base (sem treino) | - | - | - | - | - | - | 13.0% | 55.2% |
| A_lr2e-4_r8_ep1 | 0.0002 | 8 | 1 | 0.1147 | 440.8 | 2.77 | 57.0% | 96.1% |
| B_lr5e-5_r8_ep1 | 5e-05 | 8 | 1 | 0.1361 | 3673.9 | 2.77 | 56.0% | 95.7% |
| C_lr2e-4_r16_ep1 ⭐ | 0.0002 | 16 | 1 | 0.1109 | 3961.0 | 2.83 | 63.0% | 96.2% |

⭐ Melhor configuração: **C_lr2e-4_r16_ep1** (Exact Match 63.0% vs 13.0% do modelo base).

![Curvas de loss](results/loss_curves.png)
