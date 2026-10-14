"""Funções compartilhadas: configuração, dataset, prompts, carregamento do modelo e métricas."""
import difflib
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_ID = "b-mc2/sql-create-context"
SEED = 42
TEST_POOL = 500  # os 500 primeiros exemplos (após embaralhar) nunca entram no treino

SYSTEM_PROMPT = (
    "Você é um assistente especialista em bancos de dados. "
    "Dado o schema (CREATE TABLE) e uma pergunta, responda APENAS com a consulta SQL "
    "correta, em uma única linha, sem explicações."
)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
def require_gpu() -> torch.device:
    """O trabalho exige GPU: aborta com mensagem clara se CUDA não estiver disponível."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[ERRO] GPU CUDA não encontrada pelo PyTorch.\n"
            "Rode `python check_env.py` e siga a seção 'Pré-requisitos' do README.\n"
            "Causa mais comum: PyTorch instalado sem CUDA (versão CPU).\n"
        )
    return torch.device("cuda")


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"


# ---------------------------------------------------------------------------
# Dataset e prompts
# ---------------------------------------------------------------------------
def build_messages(context: str, question: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{context}\n\nPergunta: {question}"},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def load_splits(n_train: int = 2000, n_test: int = 100):
    """Divide o dataset de forma determinística. Teste e treino nunca se sobrepõem."""
    assert n_test <= TEST_POOL, f"n_test deve ser <= {TEST_POOL}"
    ds = load_dataset(DATASET_ID, split="train").shuffle(seed=SEED)
    test = ds.select(range(n_test))
    train = ds.select(range(TEST_POOL, TEST_POOL + n_train))
    return train, test


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(for_training: bool = False):
    """Treino: bf16 se a GPU suportar, senão fp32 (+ AMP fp16 no Trainer).
    Inferência: bf16 ou fp16 para economizar VRAM."""
    if for_training:
        dtype = torch.bfloat16 if use_bf16() else torch.float32
    else:
        dtype = torch.bfloat16 if use_bf16() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    return model.to("cuda")


@torch.no_grad()
def generate(model, tok, batch_msgs, max_new_tokens: int = 128, temperature: float = 0.0):
    """Gera respostas para uma lista de conversas (batch)."""
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kwargs)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def clean_sql(s: str) -> str:
    """Remove blocos ```sql``` e texto extra, mantendo só a primeira consulta."""
    s = s.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.S | re.I)
    if m:
        s = m.group(1)
    s = s.strip().split(";")[0]
    return s.strip()


def normalize_sql(s: str) -> str:
    s = clean_sql(s).replace('"', "'").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>*])\s*", r"\1", s)
    return s.strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


def similarity(pred: str, gold: str) -> float:
    return difflib.SequenceMatcher(None, normalize_sql(pred), normalize_sql(gold)).ratio()
