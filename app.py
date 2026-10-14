"""Frontend (Gradio) para o usuário escrever o prompt e testar o modelo treinado.

Uso:
    python app.py     -> abra http://127.0.0.1:7860 no navegador
"""
import time
from pathlib import Path

import gradio as gr
import torch

from common import MODEL_ID, build_messages, generate, gpu_name, load_base_model, load_tokenizer, require_gpu

require_gpu()
tok = load_tokenizer()
model = load_base_model()

best_file = Path("results/best_adapter.txt")
ADAPTER = best_file.read_text(encoding="utf-8").strip() if best_file.exists() else None
if ADAPTER and Path(ADAPTER).exists():
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    print(f"Adapter carregado: {ADAPTER}")
else:
    ADAPTER = None
    print("Nenhum adapter treinado encontrado: rode run_experiments.py primeiro. Usando só o modelo base.")
model.eval()


def run(msgs, use_adapter, temperature, max_tokens):
    t0 = time.time()
    if ADAPTER and not use_adapter:
        with model.disable_adapter():
            out = generate(model, tok, [msgs], max_new_tokens=int(max_tokens), temperature=temperature)[0]
    else:
        out = generate(model, tok, [msgs], max_new_tokens=int(max_tokens), temperature=temperature)[0]
    return out, time.time() - t0


def text_to_sql(schema, question, compare, temperature, max_tokens):
    if not schema.strip() or not question.strip():
        return "Preencha o schema e a pergunta.", "", ""
    msgs = build_messages(schema, question)
    sql_ft, t_ft = run(msgs, True, temperature, max_tokens)
    sql_base, t_base = ("", 0.0)
    if compare and ADAPTER:
        sql_base, t_base = run(msgs, False, temperature, max_tokens)
    vram = torch.cuda.max_memory_allocated() / 1024**3
    info = f"GPU: {gpu_name()} | treinado: {t_ft:.2f}s" + (f" | base: {t_base:.2f}s" if compare else "") + \
           f" | VRAM pico: {vram:.2f} GB"
    return sql_ft, sql_base, info


def free_prompt(prompt, use_adapter, temperature, max_tokens):
    if not prompt.strip():
        return ""
    out, t = run([{"role": "user", "content": prompt}], use_adapter, temperature, max_tokens)
    return f"{out}\n\n---\n({t:.2f}s na {gpu_name()})"


EXAMPLES = [
    ["CREATE TABLE funcionarios (id INTEGER, nome VARCHAR, salario INTEGER, departamento VARCHAR)",
     "Qual o salário médio do departamento de Engenharia?"],
    ["CREATE TABLE head (age INTEGER)", "How many heads of the departments are older than 56 ?"],
    ["CREATE TABLE pedidos (id INTEGER, cliente VARCHAR, valor REAL, data VARCHAR)",
     "Liste os 5 clientes com maior valor total de pedidos"],
    ["CREATE TABLE table_name_61 (team VARCHAR, wins INTEGER, losses INTEGER)",
     "Which team has more than 10 wins and fewer than 3 losses?"],
]

with gr.Blocks(title="Text-to-SQL com LLM local") as demo:
    gr.Markdown(
        f"# 🧠 Text-to-SQL com LLM local\n"
        f"Modelo **{MODEL_ID}** com fine-tuning LoRA · rodando na GPU **{gpu_name()}** · "
        f"adapter: `{ADAPTER or 'nenhum'}`"
    )
    with gr.Tab("Gerar SQL"):
        with gr.Row():
            with gr.Column():
                schema = gr.Textbox(label="Schema (CREATE TABLE ...)", lines=4)
                question = gr.Textbox(label="Pergunta em linguagem natural", lines=2)
                compare = gr.Checkbox(label="Comparar com o modelo base (sem treino)", value=True)
                with gr.Row():
                    temp = gr.Slider(0, 1.2, value=0.0, step=0.1, label="Temperatura (0 = determinístico)")
                    max_tok = gr.Slider(16, 256, value=128, step=16, label="Máx. tokens")
                btn = gr.Button("Gerar SQL", variant="primary")
            with gr.Column():
                out_ft = gr.Code(label="SQL – modelo treinado", language="sql")
                out_base = gr.Code(label="SQL – modelo base", language="sql")
                info = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[schema, question])
        btn.click(text_to_sql, [schema, question, compare, temp, max_tok], [out_ft, out_base, info])

    with gr.Tab("Prompt livre"):
        prompt = gr.Textbox(label="Escreva qualquer prompt", lines=5)
        use_ad = gr.Checkbox(label="Usar modelo treinado", value=True)
        with gr.Row():
            temp2 = gr.Slider(0, 1.2, value=0.7, step=0.1, label="Temperatura")
            max_tok2 = gr.Slider(16, 512, value=256, step=16, label="Máx. tokens")
        btn2 = gr.Button("Enviar", variant="primary")
        out2 = gr.Textbox(label="Resposta", lines=10)
        btn2.click(free_prompt, [prompt, use_ad, temp2, max_tok2], out2)

if __name__ == "__main__":
    demo.launch()
