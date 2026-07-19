"""Браузерный UI переводчика myizam-translator (через Ollama, модель в питон не грузится).

Usage:
    python translate_ui.py            # http://localhost:7861

Пока на GPU идёт обучение, Ollama выполнит модель на CPU — медленно (десятки
секунд), но работать будет. На свободной карте перевод занимает ~1-3 секунды.
"""

import json
import urllib.request

import gradio as gr

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "myizam-translator"

PROMPTS = {
    "ky → ru": "Тапшырма: Төмөнкү юридикалык текстти орус тилине так которуу.\nТекст: {text}\nКотормо:",
    "ru → ky": "Тапшырма: Төмөнкү юридикалык текстти кыргыз тилине так которуу.\nТекст: {text}\nКотормо:",
}

EXAMPLES = [
    ["ru → ky", "На одного ребенка взыскивается одна четверть заработка родителя."],
    ["ru → ky", "Работникам предоставляется ежегодный оплачиваемый отпуск продолжительностью 28 календарных дней."],
    ["ky → ru", "Кызматкер эмгек келишимин бузуу жөнүндө иш берүүчүгө бир ай мурда жазуу жүзүндө билдирүүгө милдеттүү."],
    ["ky → ru", "Балдарга алимент ким жана канча өлчөмдө төлөшү керек?"],
]


def translate(direction: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    payload = {
        "model": MODEL,
        "prompt": PROMPTS[direction].format(text=text),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 700},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.load(resp)["response"].strip()
    except Exception as e:  # noqa: BLE001 — UI должен показать причину, а не упасть
        return f"[ошибка: {e} — проверьте, что Ollama запущена и модель {MODEL} создана]"


with gr.Blocks(title="Myizam Translator") as demo:
    gr.Markdown(
        "## myizam-translator — юридический перевод ru ↔ ky\n"
        "KazLLM-8B + QLoRA на параллельном корпусе кодексов КР. Модель заточена под "
        "юридические тексты; бытовую речь и английский переводит плохо — это норма."
    )
    direction = gr.Radio(list(PROMPTS), value="ru → ky", label="Направление")
    src = gr.Textbox(lines=5, label="Текст", placeholder="Вставьте юридический текст…")
    btn = gr.Button("Перевести", variant="primary")
    dst = gr.Textbox(lines=5, label="Перевод", interactive=False)
    gr.Examples(EXAMPLES, inputs=[direction, src])
    btn.click(translate, inputs=[direction, src], outputs=dst)
    src.submit(translate, inputs=[direction, src], outputs=dst)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
