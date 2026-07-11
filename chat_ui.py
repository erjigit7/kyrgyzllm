"""Browser-based chat UI for trying the model, instead of the raw CLI.

Same model/adapter loading as chat.py, just wrapped in a Gradio interface.

Usage:
    python chat_ui.py --adapter outputs/kazllm-kyrgyz-sft-v2-final
    python chat_ui.py --adapter outputs/kazllm-kyrgyz-lora-round3/checkpoint-950 --raw
    python chat_ui.py                                          # base model only

Opens a local browser tab automatically. Close the terminal (Ctrl+C) to stop.
Don't run this at the same time as chat.py — only one process can hold the
model in the 12GB of VRAM at once.

The sft-v2 adapter was only trained on ONE task (summarize a pasted
Kyrgyz text), using a specific "Тапшырма:/Текст:/Жыйынтык:" prompt shape
-- not open-ended chat. By default this wraps whatever you type as the
"Текст:" so it actually matches what the model was trained on. Pass
--raw to send your message unwrapped instead (e.g. for the base model
or the pre-SFT checkpoints, which don't expect this template).
"""

import argparse

import gradio as gr
from unsloth import FastLanguageModel

MODEL_PATH = "C:/hf/kazllm"

SYSTEM_PREFIX = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. "
    "Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
)
TASK_INSTRUCTION = "Төмөнкү тексттеги фактыларга гана таянып, кыргызча кыскача жыйынтыкта."


def build_summarize_prompt(text):
    return (
        f"{SYSTEM_PREFIX}Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {text}\n"
        f"Жыйынтык:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter dir; omit for base model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--raw", action="store_true", help="Send your message unwrapped (no summarize template)")
    args = parser.parse_args()

    print(f"Loading base model from {MODEL_PATH} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )

    if args.adapter:
        print(f"Loading adapter from {args.adapter} ...")
        model.load_adapter(args.adapter)

    FastLanguageModel.for_inference(model)
    label = args.adapter if args.adapter else "BASE (no adapter)"
    mode = "raw (unwrapped)" if args.raw else "summarize template"

    def respond(message, history, max_new_tokens):
        prompt = (SYSTEM_PREFIX + message) if args.raw else build_summarize_prompt(message)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    with gr.Blocks(title="KyrgyzLLM chat") as demo:
        gr.Markdown(f"### KyrgyzLLM — модель: `{label}` | режим: `{mode}`")
        max_tokens_slider = gr.Slider(50, 500, value=args.max_new_tokens, step=10, label="Max new tokens")
        chat = gr.ChatInterface(
            fn=lambda message, history: respond(message, history, max_tokens_slider.value),
        )

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
