"""Interactive playground for trying the model by hand.

Loads the base KazLLM model, optionally with a LoRA adapter on top, and lets
you type prompts in a loop instead of editing a hardcoded test prompt.

Usage:
    python chat.py                                          # base model only
    python chat.py --adapter outputs/kazllm-kyrgyz-lora-final
    python chat.py --adapter outputs/kazllm-kyrgyz-lora-round2-final --max-new-tokens 300

Don't run this while a training script (train.py / continue_train.py) is
using the GPU — there's only one 12GB card, loading the model twice will
likely crash both with an out-of-memory error.
"""

import argparse

from unsloth import FastLanguageModel

MODEL_PATH = "C:/hf/kazllm"

INSTRUCTION = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. "
    "Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
)


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter dir; omit for base model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
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
    print(f"\nReady — model: {label}")
    print("Type a question/text in Kyrgyz, empty line or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        prompt = INSTRUCTION + user_input
        response = generate(model, tokenizer, prompt, args.max_new_tokens)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
