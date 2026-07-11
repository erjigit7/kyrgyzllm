"""Smoke test: can the sft-v2 adapter do RagGuard's HALLUCINATED_PROMPT
(deliberately corrupt one fact) reliably, not just the GROUNDED_PROMPT
(accurate summary) it was actually trained on?

This matters before generating thousands of examples for real -- if the
model can't do controlled single-fact corruption, RagGuard's dataset
generation needs a different (programmatic) approach for the label=0
half instead of relying on the model for it.
"""

from unsloth import FastLanguageModel

from data import load_kyrgyz_text

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-sft-v2-final"

GROUNDED_PROMPT = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
    "Тапшырма: Төмөнкү тексттеги фактыларга гана таянып, кыргызча кыскача жыйынтыкта.\n"
    "Текст: {text}\n"
    "Жыйынтык:"
)
HALLUCINATED_PROMPT = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
    "Тапшырма: Төмөнкү тексттин кыскача жыйынтыгын жаз, бирок атайылап так бир фактыны "
    "(сан, дата, ат же жер) тексттеги чындыкка карама-каршы келтир — калганы туура болсун.\n"
    "Текст: {text}\n"
    "Жыйынтык:"
)


def generate(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    dataset = load_kyrgyz_text(min_chars=300, max_chars=800, limit=8)
    for i in [5, 6, 7]:
        text = dataset[i]["text"]
        print(f"\n=== CONTEXT idx={i} ===")
        print(text)
        print(f"\n=== GROUNDED idx={i} ===")
        print(generate(model, tokenizer, GROUNDED_PROMPT.format(text=text)))
        print(f"\n=== HALLUCINATED idx={i} ===")
        print(generate(model, tokenizer, HALLUCINATED_PROMPT.format(text=text)))


if __name__ == "__main__":
    main()
