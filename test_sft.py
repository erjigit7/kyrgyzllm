"""Test the SFT adapter on the exact prompt format it was trained on,
using the SAME held-out landslide article from fewshot_test.py (not in
sft_data.jsonl) -- this is the real generalization check, since train
loss dropping to ~0.01 over 22 examples is a strong overfitting signal
and could just mean memorization of the training pairs, not real
summarization ability.
"""

from unsloth import FastLanguageModel

from data import load_kyrgyz_text

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-sft-v2-final"
TASK_INSTRUCTION = "Төмөнкү тексттеги фактыларга гана таянып, кыргызча кыскача жыйынтыкта."


def build_prompt(context):
    return (
        "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
        f"Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {context}\n"
        f"Жыйынтык:"
    )


def generate(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    dataset = load_kyrgyz_text(min_chars=300, max_chars=800, limit=5)

    for i in range(len(dataset)):
        test_context = dataset[i]["text"]
        print(f"\n=== TEST CONTEXT idx={i} (held out) ===")
        print(test_context)
        print(f"\n=== SFT ADAPTER OUTPUT idx={i} ===")
        print(generate(model, tokenizer, build_prompt(test_context)))


if __name__ == "__main__":
    main()
