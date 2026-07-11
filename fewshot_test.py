"""Compare zero-shot vs one-shot prompting on the KyrgyzLLM adapter.

Continued pretraining (what train.py/continue_train.py do) teaches the
model to write fluent Kyrgyz text, but doesn't teach it to follow an
instruction and stay on-topic the way SFT/instruction-tuning would.
Symptom reported: on new prompts the model sometimes "loses the thread"
mid-generation. Putting one worked example of the exact task/format in
the prompt itself is a cheap way to test whether that alone fixes
on-topic-ness, before investing in an actual SFT pass.

The one-shot example reuses the Donbas context + the round-1 adapter's
own answer to it (already judged coherent and correct by the user,
native speaker, aside from one Kazakh-influenced word) -- so the example
itself isn't a new source of Kyrgyz errors. The actual test prompt uses
a fresh, different article pulled from the training corpus, so the
comparison isn't just repeating a text the model has seen judged before.

Usage:
    python fewshot_test.py
    python fewshot_test.py --adapter outputs/kazllm-kyrgyz-lora-round3/checkpoint-950
"""

import argparse

from unsloth import FastLanguageModel

from data import load_kyrgyz_text

MODEL_PATH = "C:/hf/kazllm"
DEFAULT_ADAPTER = "outputs/kazllm-kyrgyz-lora-round3/checkpoint-950"

FEWSHOT_CONTEXT = (
    "Украинанын чыгышындагы жикчилдер менен өкмөттүн куралдуу тирешүүсү башталгандан бери "
    "2,5 миңден ашуун карапайым адам өлүп, 9 миңден ашууну жарадар болду. Бул тууралуу "
    "Бириккен Улуттар Уюмунун Украинадагы адам укуктарына арналган баяндамасында көрсөтүлөт. "
    "Анда Донбасстагы чатак аймагында адамдардын жоголушу, сотсуз жазалоо, кыйноо жана "
    "сексуалдык зомбулук көбүрөөк катталганы айтылат."
)
FEWSHOT_ANSWER = (
    "Бул баяндамада 2,5 миңден ашуун адам өлүп, 9 миңден ашууну жарадар болгону айтылат. "
    "Бул тууралуу Бириккен Улуттар Уюмунун Украинадагы адам укуктарына арналган "
    "баяндамасында көрсөтүлөт. Анда Донбасстагы чатак аймагында адамдардын жоголушу, "
    "сотсуз жазалоо жана кыйноо фактылары катталган."
)

TASK_INSTRUCTION = "Төмөнкү тексттеги фактыларга гана таянып, кыргызча 2-3 сүйлөм менен кыскача жыйынтыкта."


def build_zero_shot(context):
    return (
        "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз. "
        f"{TASK_INSTRUCTION}\n\n{context}"
    )


def build_few_shot(context):
    return (
        "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
        f"Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {FEWSHOT_CONTEXT}\n"
        f"Жыйынтык: {FEWSHOT_ANSWER}\n\n"
        f"Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {context}\n"
        f"Жыйынтык:"
    )


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading base model + adapter ({args.adapter}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    # Skip the first couple of articles so this isn't the exact same
    # snippet every run happens to grab first; still deterministic.
    dataset = load_kyrgyz_text(min_chars=300, max_chars=800, limit=5)
    test_context = dataset[2]["text"]

    print("\n=== TEST CONTEXT (fresh article, not the few-shot example) ===")
    print(test_context)

    print("\n=== ZERO-SHOT ===")
    print(generate(model, tokenizer, build_zero_shot(test_context), args.max_new_tokens))

    print("\n=== ONE-SHOT (Donbas example in the prompt) ===")
    print(generate(model, tokenizer, build_few_shot(test_context), args.max_new_tokens))


if __name__ == "__main__":
    main()
