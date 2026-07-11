from unsloth import FastLanguageModel

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-lora-round2-final"

DONBAS_CONTEXT = (
    "Украинанын чыгышындагы жикчилдер менен өкмөттүн куралдуу тирешүүсү башталгандан бери "
    "2,5 миңден ашуун карапайым адам өлүп, 9 миңден ашууну жарадар болду. Бул тууралуу "
    "Бириккен Улуттар Уюмунун Украинадагы адам укуктарына арналган баяндамасында көрсөтүлөт. "
    "Анда Донбасстагы чатак аймагында адамдардын жоголушу, сотсуз жазалоо, кыйноо жана "
    "сексуалдык зомбулук көбүрөөк катталганы айтылат."
)

PROMPT = (
    "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз. "
    f"Төмөнкү тексттеги фактыларга гана таянып, кыргызча 2-3 сүйлөм менен кыскача жыйынтыкта:\n\n{DONBAS_CONTEXT}"
)


def generate(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    print("=== BASE MODEL (before fine-tune) ===")
    base_output = generate(model, tokenizer, PROMPT)
    print(base_output)

    print("\n=== LOADING LORA ADAPTER ===")
    model.load_adapter(ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    print("=== FINE-TUNED MODEL (after continued pretraining) ===")
    tuned_output = generate(model, tokenizer, PROMPT)
    print(tuned_output)


if __name__ == "__main__":
    main()
