"""Continued pretraining of KazLLM-8B on raw Kyrgyz news text.

Goal: KazLLM defaults to Kazakh even when explicitly prompted in/for Kyrgyz.
Continued pretraining on real Kyrgyz text should shift its output-language
preference, since a big chunk of its next-token predictions will now have
been trained on genuine Kyrgyz rather than Kazakh.

Originally also trained embed_tokens/lm_head (Unsloth's documented pattern for
continued pretraining / new-language adaptation), but on this 8B model with a
128k-token vocabulary that pushed VRAM usage high enough that Unsloth started
offloading the embedding layer to disk mid-training — each step got steadily
slower (28s -> 373s and climbing) instead of settling into a stable pace, so
313 steps would have taken days. Dropped to plain LoRA (attention/MLP only) to
stay in VRAM; a modest accuracy cost, but actually finishes in reasonable time.
"""

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

from data import load_kyrgyz_text

MODEL_PATH = "C:/hf/kazllm"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs/kazllm-kyrgyz-lora"


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_kyrgyz_text(limit=5000)

    def tokenize(batch):
        return tokenizer(batch["text"] + tokenizer.eos_token)

    dataset = dataset.map(lambda x: {"text": x["text"] + tokenizer.eos_token})

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=UnslothTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            max_steps=180,  # targets ~12h at the ~235s/step measured in the smoke test
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="steps",
            save_steps=30,  # frequent checkpoints: this is a long unattended run
            save_total_limit=3,
            optim="adamw_8bit",
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR + "-final")
    tokenizer.save_pretrained(OUTPUT_DIR + "-final")
    print(f"Saved LoRA adapter to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
