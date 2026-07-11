"""Continue fine-tuning from the already-trained LoRA adapter weights.

The first attempt used trainer.train(resume_from_checkpoint=...), which also
reloads the full optimizer state — that pushed peak VRAM past what the 12GB
card has and crashed with "No or negligible GPU memory available for fused
cross entropy" right at the first step. Loading just the adapter *weights*
(not the optimizer state) as the starting point avoids that overhead; the
tradeoff is a fresh optimizer (no momentum carried over), which is a minor
loss for LoRA continued pretraining, not a correctness problem.

Second attempt ran 9 steps at a stable ~800s/step (already slow, but
survivable) and then stalled hard on step 10 for hours with the GPU still
pegged at 100% but no progress — no crash, no OOM, just stuck, with VRAM
sitting at 11.6/12GB (almost no headroom). Padding-free packing means a
single step's real cost depends on how long the concatenated sequences in
that micro-batch happen to be; with MAX_SEQ_LENGTH=2048 and so little VRAM
headroom, an unlucky batch of long sequences had nowhere to go but into some
slow fallback path. Fixes below all aim at giving more headroom and bounding
the worst case, plus saving checkpoints often enough that a repeat stall
doesn't cost another whole run:
- MAX_SEQ_LENGTH halved (2048 -> 1024): caps the worst-case packed-sequence
  cost per step.
- gradient_accumulation_steps halved (8 -> 4): lower peak memory buildup.
- dataloader_num_workers=0: Windows + multiprocess DataLoader workers is a
  known source of silent deadlocks; this removes that whole failure class.
- save_steps dropped way down (30 -> 5): so a stall loses minutes of
  progress, not hours.
"""

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

from data import load_kyrgyz_text

ADAPTER_PATH = "outputs/kazllm-kyrgyz-lora-round2-final"  # round 2 result (180 + 75 steps)
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "outputs/kazllm-kyrgyz-lora-round3"
MAX_STEPS = 1000


def main():
    # Loading directly from the adapter dir attaches the trained LoRA weights
    # on top of the base model in one step, ready for further training.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    dataset = load_kyrgyz_text(limit=50000)
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
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            max_steps=MAX_STEPS,
            learning_rate=3e-5,  # slightly lower: this is a top-up, not the initial pass
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=20,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=3,
            optim="adamw_8bit",
            dataloader_num_workers=0,
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR + "-final")
    tokenizer.save_pretrained(OUTPUT_DIR + "-final")
    print(f"Saved LoRA adapter to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
