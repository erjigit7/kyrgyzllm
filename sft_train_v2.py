"""SFT pass on real XLSum Kyrgyz (article, summary) pairs, replacing the
22 hand-written examples from sft_train.py.

sft_train.py's tiny hand-written set worked (fixed pure verbatim-copying
into genuine paraphrasing on all 5 held-out tests) but loss collapsed to
~0.01 over 15 epochs -- a clear overfitting signal, and 2/5 held-out
generations contained fabricated facts not in the source (an invented
name, an invented event number). XLSum's Kyrgyz split has 2266 real,
professionally-written BBC article/summary pairs -- filtered here to
300-2000 char articles and 30-300 char summaries (508 train / 122
validation), which should generalize far better than 22 toy examples
without me having hand-written (and possibly gotten subtly wrong) the
targets myself.

Starts fresh from round3/checkpoint-950 (not from the overfit sft-v1
adapter) so this is a clean comparison, not a compounding one.

Includes periodic eval on the real held-out validation split so
overfitting shows up as a rising eval_loss instead of only being
discoverable after the fact by eyeballing generations.

Usage:
    python sft_train_v2.py
"""

from datasets import load_dataset
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-lora-round3/checkpoint-950"
OUTPUT_DIR = "outputs/kazllm-kyrgyz-sft-v2"
MAX_SEQ_LENGTH = 1024
TASK_INSTRUCTION = "Төмөнкү тексттеги фактыларга гана таянып, кыргызча кыскача жыйынтыкта."

MIN_TEXT, MAX_TEXT = 300, 2000
MIN_SUM, MAX_SUM = 30, 300


def build_example(text, summary, eos_token):
    prompt = (
        "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
        f"Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {text}\n"
        f"Жыйынтык:"
    )
    return prompt + " " + summary + eos_token


def load_split(revision_files_key, tokenizer):
    d = load_dataset(
        "csebuetnlp/xlsum",
        revision="refs/convert/parquet",
        data_files={"data": revision_files_key},
    )["data"]
    d = d.filter(lambda x: MIN_TEXT <= len(x["text"]) <= MAX_TEXT and MIN_SUM <= len(x["summary"]) <= MAX_SUM)
    texts = [build_example(r["text"], r["summary"], tokenizer.eos_token) for r in d]
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    train_dataset = load_split("kyrgyz/train/0000.parquet", tokenizer)
    eval_dataset = load_split("kyrgyz/validation/0000.parquet", tokenizer)
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=UnslothTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=8e-5,
            warmup_ratio=0.1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=32,
            save_strategy="steps",
            save_steps=64,
            save_total_limit=2,
            optim="adamw_8bit",
            dataloader_num_workers=0,
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR + "-final")
    tokenizer.save_pretrained(OUTPUT_DIR + "-final")
    print(f"Saved SFT-v2 adapter to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
