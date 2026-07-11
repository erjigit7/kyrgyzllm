"""Light SFT pass on top of round3/checkpoint-950 to teach genuine
paraphrasing instead of near-verbatim copying.

fewshot_test.py showed: zero-shot produced nothing (no cue telling the
model where to start answering); one-shot fixed that but the model
mostly echoed the source's opening sentences back rather than actually
summarizing. Continued pretraining (train.py/continue_train.py) only
ever saw raw article text -- it never saw an (instruction, context) ->
(different, shorter, reworded) pair, so it has no supervision for
"produce new text" as opposed to "continue this text". This trains on
22 hand-written (context, summary) pairs (sft_data.jsonl) to add that
signal on top of the already-good Kyrgyz fluency from continued
pretraining.

Each example is formatted with the same "Тапшырма/Текст/Жыйынтык:" cue
used in fewshot_test.py's one-shot prompt (that explicit cue, not the
few-shot-ness per se, is likely what made generation happen at all).
Training is on the full prompt+answer sequence, unmasked -- same
dataset_text_field="text" pattern as train.py/continue_train.py, kept
deliberately simple for this small experiment rather than adding a
custom prompt-masking collator.

Usage:
    python sft_train.py
"""

import json

from datasets import Dataset
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-lora-round3/checkpoint-950"
OUTPUT_DIR = "outputs/kazllm-kyrgyz-sft-v1"
DATA_PATH = "sft_data.jsonl"
MAX_SEQ_LENGTH = 768
TASK_INSTRUCTION = "Төмөнкү тексттеги фактыларга гана таянып, кыргызча 2-3 сүйлөм менен кыскача жыйынтыкта."


def build_example(context, answer, eos_token):
    prompt = (
        "Сен кыргыз тилинде гана жооп берүүчү жардамчысың. Казак тилинде эмес, так кыргыз тилинде жаз.\n\n"
        f"Тапшырма: {TASK_INSTRUCTION}\n"
        f"Текст: {context}\n"
        f"Жыйынтык:"
    )
    return prompt + " " + answer + eos_token


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    with open(DATA_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    texts = [build_example(r["context"], r["answer"], tokenizer.eos_token) for r in rows]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Loaded {len(dataset)} SFT examples")

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=UnslothTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=15,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            save_strategy="no",
            optim="adamw_8bit",
            dataloader_num_workers=0,
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR + "-final")
    tokenizer.save_pretrained(OUTPUT_DIR + "-final")
    print(f"Saved SFT adapter to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
