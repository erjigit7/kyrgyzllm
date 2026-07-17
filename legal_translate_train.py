"""SFT переводчика ru<->ky на параллельном юридическом корпусе Myizam.

Зачем: в Myizam кыргызский путь = перевод вопроса ky->ru (мост) и перевод
ответа ru->ky. qwen2.5-7b пишет по-кыргызски мусор (eval Myizam: ky hit@5 40%
из-за моста). Данные: legal_translate_{train,val}.jsonl (15.8k выровненных
пар из официальных двуязычных текстов 8 кодексов, см. legal_translate_data.py)
— в ~30 раз больше, чем было у суммаризации, и провенанс чистый (официальные
тексты законов, никакого CC BY-NC).

Уроки sft_train_v2 соблюдены: старт с round3/checkpoint-950 (чистый языковой
чекпойнт, не поверх суммаризационного адаптера), attention/MLP-only LoRA уже
в чекпойнте, периодический eval, и ОБЯЗАТЕЛЬНЫЙ smoke-прогон перед полным:

    python legal_translate_train.py --max-steps 15   # smoke: время шага стабильно?
    python legal_translate_train.py                  # полный (1 эпоха)
"""

import argparse
import json
import pathlib

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

MODEL_PATH = "C:/hf/kazllm"
ADAPTER_PATH = "outputs/kazllm-kyrgyz-lora-round3/checkpoint-950"
OUTPUT_DIR = "outputs/kazllm-legal-translate-v1"
MAX_SEQ_LENGTH = 1024

PROMPT_KY2RU = "Тапшырма: Төмөнкү юридикалык текстти орус тилине так которуу.\nТекст: {src}\nКотормо:"
PROMPT_RU2KY = "Тапшырма: Төмөнкү юридикалык текстти кыргыз тилине так которуу.\nТекст: {src}\nКотормо:"


def build_examples(path: pathlib.Path, eos: str) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        texts.append(PROMPT_KY2RU.format(src=r["ky"]) + " " + r["ru"] + eos)
        texts.append(PROMPT_RU2KY.format(src=r["ru"]) + " " + r["ky"] + eos)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1, help="15 = smoke-прогон")
    ap.add_argument("--resume", default=None, help="путь к checkpoint-N для продолжения после обрыва")
    ap.add_argument("--save-steps", type=int, default=600)
    ap.add_argument("--adapter", default=ADAPTER_PATH, help="стартовый адаптер (для continue-прохода — v1-final)")
    ap.add_argument("--data", default="legal_translate_train.jsonl")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--bs", type=int, default=2, help="1 при занятой VRAM (Photoshop и т.п.)")
    ap.add_argument("--ga", type=int, default=4, help="держать bs*ga=8 для идентичного эффективного батча")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    here = pathlib.Path(__file__).parent
    from datasets import Dataset
    train_dataset = Dataset.from_dict({"text": build_examples(here / args.data, tokenizer.eos_token)})
    eval_dataset = Dataset.from_dict({"text": build_examples(here / "legal_translate_val.jsonl", tokenizer.eos_token)})
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=UnslothTrainingArguments(
            output_dir=args.output,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.ga,
            num_train_epochs=1,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps" if args.max_steps < 0 else "no",
            eval_steps=300,
            save_strategy="steps" if args.max_steps < 0 else "no",
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            dataloader_num_workers=0,
        ),
    )

    trainer.train(resume_from_checkpoint=args.resume)
    if args.max_steps < 0:
        model.save_pretrained(args.output + "-final")
        tokenizer.save_pretrained(args.output + "-final")
        print(f"Saved translate adapter to {args.output}-final")


if __name__ == "__main__":
    main()
