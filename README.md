# KyrgyzLLM

Fine-tuning KazLLM-8B so it actually writes **Kyrgyz** — on a single consumer GPU (RTX 4070 Ti, 12 GB).

## Why

No open model I could run locally writes decent Kyrgyz. The closest linguistic relative with open weights — [ISSAI's KazLLM-8B](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B) (Kazakh) — answers in Kazakh no matter what the prompt says, even when explicitly instructed "answer in Kyrgyz, not Kazakh". Generating Kyrgyz data through large-model APIs works, but thousands of examples get expensive fast for a pet project.

This model exists to unblock a downstream project: [RagGuard](https://github.com/erjigit7/ragguard) needed a dataset of truthful/hallucinated Kyrgyz summaries, and *something* had to write the truthful half. That something is this model — the final adapter generated all 3,000 grounded summaries in RagGuard's Kyrgyz dataset.

## How

Two stages, both QLoRA (4-bit) via [Unsloth](https://github.com/unslothai/unsloth):

1. **Continued pretraining** (`train.py`, `continue_train.py`) — raw-text causal LM on the [Kyrgyz News Corpus](https://huggingface.co/datasets/the-cramer-project/Kyrgyz_News_Corpus) (256k real news articles), three rounds totaling ~1,200 steps. This shifts the model's default output language from Kazakh to Kyrgyz. LoRA r=32 on attention/MLP projections only — see "lessons" below for why `embed_tokens` is *not* in the list.
2. **SFT for summarization** (`sft_train_v2.py`) — 508 real (article, summary) pairs from [XLSum's](https://huggingface.co/datasets/csebuetnlp/xlsum) Kyrgyz split, 3 epochs, prompt format `Тапшырма: ... / Текст: ... / Жыйынтык:`. Turns "writes Kyrgyz" into "reliably summarizes the given text without inventing facts".

Final adapter: `outputs/kazllm-kyrgyz-sft-v2-final` (adapters are gitignored — retrain with the scripts, ~13h for stage 1 on a 4070 Ti, ~3 min for stage 2).

## Lessons that cost real time

- **Don't LoRA-train `embed_tokens`/`lm_head` on a 128k-vocab 8B model with 12 GB VRAM.** Unsloth silently offloads the embedding matrix to disk and per-step time grows unboundedly (28s → 373s and climbing). Attention/MLP-only LoRA still fixed the output language.
- **Smoke-test ~15 steps, not 3.** Three steps wasn't enough to catch the problem above.
- **22 hand-written SFT examples are worse than none** (`sft_train.py`, kept as a record): the model memorized them (train loss 0.01) and started hallucinating on held-out text — inventing a person's name, a tournament number. 508 real XLSum pairs fixed it.
- **The SFT'd model cannot lie on request.** Asked to "summarize but deliberately corrupt one fact", it returns output byte-identical to the truthful prompt — the single-task SFT is that narrow. Downstream negative examples had to be generated programmatically instead.
- This is a **one-task tool**, not a chatbot. It summarizes Kyrgyz text given in the trained prompt format; it does not chat.

## Try it

```bash
# browser UI (wraps your text in the trained summarize template)
python chat_ui.py --adapter outputs/kazllm-kyrgyz-sft-v2-final

# CLI
python chat.py --adapter outputs/kazllm-kyrgyz-sft-v2-final
```

Base model weights are expected at `C:/hf/kazllm` (gated repo — request access on Hugging Face, then `hf download issai/LLama-3.1-KazLLM-1.0-8B --local-dir C:/hf/kazllm`).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install unsloth
# unsloth silently replaces torch with a CPU build on Windows — reinstall CUDA torch AFTER it:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "datasets==5.0.0" gradio
```

## License

The code in this repo is free to use. The **base model and the fine-tuned adapter are not**: [ISSAI's KazLLM-8B](https://huggingface.co/issai/LLama-3.1-KazLLM-1.0-8B) is licensed under Meta's Llama 3.1 License plus **CC BY-NC 4.0 (non-commercial only)**, and that restriction carries over to this fine-tune. This project is a personal, non-commercial portfolio project — attribution to ISSAI for KazLLM-8B. Commercial use of the model or adapter would need a separate license from ISSAI.
