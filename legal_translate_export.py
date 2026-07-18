"""Экспорт переводчика в GGUF q4_k_m для Ollama.

Usage:  python legal_translate_export.py
Выход:  outputs/kazllm-legal-translate-v1-gguf/  (unsloth сам соберёт llama.cpp)
Затем:  ollama create myizam-translator -f Modelfile.translator
"""

import sys

from unsloth import FastLanguageModel

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "outputs/kazllm-legal-translate-v2-final"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/kazllm-legal-translate-v2-gguf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=1024,
    load_in_4bit=True,
)
model.save_pretrained_gguf(OUT, tokenizer, quantization_method="q4_k_m")
print(f"GGUF сохранён в {OUT}")
