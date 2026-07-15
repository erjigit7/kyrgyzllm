"""Экспорт переводчика в GGUF q4_k_m для Ollama.

Usage:  python legal_translate_export.py
Выход:  outputs/kazllm-legal-translate-v1-gguf/  (unsloth сам соберёт llama.cpp)
Затем:  ollama create myizam-translator -f Modelfile.translator
"""

from unsloth import FastLanguageModel

ADAPTER = "outputs/kazllm-legal-translate-v1-final"
OUT = "outputs/kazllm-legal-translate-v1-gguf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=1024,
    load_in_4bit=True,
)
model.save_pretrained_gguf(OUT, tokenizer, quantization_method="q4_k_m")
print(f"GGUF сохранён в {OUT}")
