"""Ручная проверка переводчика myizam-translator (Ollama).

Разовый перевод:
    python translate.py ky2ru "Кызматкер эмгек келишимин бузууга укуктуу."
    python translate.py ru2ky "Работник вправе расторгнуть трудовой договор."

Интерактивный режим (направление переключается командами :ky и :ru, выход — :q):
    python translate.py
"""

import json
import sys
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "myizam-translator"

PROMPTS = {
    "ky2ru": "Тапшырма: Төмөнкү юридикалык текстти орус тилине так которуу.\nТекст: {text}\nКотормо:",
    "ru2ky": "Тапшырма: Төмөнкү юридикалык текстти кыргыз тилине так которуу.\nТекст: {text}\nКотормо:",
}


def translate(direction: str, text: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": PROMPTS[direction].format(text=text.strip()),
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 600},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["response"].strip()


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] in PROMPTS:
        print(translate(sys.argv[1], " ".join(sys.argv[2:])))
        return

    direction = "ky2ru"
    print("myizam-translator — интерактивный режим")
    print("направление: ky→ru  (:ru — переводить НА русский, :ky — на кыргызский, :q — выход)")
    while True:
        try:
            line = input(f"[{direction}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == ":q":
            break
        if line == ":ky":
            direction = "ru2ky"
            print("→ перевожу на кыргызский")
            continue
        if line == ":ru":
            direction = "ky2ru"
            print("→ перевожу на русский")
            continue
        print(translate(direction, line))


if __name__ == "__main__":
    main()
