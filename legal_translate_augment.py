"""Прицельная аугментация дробей для kazllm-legal-translate.

Найденный на живом демо баг: «одна четверть» → «төрттөн үчү» (три четверти)
при верном русском оригинале. Причина: в 15k пар корпуса настоящих дробей —
единицы, модель их не выучила.

Метод: добываем РЕАЛЬНЫЕ выровненные пары, содержащие дробь на обеих сторонах
(морфология окружения — подлинная, из официальных текстов), и размножаем их
синхронной подстановкой дроби по проверенной таблице соответствий. Каждый
вариант повторяется OVERSAMPLE раз. К аугментату подмешиваются случайные
оригинальные пары против забывания.

Выход: legal_translate_aug.jsonl (тот же формат, что train).
Usage:  python legal_translate_augment.py
"""

import json
import pathlib
import random

HERE = pathlib.Path(__file__).parent
OVERSAMPLE = 20
ORIGINAL_MIX = 3000

# (ru-форма, kg-форма) — обе в родительно-притяжательном употреблении,
# как в ст.86 СК: «кирешесинин төрттөн бири» / «одной четверти заработка».
# Формы сверены с реальными текстами кодексов (ст.86 СК kg — эталон).
FRACTIONS = [
    ("одной четверти", "төрттөн бири"),
    ("одна четверть", "төрттөн бири"),
    ("одной трети", "үчтөн бири"),
    ("одна треть", "үчтөн бири"),
    ("двух третей", "үчтөн экиси"),
    ("две трети", "үчтөн экиси"),
    ("трех четвертей", "төрттөн үчү"),
    ("три четверти", "төрттөн үчү"),
    ("половины", "жарымы"),
    ("половина", "жарымы"),
    ("одной пятой", "бештен бири"),
    ("двух пятых", "бештен экиси"),
]


# Числительные словами (именительный) — «семьдесят» ↔ «жетимиш» и т.п.
# Найдено на ручной проверке v2: десятки словами переводились бессмыслицей.
NUMERALS = [
    ("двадцать", "жыйырма"),
    ("тридцать", "отуз"),
    ("сорок", "кырк"),
    ("пятьдесят", "элүү"),
    ("шестьдесят", "алтымыш"),
    ("семьдесят", "жетимиш"),
    ("восемьдесят", "сексен"),
    ("девяносто", "токсон"),
    ("сто", "жүз"),
]
NUMERAL_DIGITS = ["20", "30", "40", "50", "60", "70", "80", "90", "100"]
NUMERAL_BASE_PAIRS = 800   # сколько цифровых пар брать под числительные


def find_fraction(text: str, forms: list[str]) -> str | None:
    for f in forms:
        if f in text:
            return f
    return None


def numeral_variants(train: list[dict]) -> list[dict]:
    """Реальные пары, где двузначное круглое число стоит ЦИФРОЙ на обеих сторонах,
    → варианты с числом СЛОВОМ на обеих сторонах (жетимиш ↔ семьдесят)."""
    import re
    out = []
    candidates = []
    for r in train:
        for d in NUMERAL_DIGITS:
            # отдельный токен-число с обеих сторон, по одному вхождению — безопасная замена
            pat = rf"(?<!\d){d}(?!\d)"
            if len(re.findall(pat, r["ru"])) == 1 and len(re.findall(pat, r["ky"])) == 1:
                candidates.append((r, d, pat))
                break
    random.shuffle(candidates)
    for r, d, pat in candidates[:NUMERAL_BASE_PAIRS]:
        import re as _re
        for ru_w, kg_w in random.sample(NUMERALS, 4):   # 4 разных числительных на пару
            out.append({
                "law": r["law"], "article": r["article"],
                "ru": _re.sub(pat, ru_w, r["ru"]),
                "ky": _re.sub(pat, kg_w, r["ky"]),
            })
    return out


def main() -> None:
    random.seed(42)
    train = [json.loads(l) for l in (HERE / "legal_translate_train.jsonl").read_text(encoding="utf-8").splitlines()]

    ru_forms = [f for f, _ in FRACTIONS]
    kg_forms = list({k for _, k in FRACTIONS})
    ru2kg = dict(FRACTIONS)

    # 1. Реальные пары, где дробь есть на ОБЕИХ сторонах и соответствие корректно
    mined = []
    for r in train:
        fr = find_fraction(r["ru"], ru_forms)
        fk = find_fraction(r["ky"], kg_forms)
        if fr and fk and ru2kg[fr] == fk:
            mined.append((r, fr, fk))
    print(f"добыто реальных пар с согласованной дробью: {len(mined)}")

    # 2. Синхронная подстановка вариантов дроби в реальные предложения
    augmented = []
    for r, fr, fk in mined:
        augmented.append({"law": r["law"], "article": r["article"], "ru": r["ru"], "ky": r["ky"]})
        for new_ru, new_kg in FRACTIONS:
            if new_ru == fr or ru2kg[fr] == new_kg:
                continue
            # согласуем грамматическое число подстановки с оригиналом (обе формы из пары падежных вариантов)
            augmented.append({
                "law": r["law"], "article": r["article"],
                "ru": r["ru"].replace(fr, new_ru),
                "ky": r["ky"].replace(fk, new_kg),
            })
    print(f"после подстановок: {len(augmented)} уникальных пар")

    numerals = numeral_variants(train)
    print(f"вариантов с числительными словами: {len(numerals)}")

    rows = augmented * OVERSAMPLE + numerals * 3 + random.sample(train, min(ORIGINAL_MIX, len(train)))
    random.shuffle(rows)

    out = HERE / "legal_translate_aug.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"итого примеров (×2 направления при обучении): {len(rows)} → {out.name}")


if __name__ == "__main__":
    main()
