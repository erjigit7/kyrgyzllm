"""Параллельный юридический датасет ru↔ky из корпуса Myizam.

Источник: D:/claude_projects/myizam/data/chunks/*_{ru,kg}.jsonl — одни и те же
статьи 8 кодексов КР в официальных русской и кыргызской версиях (cbd.minjust.gov.kg).
Выравнивание по (кодекс, номер статьи):

- если у статьи совпадает число абзацев в обеих версиях — пары АБЗАЦ↔АБЗАЦ
  (короткие, точные; основной объём);
- иначе — пара СТАТЬЯ↔СТАТЬЯ целиком, если обе стороны короче лимита.

Фильтры: длина каждой стороны 40..900 символов, соотношение длин 0.4..2.5
(отсекает битые пары: пропущенные абзацы, слитые Word-артефакты).
Сплит train/val — по хешу (кодекс, статья): пара никогда не straddle'ит сплит.

Выход: legal_translate_{train,val}.jsonl с полями src_lang, src, tgt —
обе стороны каждой пары дают ДВА примера (ky→ru и ru→ky).

Usage:  python legal_translate_data.py
"""

import glob
import hashlib
import json
import pathlib
from collections import defaultdict

MYIZAM_CHUNKS = "D:/claude_projects/myizam/data/chunks"
OUT_DIR = pathlib.Path(__file__).parent

MIN_LEN, MAX_LEN = 40, 900
RATIO_MIN, RATIO_MAX = 0.4, 2.5
VAL_FRACTION = 20  # 1/20 статей ≈ 5%


def load_articles(lang: str) -> dict[tuple[str, str], list[str]]:
    """(law, article) → полный текст статьи как список абзацев (склейка частей чанков)."""
    articles: dict[tuple[str, str], list[str]] = defaultdict(list)
    for file in sorted(glob.glob(f"{MYIZAM_CHUNKS}/*_{lang}.jsonl")):
        for line in pathlib.Path(file).read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            if c["ArticleNumber"] is None:  # преамбулы не выравниваем
                continue
            key = (c["LawCode"], c["ArticleNumber"])
            articles[key].extend(p for p in c["Text"].split("\n") if p.strip())
    return articles


def good_pair(a: str, b: str) -> bool:
    if not (MIN_LEN <= len(a) <= MAX_LEN and MIN_LEN <= len(b) <= MAX_LEN):
        return False
    ratio = len(a) / len(b)
    return RATIO_MIN <= ratio <= RATIO_MAX


def main() -> None:
    ru = load_articles("ru")
    kg = load_articles("kg")
    common = sorted(set(ru) & set(kg))
    print(f"статей ru={len(ru)}, kg={len(kg)}, общих={len(common)}")

    pairs: list[tuple[tuple[str, str], str, str]] = []  # (key, ru_text, kg_text)
    para_aligned = art_aligned = skipped = 0
    for key in common:
        pr, pk = ru[key], kg[key]
        if len(pr) == len(pk):
            for a, b in zip(pr, pk):
                if good_pair(a, b):
                    pairs.append((key, a, b))
                    para_aligned += 1
        else:
            a, b = " ".join(pr), " ".join(pk)
            if good_pair(a, b):
                pairs.append((key, a, b))
                art_aligned += 1
            else:
                skipped += 1
    print(f"пар: абзацных={para_aligned}, статейных={art_aligned}, статей пропущено={skipped}")

    train, val = [], []
    for key, ru_text, kg_text in pairs:
        bucket = val if int(hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest(), 16) % VAL_FRACTION == 0 else train
        bucket.append({"law": key[0], "article": key[1], "ru": ru_text, "ky": kg_text})

    for name, rows in (("train", train), ("val", val)):
        out = OUT_DIR / f"legal_translate_{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} пар (×2 направления = {2 * len(rows)} примеров) → {out.name}")


if __name__ == "__main__":
    main()
