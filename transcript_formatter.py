from __future__ import annotations

from collections import Counter
from typing import Iterable

from phonetic_converter import normalize_language_code


def split_language_line(line: str) -> tuple[str, str]:
    raw = str(line or "").strip()
    if ":" not in raw:
        return "", raw

    maybe_lang, text = raw.split(":", 1)
    lang = normalize_language_code(maybe_lang)
    if not lang:
        return "", raw
    return lang, text.strip()


def detect_primary_language(lines: Iterable[str]) -> str:
    ordered_langs: list[str] = []
    for line in lines:
        lang, _text = split_language_line(line)
        if lang:
            ordered_langs.append(lang)
    if not ordered_langs:
        return ""

    counts = Counter(ordered_langs)
    return max(ordered_langs, key=lambda lang: (counts[lang], -ordered_langs.index(lang)))


def format_mixed_language_turn(lines: Iterable[str]) -> str:
    source_lines = [str(line or "").strip() for line in lines if str(line or "").strip()]
    primary_lang = detect_primary_language(source_lines)
    if not primary_lang:
        return "\n".join(source_lines)

    output: list[str] = []
    primary_marked = False
    for line in source_lines:
        lang, text = split_language_line(line)
        if not text:
            continue
        if lang == primary_lang:
            if primary_marked:
                output.append(text)
            else:
                output.append(f"{primary_lang}: {text}")
                primary_marked = True
        elif lang:
            output.append(f"({text})")
        else:
            output.append(text)

    return "\n".join(output)
