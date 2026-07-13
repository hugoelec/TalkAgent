from __future__ import annotations


def normalize_language_code(language: str | None) -> str:
    if not language:
        return ""
    return str(language).strip().lower().rstrip(":")


def convert_phonetic(language: str | None, text: str | None) -> str:
    lang = normalize_language_code(language)
    raw = "" if text is None else str(text)

    if not raw:
        return ""

    if lang == "th":
        try:
            from pythainlp.transliterate import romanize

            converted = romanize(raw, engine="royin")
            if converted:
                return converted
        except Exception:
            pass

    try:
        from unidecode import unidecode

        converted = unidecode(raw)
        if converted:
            return converted
    except Exception:
        pass

    return raw
