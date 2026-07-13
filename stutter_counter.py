from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


LANGUAGE_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_-]{0,15}\s*:\s*")
STUTTER_SEPARATOR_RE = re.compile(
    r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2026,.!?;:\uFF0C\u3002\uFF01\uFF1F\u3001\uFF1B\uFF1A()\[\]{}\"'\u201C\u201D\u2018\u2019]+"
)


def strip_language_prefix(text: str) -> str:
    return LANGUAGE_PREFIX_RE.sub("", str(text or ""), count=1).strip()


def build_stutter_text(lines: Iterable[str]) -> str:
    cleaned = [strip_language_prefix(line) for line in lines]
    return " ".join(line for line in cleaned if line)


def count_char_stuttering(text: str) -> int:
    compact = "".join(str(text or "").split())
    if len(compact) < 2:
        return 0

    count = 0
    prev_code = None
    run_len = 0

    for ch in compact:
        code = ord(ch)
        if code == prev_code:
            run_len += 1
        else:
            if run_len >= 2:
                count += run_len - 1
            prev_code = code
            run_len = 1

    if run_len >= 2:
        count += run_len - 1

    return count


def simple_tokenize_for_stutter(text: str) -> list[str]:
    text = str(text or "").lower()
    text = STUTTER_SEPARATOR_RE.sub(" ", text)
    return [tok for tok in text.split() if tok]


def count_token_stuttering(text: str) -> int:
    tokens = simple_tokenize_for_stutter(text)
    if len(tokens) < 2:
        return 0

    # Hesitation tokens can recur across the whole turn, not only adjacently.
    counts = Counter(tokens)
    return sum(token_count - 1 for token_count in counts.values() if token_count >= 2)


def count_stuttering_hybrid_simple(text: str) -> tuple[int, int, int]:
    char_count = count_char_stuttering(text)
    token_count = count_token_stuttering(text)
    final_count = max(char_count, token_count)
    return final_count, char_count, token_count


def calc_effective_long_silence_ms(
    base_long_silence_ms: int,
    current_asr_text: str,
    stutter_enabled: bool,
    stutter_delay_ms: int,
    stutter_delay_max_ms: int,
) -> tuple[int, int, int, int]:
    if not stutter_enabled:
        return base_long_silence_ms, 0, 0, 0

    stutter_count, char_count, token_count = count_stuttering_hybrid_simple(current_asr_text)
    extra_ms = min(stutter_count * stutter_delay_ms, stutter_delay_max_ms)
    effective_long_silence_ms = base_long_silence_ms + extra_ms
    return effective_long_silence_ms, stutter_count, char_count, token_count
