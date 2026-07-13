from __future__ import annotations

import argparse
import base64
import io
import re
import wave
from typing import Tuple

import numpy as np
import requests


def pcm_float32_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.nan_to_num(pcm.astype(np.float32, copy=False))
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue()


LANGUAGE_LABELS = {
    "zh": "cn", "chinese": "cn", "mandarin": "cn", "cn": "cn",
    "en": "eng", "english": "eng", "eng": "eng",
    "yue": "yue", "cantonese": "yue",
    "ar": "ar", "arabic": "ar",
    "de": "de", "german": "de",
    "fr": "fr", "french": "fr",
    "es": "es", "spanish": "es",
    "pt": "pt", "portuguese": "pt",
    "id": "id", "indonesian": "id",
    "it": "it", "italian": "it",
    "ko": "ko", "korean": "ko",
    "ru": "ru", "russian": "ru",
    "th": "th", "thai": "th",
    "vi": "vi", "vietnamese": "vi",
    "ja": "ja", "japanese": "ja",
    "tr": "tr", "turkish": "tr",
    "hi": "hi", "hindi": "hi",
    "ms": "ms", "malay": "ms",
    "nl": "nl", "dutch": "nl",
    "sv": "sv", "swedish": "sv",
    "da": "da", "danish": "da",
    "fi": "fi", "finnish": "fi",
    "pl": "pl", "polish": "pl",
    "cs": "cs", "czech": "cs",
    "fil": "fil", "filipino": "fil",
    "fa": "fa", "persian": "fa",
    "el": "el", "greek": "el",
    "hu": "hu", "hungarian": "hu",
    "mk": "mk", "macedonian": "mk",
    "ro": "ro", "romanian": "ro",
}


def normalize_language_label(language: str) -> str:
    key = str(language or "").strip().lower()
    key = key.removeprefix("<|").removesuffix("|>")
    return LANGUAGE_LABELS.get(key, key)


def parse_asr_text(content: str) -> Tuple[str, str]:
    if not content:
        return "", ""

    text = str(content).strip()
    language = ""

    m = re.match(r"^\s*language\s+([^<\n\r]+)<asr_text>\s*", text, flags=re.IGNORECASE)
    if m:
        language = m.group(1)
        text = text[m.end():]
    else:
        m = re.match(r"^\s*([A-Za-z_-]+)<asr_text>\s*", text, flags=re.IGNORECASE)
        if m:
            language = m.group(1)
            text = text[m.end():]

    m = re.match(r"^\s*<\|([^|]+)\|>\s*", text)
    if m:
        language = language or m.group(1)
        text = text[m.end():]

    text = re.sub(r"<asr_text>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return normalize_language_label(language), text.strip()


class ASRClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def transcribe_segment(self, pcm: np.ndarray) -> Tuple[str, str]:
        wav_bytes = pcm_float32_to_wav_bytes(pcm, sample_rate=int(self.args.sample_rate))

        if self.args.method == "transcriptions":
            url = self.args.asr_base_url.rstrip("/") + "/audio/transcriptions"
            files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
            data = {"model": self.args.model}
            response = requests.post(url, data=data, files=files, timeout=float(self.args.timeout))
            self._raise_for_status(response)
            payload = response.json()
            return parse_asr_text(payload.get("text", ""))

        b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_uri = "data:audio/wav;base64," + b64
        url = self.args.asr_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": int(self.args.max_tokens),
        }
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=float(self.args.timeout),
        )
        self._raise_for_status(response)
        resp = response.json()
        content = resp["choices"][0]["message"].get("content", "")
        return parse_asr_text(content)

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text[:2000] if exc.response is not None else ""
            raise requests.HTTPError(f"{exc}; body={body}", response=exc.response) from exc
