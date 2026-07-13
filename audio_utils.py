from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def wav_result_to_audio(result: bytes | str) -> tuple[np.ndarray, int]:
    if isinstance(result, str):
        wav_bytes = Path(result).read_bytes()
    else:
        wav_bytes = bytes(result)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype(np.float32, copy=False), sample_rate
