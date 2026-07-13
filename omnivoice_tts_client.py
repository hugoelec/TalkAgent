from __future__ import annotations

from typing import Any

import requests


def join_url(base_url: str, endpoint: str) -> str:
    return f"{str(base_url or '').rstrip('/')}/{str(endpoint or '').lstrip('/')}"


class OmniVoiceTTSClient:
    def __init__(self, args: Any):
        self.args = args

    def call_tts_api(self, text: str) -> bytes:
        url = join_url(self.args.tts_base_url, self.args.tts_endpoint)

        lock = getattr(self.args, "tts_settings_lock", None)
        if lock is not None:
            with lock:
                settings = {
                    "voice": self.args.tts_voice,
                    "language": self.args.tts_language,
                    "instruct": self.args.tts_instruct,
                    "speed": self.args.tts_speed,
                    "duration": self.args.tts_duration,
                    "num_step": self.args.tts_num_step,
                    "guidance_scale": self.args.tts_guidance_scale,
                    "denoise": self.args.tts_denoise,
                }
        else:
            settings = {
                "voice": self.args.tts_voice,
                "language": self.args.tts_language,
                "instruct": self.args.tts_instruct,
                "speed": self.args.tts_speed,
                "duration": self.args.tts_duration,
                "num_step": self.args.tts_num_step,
                "guidance_scale": self.args.tts_guidance_scale,
                "denoise": self.args.tts_denoise,
            }

        duration = float(settings["duration"])
        if duration <= 0:
            duration = None

        payload: dict[str, Any] = {
            "text": text,
            "voice": settings["voice"] or "auto",
            "language": settings["language"] or None,
            "instruct": settings["instruct"] or None,
            "speed": float(settings["speed"]),
            "duration": duration,
            "num_step": int(settings["num_step"]),
            "guidance_scale": float(settings["guidance_scale"]),
            "denoise": bool(settings["denoise"]),
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(
            url,
            json=payload,
            timeout=float(self.args.tts_timeout_sec),
        )
        response.raise_for_status()

        return response.content
