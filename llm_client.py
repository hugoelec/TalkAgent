from __future__ import annotations

import json
from typing import Any, Iterable

import requests


def join_url(base_url: str, endpoint: str) -> str:
    return f"{str(base_url or '').rstrip('/')}/{str(endpoint or '').lstrip('/')}"


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if content.get(key):
                return str(content[key])
    return ""


class LLMClient:
    def __init__(self, args: Any):
        self.args = args

    def resolve_temperature(self, temperature: float | None = None) -> float:
        value = self.args.llm_temperature if temperature is None else temperature
        return float(value)

    def build_messages(self, user_text: str, history: Iterable[dict[str, str]] | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.args.llm_disable_thinking:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Return only the final spoken reply. Do not output chain-of-thought, "
                        "reasoning, analysis, hidden thinking, or planning. Keep the reply concise "
                        "and conversational."
                    ),
                }
            )
        for item in history or []:
            user = item.get("user")
            assistant = item.get("assistant")
            if user:
                messages.append({"role": "user", "content": user})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": user_text})
        return messages

    def build_payload(
        self,
        messages: list[dict[str, str]],
        stream: bool,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.args.llm_model,
            "messages": messages,
            "max_tokens": int(self.args.llm_max_tokens),
            "stream": stream,
        }
        extra_body = getattr(self.args, "llm_extra_body", None)
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        payload["temperature"] = self.resolve_temperature(temperature)
        return payload

    def stream_reply(
        self,
        user_text: str,
        history: Iterable[dict[str, str]] | None = None,
        temperature: float | None = None,
    ):
        messages = self.build_messages(user_text, history=history)
        payload = self.build_payload(messages, stream=bool(self.args.llm_stream), temperature=temperature)
        url = join_url(self.args.llm_base_url, self.args.llm_endpoint)

        if not bool(self.args.llm_stream):
            response = requests.post(url, json=payload, timeout=float(self.args.llm_timeout_sec))
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                text = extract_message_text(message.get("content")) or extract_message_text(choices[0].get("text"))
                if text:
                    yield text
            return

        with requests.post(url, json=payload, timeout=float(self.args.llm_timeout_sec), stream=True) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = extract_message_text(delta.get("content")) or extract_message_text(delta.get("text"))
                if text:
                    yield text
