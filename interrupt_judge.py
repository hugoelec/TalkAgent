from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mic_engine import MicEngine


CONTROL_CODE_TEXT = re.compile(r"%%[^%\r\n]{1,80}%%")


def strip_control_codes(text: str) -> str:
    return CONTROL_CODE_TEXT.sub("", str(text or "")).strip()


class InterruptJudge:
    def __init__(self, engine: MicEngine):
        self.engine = engine

    def language_allowed(self, lang: str) -> bool:
        interrupt_languages = self.engine.interrupt_languages
        if not interrupt_languages:
            return True
        normalized = str(lang or "").strip().lower()
        if not normalized:
            return False
        aliases = {normalized}
        if normalized in {"en", "eng", "english"}:
            aliases.update({"en", "eng", "english"})
        if normalized in {"cn", "zh", "zh-cn", "zho", "chinese"}:
            aliases.update({"cn", "zh", "zh-cn", "zho", "chinese"})
        return bool(aliases & interrupt_languages)

    def analyze_decision(self, asr_text: str, lang: str) -> str:
        engine = self.engine
        with engine.lock:
            enabled = engine.echo_filter_enabled
            prompt = engine.interrupt_analyze_prompt.strip()
            current_tts = engine.current_tts_chunk.strip()
            prev_tts = engine.prev_tts_chunk.strip()

        if not enabled:
            return "interrupt"
        if not prompt:
            self.record_result(asr_text, "continue", f"missing_prompt lang={lang or '-'}")
            return "continue"

        tts_text = "\n".join(part for part in [prev_tts, current_tts] if part).strip()
        judge_prompt = (
            f"{prompt}\n\n"
            f"Current TTS playback:\n{tts_text or '(empty)'}\n\n"
            f"ASR recognized text:\n{asr_text}\n\n"
            "Answer only interrupt or continue."
        )
        try:
            with engine.lock:
                engine.last_llm_temperature = 0.0
                engine._push_log_locked(f"[LLM] send source=InterruptAnalyze mode=control temp=0 chars={len(judge_prompt)}")
            reply = "".join(engine.llm_client.stream_reply(judge_prompt, history=[], temperature=0.0)).strip()
        except Exception as exc:
            self.record_result(asr_text, "continue", f"error={exc!r} lang={lang or '-'}")
            return "continue"

        normalized = re.sub(r"[^a-z]", "", reply.lower())
        decision = "interrupt" if normalized.startswith("interrupt") else "continue"
        self.record_result(asr_text, decision, f"{reply} lang={lang or '-'}")
        return decision

    def record_result(self, asr_text: str, decision: str, reply: str) -> None:
        asr_preview = str(asr_text or "").replace("\n", " ")[:160]
        reply_preview = str(reply or "").replace("\n", " ")[:160]
        line = f"InterruptAnalyze: decision={decision} asr={asr_preview} llm={reply_preview}"
        record = {
            "asr": str(asr_text or ""),
            "decision": decision,
            "llm": str(reply or ""),
            "tts": "",
        }
        engine = self.engine
        with engine.lock:
            engine.interrupt_analyze_last = line
            record["tts"] = strip_control_codes(engine.current_tts_chunk)
            self._append_history_locked(record)
            engine._push_log_locked(f"[InterruptAnalyze] {line}")

    def _append_history_locked(self, record: dict[str, str]) -> None:
        engine = self.engine
        active_reply = engine._active_llm_reply_text_locked()
        active_user = engine.current_llm_user_text
        engine.memory.append_interrupt_record(active_user, active_reply, record)
