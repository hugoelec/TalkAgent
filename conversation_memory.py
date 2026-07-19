from __future__ import annotations

import re
import time
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mic_engine import MicEngine


CONTROL_CODE_TEXT = re.compile(r"%%[^%\r\n]{1,80}%%")


def strip_control_codes(text: str) -> str:
    return CONTROL_CODE_TEXT.sub("", str(text or "")).strip()


class ConversationMemory:
    def __init__(self, engine: "MicEngine") -> None:
        self.engine = engine
        self.reset_generation = 0
        self.llm_history: list[dict[str, Any]] = []
        self.control_prompt_history: list[dict[str, Any]] = []
        self.router_history: list[dict[str, Any]] = []
        self.reader_control_history: list[dict[str, Any]] = []
        self.control_prompt_current_tokens = 0
        self.control_prompt_token_since_inject = 0
        self.control_prompt_injected_once = False
        self.control_prompt_manual_inject = False
        self.memory_round_current = 0
        self.memory_round_since_extract = 0
        self.memory_extract_freq = max(0, int(getattr(engine.args, "memory_extract_freq", 0)))
        self.memory_extract_rounds = max(0, int(getattr(engine.args, "memory_extract_rounds", 0)))

    def trim_all(self) -> None:
        raw_history_rounds = max(0, int(self.engine.raw_history_rounds))
        if raw_history_rounds > 0:
            self.llm_history = self.llm_history[-raw_history_rounds:]
            self.control_prompt_history = self.control_prompt_history[-raw_history_rounds:]
            self.router_history = self.router_history[-raw_history_rounds:]
            self.reader_control_history = self.reader_control_history[-raw_history_rounds:]
        else:
            self.llm_history = []
            self.control_prompt_history = []
            self.router_history = []
            self.reader_control_history = []

    def recent_llm_history(self) -> list[dict[str, Any]]:
        raw_recent_rounds = max(0, int(self.engine.raw_recent_rounds))
        if raw_recent_rounds <= 0:
            return []
        return list(self.llm_history[-raw_recent_rounds:])

    def history_view(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw_history_rounds = max(0, int(self.engine.raw_history_rounds))
        if raw_history_rounds <= 0:
            return []
        return list(history[-raw_history_rounds:])

    def snapshot_fields(self) -> dict[str, Any]:
        return {
            "llm_history": self.history_view(self.llm_history),
            "control_prompt_history": self.history_view(self.control_prompt_history),
            "router_history": self.history_view(self.router_history),
            "reader_control_history": self.history_view(self.reader_control_history),
            "control_prompt_current_tokens": self.control_prompt_current_tokens,
            "control_prompt_token_since_inject": self.control_prompt_token_since_inject,
            "control_prompt_delta_tokens": self.control_prompt_delta_tokens,
            "control_prompt_manual_inject": self.control_prompt_manual_inject,
            "memory_round_current": self.memory_round_current,
            "memory_round_since_extract": self.memory_round_since_extract,
            "memory_extract_freq": self.memory_extract_freq,
            "memory_extract_rounds": self.memory_extract_rounds,
        }

    @property
    def control_prompt_delta_tokens(self) -> int:
        return max(0, self.control_prompt_current_tokens - self.control_prompt_token_since_inject)

    def add_control_prompt_tokens(self, count: int) -> None:
        self.control_prompt_current_tokens += max(0, int(count))

    def mark_manual_inject(self) -> None:
        self.control_prompt_token_since_inject = int(self.control_prompt_current_tokens)
        self.control_prompt_manual_inject = True

    def mark_persona_injected(self, token_position: int) -> None:
        self.control_prompt_injected_once = True
        self.control_prompt_manual_inject = False
        self.control_prompt_token_since_inject = int(token_position)

    def set_memory_extract_config(self, freq: int | None = None, rounds: int | None = None) -> None:
        if freq is not None:
            self.memory_extract_freq = max(0, int(freq))
            self.engine.args.memory_extract_freq = self.memory_extract_freq
        if rounds is not None:
            self.memory_extract_rounds = max(0, int(rounds))
            self.engine.args.memory_extract_rounds = self.memory_extract_rounds

    def upsert_llm_history(self, user_text: str, assistant_text: str, state: str) -> None:
        user = strip_control_codes(user_text)
        assistant = strip_control_codes(assistant_text)
        history_state = strip_control_codes(state)
        if not user or not assistant:
            return
        if self.llm_history and self.llm_history[-1].get("user") == user:
            self.llm_history[-1]["assistant"] = assistant
            self.llm_history[-1]["state"] = history_state
            return
        self.llm_history.append({"user": user, "assistant": assistant, "state": history_state})
        self.memory_round_current += 1
        if self.memory_round_current > self.engine.raw_recent_rounds:
            self.memory_round_since_extract += 1
            if self.memory_extract_freq > 0 and self.memory_round_since_extract >= self.memory_extract_freq:
                self.memory_round_since_extract = 0
        self.trim_all()

    def update_last_state_for_user(self, user_text: str, state: str) -> None:
        user = strip_control_codes(user_text)
        if not user or not self.llm_history:
            return
        if self.llm_history[-1].get("user") == user:
            self.llm_history[-1]["state"] = strip_control_codes(state)

    def append_control_prompt_record(self, record: dict[str, Any]) -> None:
        self.control_prompt_history.append(record)
        self.trim_all()

    def append_reader_control_record(self, record: dict[str, Any]) -> None:
        self.reader_control_history.append(record)
        self.trim_all()

    def append_router_record(self, record: dict[str, Any]) -> None:
        self.router_history.append(record)
        self.trim_all()

    def append_interrupt_record(self, active_user: str, active_reply: str, record: dict[str, str]) -> None:
        if not (active_reply and active_user):
            return
        self.upsert_llm_history(active_user, active_reply, "speaking")
        if not self.llm_history or self.llm_history[-1].get("user") != strip_control_codes(active_user):
            return
        interrupts = self.llm_history[-1].setdefault("interrupts", [])
        if isinstance(interrupts, list):
            interrupts.append(record)

    def build_control_prompt_record(
        self,
        *,
        reasons: list[str],
        threshold: int,
        persona_prompt: str,
        tool_prompt: str,
        control_prompt: str,
        user_text: str,
    ) -> dict[str, Any]:
        return {
            "time": time.strftime("%H:%M:%S"),
            "reason": "+".join(reasons),
            "current_tokens": self.control_prompt_current_tokens,
            "delta_tokens": self.control_prompt_delta_tokens,
            "threshold": threshold,
            "persona_chars": len(persona_prompt),
            "tool_prompt_chars": len(tool_prompt),
            "prompt_chars": len(control_prompt),
            "user": user_text,
            "prompt": control_prompt,
        }

    def reset_history(self) -> None:
        engine = self.engine
        with engine.lock:
            self.reset_generation += 1
            if engine.current_llm_cancel_event is not None:
                engine.current_llm_cancel_event.set()
            if engine.tts_cancel_event is not None:
                engine.tts_cancel_event.set()
            engine._drain_pending_speech_turns("[Skipped by memory reset]")
            engine.transcript = []
            engine.silence_counters = []
            engine.reply_results = []
            engine.tts_logs = []
            engine.prev_tts_chunk = ""
            engine.current_tts_chunk = ""
            engine.llm_pending_stop_at = ""
            self.llm_history = []
            self.control_prompt_history = []
            self.router_history = []
            self.reader_control_history = []
            self.control_prompt_current_tokens = 0
            self.control_prompt_token_since_inject = 0
            self.control_prompt_injected_once = False
            self.control_prompt_manual_inject = False
            self.memory_round_current = 0
            self.memory_round_since_extract = 0
            engine.last_asr_text = ""
            engine.last_asr_duration_ms = 0
            engine.last_asr_update_ts = None
            engine.effective_long_silence_ms = engine.long_silence_ms
            engine.remaining_long_silence_ms = engine.long_silence_ms
            engine.stuttering_count = 0
            engine.char_stutter_count = 0
            engine.token_stutter_count = 0
            engine.interupt_count = engine.interupt_threshold
            engine.interupt_qasr_parts = []
            engine.interupt_qasr_text = ""
            engine.interupt_last_volume = 0.0
            engine.interrupt_analyze_last = "InterruptAnalyze: idle"
            engine.errors = []

    def is_current_generation(self, generation: int | None) -> bool:
        return generation is None or generation == self.reset_generation
