from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any


@dataclass
class TalkerPromptResult:
    mode: str
    prompt: str
    history: list[dict[str, Any]]
    router_allowed: bool
    record_target: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class TalkerPromptSwitch:
    def __init__(self, engine: Any):
        self.engine = engine

    def build(
        self,
        user_text: str,
        *,
        use_control_prompt: bool,
        enable_router: bool,
        section_id_turn: bool,
        reader_mode: bool,
        reader_context: str,
        reader_summary_index: str,
        reader_context_override: str = "",
        reader_context_override_source: str = "",
    ) -> TalkerPromptResult:
        history = (
            []
            if reader_mode
            else list(self.engine.llm_history[-self.engine.raw_recent_rounds:])
            if self.engine.raw_recent_rounds > 0
            else []
        )

        if not use_control_prompt:
            return TalkerPromptResult(
                mode="section_id" if section_id_turn else "raw",
                prompt=user_text,
                history=history,
                router_allowed=False,
                record_target="",
            )

        if reader_mode:
            prompt, context_meta = self._build_reader_prompt(
                user_text,
                reader_context,
                reader_summary_index,
                reader_context_override,
                reader_context_override_source,
            )
            record = self._record_reader_prompt(user_text, reader_context, reader_summary_index, prompt, history, context_meta)
            return TalkerPromptResult(
                mode="reading_event" if str(reader_context_override or "").strip() else "reader",
                prompt=prompt,
                history=[],
                router_allowed=False,
                record_target="reader_control_history",
                meta={"reader_context": context_meta, "record": record},
            )

        prompt = self._build_normal_prompt(user_text)
        return TalkerPromptResult(
            mode="normal",
            prompt=prompt,
            history=history,
            router_allowed=bool(enable_router),
            record_target="control_prompt_history",
        )

    def _latest_reader_context_from_summary(self, analyze_resault: str, summary_index: str) -> tuple[str, dict[str, Any]]:
        full_context = str(analyze_resault or "").strip()
        summary_lines = [line.strip() for line in str(summary_index or "").splitlines() if line.strip()]
        meta: dict[str, Any] = {
            "source": "full",
            "summary_last_line": summary_lines[-1] if summary_lines else "",
            "line_start": None,
            "line_end": None,
            "full_chars": len(full_context),
            "context_chars": len(full_context),
        }
        if not full_context or not summary_lines:
            return full_context, meta

        last_line = summary_lines[-1]
        numbers = [int(item) for item in re.findall(r"\d+", last_line)]
        if len(numbers) < 3:
            return full_context, meta

        line_start = max(1, numbers[-2])
        line_end = max(line_start, numbers[-1])
        lines = str(analyze_resault or "").splitlines()
        if line_start > len(lines):
            return full_context, meta

        sliced = "\n".join(lines[line_start - 1: min(line_end, len(lines))]).strip()
        if not sliced:
            return full_context, meta

        meta.update(
            {
                "source": "summary_index_last_line",
                "line_start": line_start,
                "line_end": line_end,
                "context_chars": len(sliced),
            }
        )
        return sliced, meta

    def _build_reader_prompt(
        self,
        user_text: str,
        analyze_resault: str,
        summary_index: str = "",
        context_override: str = "",
        context_override_source: str = "",
    ) -> tuple[str, dict[str, Any]]:
        if str(context_override or "").strip():
            context = str(context_override or "").strip()
            context_meta: dict[str, Any] = {
                "source": "now_chapter",
                "override_source": context_override_source,
                "summary_last_line": "",
                "line_start": None,
                "line_end": None,
                "full_chars": len(str(analyze_resault or "").strip()),
                "context_chars": len(context),
            }
        else:
            context = str(analyze_resault or "").strip()
            context_meta = {
                "source": "analyze_resault_full",
                "summary_last_line": "",
                "line_start": None,
                "line_end": None,
                "full_chars": len(context),
                "context_chars": len(context),
            }
        return (
            "ReaderMode is active. Answer only from the uploaded book analysis context and the user's ASR question. "
            "Do not use normal chat history, memory summaries, persona preference history, or router control codes. "
            "If the answer is not supported by the book context, say that briefly.\n\n"
            "[AnalyzeResault]\n"
            f"{context if context else '(empty)'}\n\n"
            "[User ASR]\n"
            f"{user_text}"
        ), context_meta

    def _record_reader_prompt(
        self,
        user_text: str,
        reader_context: str,
        reader_summary_index: str,
        prompt: str,
        history: list[dict[str, Any]],
        context_meta: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "time": time.strftime("%H:%M:%S"),
            "user": user_text,
            "reader_context_chars": len(reader_context),
            "reader_summary_index_chars": len(reader_summary_index),
            "reader_context_source": context_meta.get("source", ""),
            "reader_context_override_source": context_meta.get("override_source", ""),
            "reader_context_line_start": context_meta.get("line_start"),
            "reader_context_line_end": context_meta.get("line_end"),
            "reader_context_slice_chars": context_meta.get("context_chars"),
            "summary_last_line": context_meta.get("summary_last_line", ""),
            "prompt_chars": len(prompt),
            "history_rounds": len(history),
            "prompt": prompt,
        }
        self.engine.reader_control_history.append(record)
        if self.engine.raw_history_rounds > 0:
            self.engine.reader_control_history = self.engine.reader_control_history[-self.engine.raw_history_rounds:]
        else:
            self.engine.reader_control_history = []
        return record

    def _build_normal_prompt(self, user_text: str) -> str:
        with self.engine.llm_control_prompt_lock:
            persona_prompt = str(
                getattr(
                    self.engine.args,
                    "llm_persona_prompt",
                    getattr(self.engine.args, "llm_control_prompt", ""),
                )
                or ""
            ).strip()
            tool_prompt = str(getattr(self.engine.args, "llm_tool_prompt", "") or "").strip()
            current_tokens = int(self.engine.control_prompt_current_tokens)
            delta_tokens = max(0, current_tokens - int(self.engine.control_prompt_token_since_inject))
            threshold = max(0, int(self.engine.control_prompt_inject_threshold))
            section_id_ready = (not self.engine.report_section_id_enabled) or self.engine.report_section_id_done
            first_inject = not self.engine.control_prompt_injected_once
            manual_inject = bool(self.engine.control_prompt_manual_inject)
            threshold_inject = delta_tokens > threshold
            should_inject_persona = section_id_ready and bool(persona_prompt) and (
                first_inject
                or manual_inject
                or threshold_inject
            )
            should_send_tool_prompt = section_id_ready and bool(tool_prompt)
            prompt_parts = []
            reasons = []
            if should_send_tool_prompt:
                prompt_parts.append(f"[Tool Prompt]\n{tool_prompt}")
                reasons.append("tool")
            if should_inject_persona:
                if first_inject:
                    persona_reason = "persona:first"
                elif manual_inject:
                    persona_reason = "persona:manual"
                else:
                    persona_reason = "persona:threshold"
                prompt_parts.append(f"[Persona]\n{persona_prompt}")
                reasons.append(persona_reason)
            control_prompt = "\n\n".join(prompt_parts).strip()
            if control_prompt:
                self.engine.control_prompt_history.append(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "reason": "+".join(reasons),
                        "current_tokens": current_tokens,
                        "delta_tokens": delta_tokens,
                        "threshold": threshold,
                        "persona_chars": len(persona_prompt),
                        "tool_prompt_chars": len(tool_prompt),
                        "prompt_chars": len(control_prompt),
                        "user": user_text,
                        "prompt": control_prompt,
                    }
                )
                if self.engine.raw_history_rounds > 0:
                    self.engine.control_prompt_history = self.engine.control_prompt_history[-self.engine.raw_history_rounds:]
                else:
                    self.engine.control_prompt_history = []
            if should_inject_persona:
                self.engine.control_prompt_injected_once = True
                self.engine.control_prompt_manual_inject = False
                self.engine.control_prompt_token_since_inject = current_tokens
        if not control_prompt:
            return user_text
        return f"{control_prompt}\n\n[User ASR]\n{user_text}"
