from __future__ import annotations

import argparse
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Dict, List

import numpy as np
import sounddevice as sd
import yaml

from asr_client import ASRClient
from conversation_memory import ConversationMemory
from interrupt_judge import InterruptJudge
from llm_client import LLMClient
from omnivoice_tts_client import OmniVoiceTTSClient
from phonetic_converter import convert_phonetic, normalize_language_code
from stutter_counter import build_stutter_text, calc_effective_long_silence_ms
from talker_prompt_switch import TalkerPromptSwitch
from transcript_formatter import format_mixed_language_turn
from tts_pipeline import TtsPipeline


CONTROL_CODE_TEXT = re.compile(r"%%[^%\r\n]{1,80}%%")


def calc_volume(float32: np.ndarray) -> float:
    if float32.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(float32, dtype=np.float32), dtype=np.float32)))


def strip_control_codes(text: str) -> str:
    return CONTROL_CODE_TEXT.sub("", str(text or "")).strip()


def estimate_token_count(text: str) -> int:
    tokens = 0
    ascii_run = 0
    for char in text:
        code = ord(char)
        if char.isspace():
            if ascii_run:
                tokens += max(1, (ascii_run + 3) // 4)
                ascii_run = 0
            continue
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            if ascii_run:
                tokens += max(1, (ascii_run + 3) // 4)
                ascii_run = 0
            tokens += 1
        elif char.isascii():
            ascii_run += 1
        else:
            if ascii_run:
                tokens += max(1, (ascii_run + 3) // 4)
                ascii_run = 0
            tokens += 1
    if ascii_run:
        tokens += max(1, (ascii_run + 3) // 4)
    return tokens


class MicEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.asr_client = ASRClient(args)
        self.llm_client = LLMClient(args)
        self.tts_client = OmniVoiceTTSClient(args)
        self.tts_pipeline = TtsPipeline(self)
        self.memory = ConversationMemory(self)
        self.talker_prompt_switch = TalkerPromptSwitch(self)
        config_path = Path(str(getattr(args, "config", "config.yaml"))).resolve()
        self.control_prompt_config_path = config_path.parent / "ControlPrompt.yaml"
        self.router_config_paths = [
            config_path.parent / "RouterConfig.yaml",
            config_path.parent / "RouterConfigh.yaml",
        ]
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.segment_queue: "queue.Queue[tuple[np.ndarray, bool] | None]" = queue.Queue()
        self.speech_turn_queue: "queue.Queue[Dict[str, Any] | None]" = queue.Queue()
        self.speech_turn_active = False
        self.sender_thread = threading.Thread(target=self._sender_loop, name="asr-sender", daemon=True)
        self.turn_thread = threading.Thread(target=self._turn_loop, name="long-silence-turn", daemon=True)
        self.speech_turn_thread = threading.Thread(
            target=self._speech_turn_loop,
            name="speech-turn-worker",
            daemon=True,
        )

        self.stream: sd.InputStream | None = None
        self.running = False
        self.status = "idle"
        self.volume = 0.0
        self.segment = "idle"
        self.chunk_count = 0
        self.transcript: List[str] = []
        self.silence_counters: List[str] = []
        self.reply_results: List[Dict[str, str]] = []
        self.tts_logs: List[str] = []
        self.prev_tts_chunk = ""
        self.current_tts_chunk = ""
        self.llm_pending_stop_at = ""
        self.reader_mode = False
        self.reading_mode = False
        self.reader_analyze_resault = ""
        self.reader_summary_index = ""
        self.reader_now_chapter = ""
        self.talker_prompt_mode = "normal"
        self.last_llm_temperature = float(getattr(self.args, "llm_temperature", 0.0))
        self.errors: List[str] = []
        self.llm_control_prompt_lock = threading.Lock()
        self.control_prompt_inject_threshold = max(0, int(getattr(self.args, "llm_control_prompt_inject_threshold", 1000000)))
        self.raw_history_rounds = max(0, int(getattr(self.args, "raw_history_rounds", 10)))
        self.raw_recent_rounds = max(0, int(getattr(self.args, "raw_recent_rounds", 3)))
        self.report_section_id_enabled = bool(getattr(self.args, "llm_report_section_id", False))
        self.report_section_id_inflight = False
        self.report_section_id_done = False
        self.last_asr_text = ""
        self.last_asr_duration_ms = 0
        self.asr_inflight = False
        self.llm_inflight = False
        self.tts_playing = False
        self.interupt_threshold = max(0, int(getattr(self.args, "tts_interupt_threshold", 2)))
        self.interupt_volume = max(0.0, float(getattr(self.args, "tts_interupt_volume", 0.0)))
        self.interupt_early_release = max(0.0, float(getattr(self.args, "interupt_early_release", 0.0)))
        self.interupt_switch_delay = max(0.0, float(getattr(self.args, "interupt_switch_delay", 0.5)))
        self.interupt_switch_until_ts = 0.0
        self.tts_early_release_active = False
        self.active_voice_start_volume = float(self.args.voice_start_volume)
        self.interupt_count = self.interupt_threshold
        self.interupt_qasr_parts: List[str] = []
        self.interupt_qasr_text = ""
        self.interupt_last_volume = 0.0
        self.echo_filter_enabled = bool(getattr(self.args, "echo_filter_enabled", False))
        self.interrupt_languages = set(getattr(self.args, "interrupt_languages", set()) or set())
        self.interrupt_analyze_prompt = str(getattr(self.args, "interrupt_analyze_prompt", "") or "")
        self.interrupt_analyze_last = "InterruptAnalyze: idle"
        self.interrupt_judge = InterruptJudge(self)
        self.current_llm_cancel_event: threading.Event | None = None
        self.current_llm_user_text = ""
        self.current_llm_reply_index: int | None = None
        self.tts_cancel_event: threading.Event | None = None

        self.recording = False
        self.tail_mode = False
        self.acc: List[np.ndarray] = []
        self.acc_len = 0
        self.silence_acc_ms = 0.0
        self.tail_acc_ms = 0.0
        self.segment_interupt_mode = False
        self.last_asr_update_ts: float | None = None
        self.long_silence_ms = int(float(self.args.long_silence_ms))
        self.effective_long_silence_ms = self.long_silence_ms
        self.remaining_long_silence_ms = self.long_silence_ms
        self.stuttering_count = 0
        self.char_stutter_count = 0
        self.token_stutter_count = 0

        self.block_size = max(1, int(round(self.args.sample_rate * self.args.block_ms / 1000.0)))
        self.sender_thread.start()
        self.turn_thread.start()
        self.speech_turn_thread.start()

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self._reset_segment_locked()
            self.status = f"starting {self.args.sample_rate} Hz"
            self.errors = self.errors[-40:]

        try:
            stream = sd.InputStream(
                samplerate=int(self.args.sample_rate),
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            stream.start()
        except Exception as exc:
            self._push_error(f"mic start error: {exc!r}")
            with self.lock:
                self.running = False
                self.status = "error"
            raise

        with self.lock:
            self.stream = stream
            self.running = True
            self.status = f"recording {self.args.sample_rate} Hz"
            self.segment = "idle"
            should_report_section_id = (
                self.report_section_id_enabled
                and not self.report_section_id_done
                and not self.report_section_id_inflight
            )
            if should_report_section_id:
                self.report_section_id_inflight = True
                self._push_log_locked("[SectionID] start mic handshake queued")
        if should_report_section_id:
            threading.Thread(
                target=self._start_section_id_turn,
                name="llm-section-id-handshake",
                daemon=True,
            ).start()

    def stop(self) -> None:
        with self.lock:
            stream = self.stream
            self.stream = None
            was_running = self.running
            self.running = False
            self.status = "stopped" if was_running else "idle"

        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

        self._flush_on_stop()
        with self.lock:
            self._reset_segment_locked()
            self.volume = 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            self._active_voice_start_volume_locked(now)
            return {
                "running": self.running,
                "status": self.status,
                "volume": self.volume,
                "segment": self.segment,
                "chunk_count": self.chunk_count,
                "last_asr_text": self.last_asr_text,
                "last_asr_duration_ms": self.last_asr_duration_ms,
                "transcript": list(self.transcript),
                "silence_counters": list(self.silence_counters),
                "reply_results": list(self.reply_results),
                **self.memory.snapshot_fields(),
                "reader_mode": self.reader_mode,
                "reading_mode": self.reading_mode,
                "reader_context_chars": len(self.reader_analyze_resault or ""),
                "reader_summary_index_chars": len(self.reader_summary_index or ""),
                "reader_now_chapter_chars": len(self.reader_now_chapter or ""),
                "talker_prompt_mode": self.talker_prompt_mode,
                "llm_temperature": float(self.args.llm_temperature),
                "last_llm_temperature": float(self.last_llm_temperature),
                "raw_history_rounds": self.raw_history_rounds,
                "raw_recent_rounds": self.raw_recent_rounds,
                "tts_logs": list(self.tts_logs),
                "errors": list(self.errors),
                "long_silence_ms": self.long_silence_ms,
                "effective_long_silence_ms": self.effective_long_silence_ms,
                "remaining_long_silence_ms": self.remaining_long_silence_ms,
                "stuttering_count": self.stuttering_count,
                "char_stutter_count": self.char_stutter_count,
                "token_stutter_count": self.token_stutter_count,
                "asr_inflight": self.asr_inflight,
                "llm_inflight": self.llm_inflight,
                "asr_queue_size": self.segment_queue.qsize(),
                "speech_turn_active": self.speech_turn_active,
                "speech_turn_queue_size": self.speech_turn_queue.qsize(),
                "tts_playing": self.tts_playing,
                "interupt_mode": self._interupt_mode_active_locked(now),
                "interupt_threshold": self.interupt_threshold,
                "interupt_volume": self.interupt_volume,
                "interupt_early_release": self.interupt_early_release,
                "tts_early_release_active": self.tts_early_release_active,
                "interupt_switch_delay": self.interupt_switch_delay,
                "interupt_switch_remaining": self._interupt_switch_remaining_locked(now),
                "active_voice_start_volume": self.active_voice_start_volume,
                "interupt_count": self.interupt_count,
                "interupt_qasr_text": self.interupt_qasr_text,
                "interupt_last_volume": self.interupt_last_volume,
                "mic_capture_backend": "sounddevice",
                "browser_aec_requested": False,
                "browser_aec_active": False,
                "browser_aec_note": "Browser echoCancellation is not in the active capture path.",
                "echo_filter_enabled": self.echo_filter_enabled,
                "interrupt_languages": sorted(self.interrupt_languages),
                "interrupt_analyze_last": self.interrupt_analyze_last,
                "control_prompt_inject_threshold": self.control_prompt_inject_threshold,
            }

    def shutdown(self) -> None:
        self.stop()
        self.shutdown_event.set()
        self.segment_queue.put(None)
        self.speech_turn_queue.put(None)
        self.sender_thread.join(timeout=1.0)
        self.turn_thread.join(timeout=1.0)
        self.speech_turn_thread.join(timeout=1.0)

    def _audio_callback(self, indata: np.ndarray, frames: int, callback_time: Any, status: sd.CallbackFlags) -> None:
        del callback_time
        if status:
            self._push_error(f"audio callback status: {status}")

        with self.lock:
            if not self.running:
                return

        block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        if frames != block.size:
            block = block[:frames]
        self._process_block(block)

    def _process_block(self, block: np.ndarray) -> None:
        volume = calc_volume(block)
        frame_ms = block.size / float(self.args.sample_rate) * 1000.0

        with self.lock:
            self.volume = volume
            now = time.monotonic()
            voice_start_volume = self._active_voice_start_volume_locked(now)
            interupt_mode_active = self._interupt_mode_active_locked(now)
            has_voice = volume >= voice_start_volume

            if not self.recording:
                if not has_voice:
                    self.segment = f"idle | volume {volume:.4f} < {voice_start_volume:.4f}"
                    return
                self.recording = True
                self.segment_interupt_mode = interupt_mode_active
                self.acc = []
                self.acc_len = 0
                self.silence_acc_ms = 0.0
                self.tail_mode = False
                self.tail_acc_ms = 0.0
            elif interupt_mode_active:
                self.segment_interupt_mode = True

            self.acc.append(block)
            self.acc_len += block.size
            segment_sec = self.acc_len / float(self.args.sample_rate)

            if self.tail_mode:
                if has_voice:
                    self.tail_mode = False
                    self.tail_acc_ms = 0.0
                    self.silence_acc_ms = 0.0
                    self.segment = f"recording {segment_sec:.2f}s | tail canceled | volume {volume:.4f}"
                    return

                self.tail_acc_ms += frame_ms
                self.segment = (
                    f"tail {segment_sec:.2f}s | extra {int(self.tail_acc_ms)} / "
                    f"{self.args.cut_tail_ms}ms | volume {volume:.4f}"
                )
                if self.tail_acc_ms >= float(self.args.cut_tail_ms):
                    self._submit_current_segment_locked()
                return

            if has_voice:
                self.silence_acc_ms = 0.0
            else:
                self.silence_acc_ms += frame_ms

            if segment_sec < float(self.args.min_sec):
                self.segment = (
                    f"recording {segment_sec:.2f}s | wait min {self.args.min_sec}s | volume {volume:.4f}"
                )
                return

            need_silence_ms = float(self.args.silence_ms) / max(0.001, segment_sec * float(self.args.cut_rate))
            self.segment = (
                f"recording {segment_sec:.2f}s | silence {int(self.silence_acc_ms)} / "
                f"{int(need_silence_ms)}ms | volume {volume:.4f}"
            )

            if not has_voice and segment_sec >= float(self.args.max_sec):
                self._enter_tail_or_submit_locked(
                    f"tail wait | max {segment_sec:.2f}s | cutTail {self.args.cut_tail_ms}ms"
                )
                return

            if self.silence_acc_ms >= need_silence_ms:
                self._enter_tail_or_submit_locked(
                    f"tail wait | silence {int(self.silence_acc_ms)}ms | cutTail {self.args.cut_tail_ms}ms"
                )

    def _enter_tail_or_submit_locked(self, message: str) -> None:
        if float(self.args.cut_tail_ms) > 0:
            self.tail_mode = True
            self.tail_acc_ms = 0.0
            self.segment = message
            return
        self._submit_current_segment_locked()

    def _active_voice_start_volume_locked(self, now: float) -> float:
        if self._interupt_mode_active_locked(now):
            self.active_voice_start_volume = max(float(self.args.voice_start_volume), self.interupt_volume)
        else:
            self.active_voice_start_volume = float(self.args.voice_start_volume)
        return self.active_voice_start_volume

    def _interupt_mode_active_locked(self, now: float) -> bool:
        if self.tts_playing:
            return not self.tts_early_release_active
        return now < self.interupt_switch_until_ts

    def _interupt_switch_remaining_locked(self, now: float) -> float:
        return max(0.0, self.interupt_switch_until_ts - now)

    def _submit_current_segment_locked(self) -> None:
        if self.acc_len < int(self.args.sample_rate * 0.1):
            self._reset_segment_locked()
            return
        audio = np.concatenate(self.acc).astype(np.float32, copy=False)
        self.segment_queue.put((audio, self.segment_interupt_mode))
        self._reset_segment_locked()

    def _reset_segment_locked(self) -> None:
        self.recording = False
        self.tail_mode = False
        self.acc = []
        self.acc_len = 0
        self.silence_acc_ms = 0.0
        self.tail_acc_ms = 0.0
        self.segment_interupt_mode = False
        self.segment = "idle"

    def _flush_on_stop(self) -> None:
        with self.lock:
            if self.acc_len < int(self.args.sample_rate * 0.1):
                return
            audio = np.concatenate(self.acc).astype(np.float32, copy=False)
            self.segment_queue.put((audio, self.segment_interupt_mode))

    def _sender_loop(self) -> None:
        while True:
            item = self.segment_queue.get()
            if item is None:
                return
            audio, segment_interupt_mode = item
            try:
                with self.lock:
                    self.chunk_count += 1
                    chunk_no = self.chunk_count
                    self.asr_inflight = True
                segment_volume = calc_volume(audio)
                asr_start_ts = time.monotonic()
                language, text = self.asr_client.transcribe_segment(audio)
                asr_duration_ms = int((time.monotonic() - asr_start_ts) * 1000.0)
                line = ""
                lang = ""
                interrupt_candidate = False
                interupt_text = None
                with self.lock:
                    self.asr_inflight = False
                    self.last_asr_duration_ms = asr_duration_ms
                    if text:
                        self._push_log_locked(
                            f"[Qwen ASR] result chunk={chunk_no} {asr_duration_ms}ms "
                            f"interrupt_mode={segment_interupt_mode} volume={segment_volume:.4f} "
                            f"samples={len(audio)} lang={language or '-'} text={text[:120]}"
                        )
                        lang = normalize_language_code(language) or ""
                        output_text = text
                        phonetic_langs = getattr(self.args, "asr_phonetic_output_languages", set())
                        if lang and lang in phonetic_langs:
                            output_text = convert_phonetic(lang, text)

                        prefix = f"{lang}: " if lang else ""
                        line = prefix + output_text
                        interrupt_candidate = self._interupt_mode_active_locked(time.monotonic()) or segment_interupt_mode
                        if interrupt_candidate:
                            self.asr_inflight = False
                            self.status = f"{'recording' if self.running else 'stopped'} {self.args.sample_rate} Hz"
                        else:
                            self.last_asr_text = line
                            self.transcript.append(line)
                            self.transcript = self.transcript[-400:]
                            self.last_asr_update_ts = time.monotonic()
                            self._refresh_long_silence_locked()
                            self._push_log_locked(
                                f"[VAD] ASR accepted chunk={chunk_no}; wait long silence "
                                f"{self.remaining_long_silence_ms}/{self.effective_long_silence_ms}ms"
                            )
                    self.status = f"{'recording' if self.running else 'stopped'} {self.args.sample_rate} Hz"

                if text and interrupt_candidate:
                    if self.interrupt_judge.language_allowed(lang):
                        decision = self.interrupt_judge.analyze_decision(line, lang)
                    else:
                        allowed = ",".join(sorted(self.interrupt_languages)) or "(none)"
                        decision = "continue"
                        self.interrupt_judge.record_result(
                            line,
                            decision,
                            f"language_blocked lang={lang or '-'} allowed={allowed}",
                        )
                    with self.lock:
                        interupt_text = self._consume_interupt_qasr_locked(
                            line,
                            segment_volume,
                            segment_interupt_mode,
                            decision,
                        )
                        self.asr_inflight = False
                        self.status = f"{'recording' if self.running else 'stopped'} {self.args.sample_rate} Hz"
                    if interupt_text is not None:
                        threading.Thread(
                            target=self._start_interupt_turn,
                            args=(interupt_text,),
                            name="tts-interupt-turn",
                            daemon=True,
                        ).start()
                    continue
            except Exception as exc:
                with self.lock:
                    self.asr_inflight = False
                self._push_error(f"send error: {exc!r}")

    def _turn_loop(self) -> None:
        while not self.shutdown_event.is_set():
            time.sleep(0.05)
            with self.lock:
                if not self.transcript or self.last_asr_update_ts is None:
                    continue
                self._refresh_long_silence_locked()
                if self._long_silence_trigger_blocked_locked():
                    continue
                elapsed_ms = int((time.monotonic() - self.last_asr_update_ts) * 1000.0)
                if elapsed_ms >= self.effective_long_silence_ms:
                    self._push_log_locked(
                        f"[VAD] send to LLM elapsed={elapsed_ms}ms "
                        f"threshold={self.effective_long_silence_ms}ms chars={len(format_mixed_language_turn(self.transcript))}"
                    )
                    self._trigger_turn_from_long_silence_locked()

    def _queue_speech_turn(
        self,
        user_text: str,
        reply_index: int,
        source: str,
        **kwargs: Any,
    ) -> None:
        self.speech_turn_queue.put(
            {
                "user_text": user_text,
                "reply_index": reply_index,
                "source": source,
                "kwargs": kwargs,
                "reset_generation": self.memory.reset_generation,
            }
        )

    def _drain_pending_speech_turns(self, skipped_message: str = "[Skipped by interrupt]") -> int:
        dropped = 0
        while True:
            try:
                turn = self.speech_turn_queue.get_nowait()
            except queue.Empty:
                return dropped
            if turn is None:
                self.speech_turn_queue.put(None)
                return dropped
            if isinstance(turn, dict):
                reply_index = turn.get("reply_index")
                if isinstance(reply_index, int) and 0 <= reply_index < len(self.reply_results):
                    if not str(self.reply_results[reply_index].get("text", "") or "").strip():
                        self.reply_results[reply_index]["text"] = skipped_message
            dropped += 1

    def cancel_speech_turns(self, reason: str = "cancelled") -> int:
        with self.lock:
            if self.current_llm_cancel_event is not None:
                self.current_llm_cancel_event.set()
            if self.tts_cancel_event is not None:
                self.tts_cancel_event.set()
            dropped_turns = self._drain_pending_speech_turns(f"[Skipped by {reason}]")
            self._push_log_locked(
                f"[SpeechTurn] cancel reason={reason} dropped_queued_turns={dropped_turns}"
            )
            return dropped_turns

    def _speech_turn_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                turn = self.speech_turn_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if turn is None:
                break
            user_text = str(turn.get("user_text", "") or "")
            reply_index = int(turn.get("reply_index", -1))
            source = str(turn.get("source", "SpeechTurn") or "SpeechTurn")
            turn_generation = turn.get("reset_generation")
            kwargs = turn.get("kwargs")
            if not isinstance(kwargs, dict):
                kwargs = {}
            with self.lock:
                if not self.memory.is_current_generation(turn_generation if isinstance(turn_generation, int) else None):
                    if 0 <= reply_index < len(self.reply_results):
                        if not str(self.reply_results[reply_index].get("text", "") or "").strip():
                            self.reply_results[reply_index]["text"] = "[Skipped by memory reset]"
                    self._push_log_locked(f"[{source}] skipped stale LLM turn generation={turn_generation}->{self.memory.reset_generation}")
                    continue
                self.speech_turn_active = True
                self._push_log_locked(
                    f"[{source}] dequeue LLM turn chars={len(user_text)} "
                    f"remaining={self.speech_turn_queue.qsize()}"
                )
            try:
                self._llm_stream_reply(user_text, reply_index, **kwargs)
            except Exception as exc:
                self._push_error(f"speech turn error: {exc!r}")
            finally:
                with self.lock:
                    self.speech_turn_active = False

    def _refresh_long_silence_locked(self) -> None:
        current_asr_text = build_stutter_text(self.transcript)
        self.long_silence_ms = int(float(self.args.long_silence_ms))
        (
            self.effective_long_silence_ms,
            self.stuttering_count,
            self.char_stutter_count,
            self.token_stutter_count,
        ) = calc_effective_long_silence_ms(
            base_long_silence_ms=self.long_silence_ms,
            current_asr_text=current_asr_text,
            stutter_enabled=bool(self.args.stutter_extend_enabled),
            stutter_delay_ms=int(self.args.stutter_delay_ms),
            stutter_delay_max_ms=int(self.args.stutter_delay_max_ms),
        )

        elapsed_ms = 0
        if self.last_asr_update_ts is not None:
            elapsed_ms = int((time.monotonic() - self.last_asr_update_ts) * 1000.0)
        self.remaining_long_silence_ms = max(0, self.effective_long_silence_ms - elapsed_ms)
        hold_reason = self._long_silence_hold_reason_locked()
        count_line = (
            f"StutterCounts: {self.stuttering_count} | char={self.char_stutter_count} | "
            f"token={self.token_stutter_count}"
        )
        countdown_line = (
            f"long: {self.remaining_long_silence_ms}/{self.effective_long_silence_ms}ms | "
            f"base: {self.long_silence_ms}ms | asr: {int(float(self.args.turn_detection_silence_ms))}ms | "
            f"state={hold_reason}"
        )
        self.silence_counters = [count_line, countdown_line, "----------", current_asr_text]

    def _long_silence_hold_reason_locked(self) -> str:
        if self.recording or self.tail_mode:
            return "audio"
        if self.asr_inflight:
            return "asr"
        if not self.segment_queue.empty():
            return "queue"
        return "ready"

    def _long_silence_trigger_blocked_locked(self) -> bool:
        return self._long_silence_hold_reason_locked() != "ready"

    def _trigger_turn_from_long_silence_locked(self) -> None:
        triggered_asr_text = format_mixed_language_turn(self.transcript).strip()
        if not triggered_asr_text:
            return

        self.reply_results.append({"role": "user", "text": triggered_asr_text})
        self.reply_results.append({"role": "assistant", "text": ""})
        self.reply_results = self.reply_results[-400:]
        reply_index = len(self.reply_results) - 1
        self.transcript = []
        self.last_asr_text = ""
        self.last_asr_update_ts = None
        self.effective_long_silence_ms = self.long_silence_ms
        self.remaining_long_silence_ms = self.long_silence_ms
        self.stuttering_count = 0
        self.char_stutter_count = 0
        self.token_stutter_count = 0
        self._queue_speech_turn(triggered_asr_text, reply_index, "ASR")
        self._push_log_locked(
            f"[ASR] queued LLM turn chars={len(triggered_asr_text)} "
            f"queued={self.speech_turn_queue.qsize()}"
        )

    def _consume_interupt_qasr_locked(
        self,
        qasr_text: str,
        segment_volume: float,
        segment_interupt_mode: bool,
        interrupt_decision: str,
    ) -> str | None:
        if not (segment_interupt_mode or self._interupt_mode_active_locked(time.monotonic())):
            return None

        self.interupt_last_volume = segment_volume
        self.interupt_qasr_text = qasr_text.strip()
        if interrupt_decision != "interrupt":
            self._push_log_locked(
                f"[InterruptAnalyze] continue volume={segment_volume:.4f}: {qasr_text[:120]}"
            )
            return None

        self._push_log_locked(
            f"[InterruptAnalyze] interrupt volume={segment_volume:.4f}: {qasr_text[:120]}"
        )

        self.llm_pending_stop_at = self.current_tts_chunk.strip()
        if self.llm_pending_stop_at:
            active_reply = self._active_llm_reply_text_locked()
            if active_reply and self.current_llm_user_text:
                self.memory.upsert_llm_history(
                    self.current_llm_user_text,
                    active_reply,
                    f"stop_at: {self.llm_pending_stop_at}",
                )
            elif self.current_llm_user_text:
                self.memory.update_last_state_for_user(
                    self.current_llm_user_text,
                    f"stop_at: {self.llm_pending_stop_at}",
                )
        if self.current_llm_cancel_event is not None:
            self.current_llm_cancel_event.set()
        if self.tts_cancel_event is not None:
            self.tts_cancel_event.set()
        dropped_turns = self._drain_pending_speech_turns()
        self._push_log_locked(
            f"[Interrupt] trigger stop requested: {self.interupt_qasr_text[:120]} "
            f"stop_at={self.llm_pending_stop_at[:120]} dropped_queued_turns={dropped_turns}"
        )

        interupt_text = self.interupt_qasr_text.strip()
        self.interupt_count = self.interupt_threshold
        self.interupt_qasr_parts = []
        self.transcript = []
        self.last_asr_text = interupt_text
        self.last_asr_update_ts = None
        return interupt_text

    def _active_llm_reply_text_locked(self) -> str:
        reply_index = self.current_llm_reply_index
        if reply_index is None or reply_index < 0 or reply_index >= len(self.reply_results):
            return ""
        return str(self.reply_results[reply_index].get("text", "") or "").strip()

    def _start_interupt_turn(self, interupt_text: str) -> None:
        sd.stop()
        with self.lock:
            self._push_log_locked(f"[Interrupt] queue LLM turn chars={len(interupt_text)}")
            self.tts_playing = False
            self.reply_results.append({"role": "user", "text": interupt_text})
            self.reply_results.append({"role": "assistant", "text": ""})
            self.reply_results = self.reply_results[-400:]
            reply_index = len(self.reply_results) - 1
            self._queue_speech_turn(interupt_text, reply_index, "Interrupt", enable_router=False)
            self._push_log_locked(
                f"[Interrupt] queued LLM turn chars={len(interupt_text)} "
                f"queued={self.speech_turn_queue.qsize()}"
            )

    def start_manual_llm_turn(self, prompt: str, llm_temperature: float | None = None) -> None:
        user_text = str(prompt or "").strip()
        if not user_text:
            raise ValueError("empty prompt")
        with self.lock:
            self._push_log_locked(f"[ManualLLM] start chars={len(user_text)}")
            self.reply_results.append({"role": "user", "text": user_text})
            self.reply_results.append({"role": "assistant", "text": ""})
            self.reply_results = self.reply_results[-400:]
            reply_index = len(self.reply_results) - 1
        threading.Thread(
            target=self._llm_stream_reply,
            args=(user_text, reply_index),
            kwargs={"enable_tts": False, "llm_temperature": llm_temperature},
            name="llm-manual-stream",
            daemon=True,
        ).start()

    def start_asr_text_turn(
        self,
        text: str,
        source: str = "ASR",
        reader_context_override: str = "",
        llm_temperature: float | None = None,
        require_reader_mode: bool = False,
    ) -> None:
        user_text = str(text or "").strip()
        if not user_text:
            raise ValueError("empty ASR text")
        with self.lock:
            if require_reader_mode and not self.reader_mode:
                raise ValueError("reader mode is not active")
            self._push_log_locked(f"[{source}] queue LLM turn chars={len(user_text)}")
            self.reply_results.append({"role": "user", "text": user_text})
            self.reply_results.append({"role": "assistant", "text": ""})
            self.reply_results = self.reply_results[-400:]
            reply_index = len(self.reply_results) - 1
            self.last_asr_text = user_text
            self.last_asr_update_ts = None
            self.transcript = []
            self.effective_long_silence_ms = self.long_silence_ms
            self.remaining_long_silence_ms = self.long_silence_ms
            self.stuttering_count = 0
            self.char_stutter_count = 0
            self.token_stutter_count = 0
            self._queue_speech_turn(
                user_text,
                reply_index,
                source,
                reader_context_override=str(reader_context_override or ""),
                reader_context_override_source=source,
                llm_temperature=llm_temperature,
            )
            self._push_log_locked(
                f"[{source}] queued LLM turn chars={len(user_text)} "
                f"queued={self.speech_turn_queue.qsize()}"
            )

    def _start_section_id_turn(self) -> None:
        prompt = "你好，你的 session ID 是多少？"
        with self.lock:
            self._push_log_locked(f"[SectionID] ask via LLM turn: {prompt}")
            self.reply_results.append({"role": "user", "text": prompt})
            self.reply_results.append({"role": "assistant", "text": ""})
            self.reply_results = self.reply_results[-400:]
            reply_index = len(self.reply_results) - 1
        self._llm_stream_reply(
            prompt,
            reply_index,
            use_control_prompt=False,
            enable_tts=False,
            section_id_turn=True,
        )

    def _llm_stream_reply(
        self,
        user_text: str,
        reply_index: int,
        use_control_prompt: bool = True,
        enable_tts: bool | None = None,
        section_id_turn: bool = False,
        enable_router: bool = True,
        reader_context_override: str = "",
        reader_context_override_source: str = "",
        llm_temperature: float | None = None,
    ) -> None:
        response_parts: List[str] = []
        tts_queue: "queue.Queue[str | None]" = queue.Queue(maxsize=max(1, int(self.args.tts_queue_ahead)))
        cancel_event = threading.Event()
        tts_enabled = bool(self.args.tts_enabled) if enable_tts is None else bool(enable_tts)
        track_current_turn = tts_enabled
        tts_thread: threading.Thread | None = None
        if tts_enabled:
            tts_thread = self.tts_pipeline.start_worker(tts_queue, cancel_event)
        sentence_buffer = ""
        sentence_group: List[str] = []
        reset_generation = 0
        try:
            with self.lock:
                reset_generation = self.memory.reset_generation
                self.llm_inflight = True
                if track_current_turn:
                    self.current_llm_cancel_event = cancel_event
                    self.current_llm_user_text = user_text
                    self.current_llm_reply_index = reply_index
                reader_mode = bool(self.reader_mode)
                reader_context = str(self.reader_analyze_resault or "")
                reader_summary_index = str(self.reader_summary_index or "")
                prompt_result = self.talker_prompt_switch.build(
                    user_text,
                    use_control_prompt=use_control_prompt,
                    enable_router=enable_router,
                    section_id_turn=section_id_turn,
                    reader_mode=reader_mode,
                    reader_context=reader_context,
                    reader_summary_index=reader_summary_index,
                    reader_context_override=reader_context_override,
                    reader_context_override_source=reader_context_override_source,
                )
            llm_user_text = prompt_result.prompt
            history = prompt_result.history
            resolved_temperature = self.llm_client.resolve_temperature(llm_temperature)
            with self.lock:
                self.talker_prompt_mode = prompt_result.mode
                self.last_llm_temperature = resolved_temperature
                self._push_log_locked(
                    f"[LLM] send source={reader_context_override_source or self.talker_prompt_mode} "
                    f"mode={self.talker_prompt_mode} temp={resolved_temperature:g} chars={len(llm_user_text)}"
                )
                self.memory.add_control_prompt_tokens(estimate_token_count(llm_user_text))
            for delta in self.llm_client.stream_reply(llm_user_text, history=history, temperature=llm_temperature):
                if cancel_event.is_set():
                    break
                if not delta:
                    continue
                response_parts.append(delta)
                with self.lock:
                    self.memory.add_control_prompt_tokens(estimate_token_count(delta))
                    if reply_index < len(self.reply_results):
                        self.reply_results[reply_index]["text"] += delta
                if tts_enabled:
                    sentence_buffer, sentence_group = self.tts_pipeline.queue_sentences(
                        tts_queue,
                        sentence_buffer + delta,
                        sentence_group,
                    )
            if tts_enabled and not cancel_event.is_set():
                self.tts_pipeline.flush_tail(tts_queue, sentence_buffer, sentence_group)
            response_text = "".join(response_parts).strip()
            should_route = False
            with self.lock:
                interrupted = cancel_event.is_set()
                stop_at = ""
                if interrupted:
                    stop_at = (self.llm_pending_stop_at or self.current_tts_chunk).strip()
                    self.llm_pending_stop_at = ""
                if not self.memory.is_current_generation(reset_generation):
                    response_text = ""
                if response_text:
                    self.memory.upsert_llm_history(
                        user_text,
                        response_text,
                        f"stop_at: {stop_at}" if stop_at else "finish_round",
                    )
                    if section_id_turn:
                        self._push_log_locked(f"[SectionID] reply via LLM turn: {response_text[:240]}")
                    should_route = bool(prompt_result.router_allowed and not interrupted)
                self.llm_inflight = False
                if section_id_turn:
                    self.report_section_id_inflight = False
                    self.report_section_id_done = True
            if should_route:
                threading.Thread(
                    target=self._run_router_after_turn,
                    args=(user_text, response_text, reset_generation),
                    name="router-after-turn",
                    daemon=True,
                ).start()
        except Exception as exc:
            with self.lock:
                history_still_current = self.memory.is_current_generation(reset_generation)
                if history_still_current and reply_index < len(self.reply_results):
                    self.reply_results[reply_index]["text"] += f"[LLM error: {exc!r}]"
                self.llm_inflight = False
                if history_still_current:
                    self.errors.append(f"llm error: {exc!r}")
                    self.errors = self.errors[-100:]
                if section_id_turn:
                    if history_still_current:
                        self._push_log_locked(f"[SectionID] error via LLM turn: {exc!r}")
                    self.report_section_id_inflight = False
                    self.report_section_id_done = True
        finally:
            if tts_enabled:
                if cancel_event.is_set():
                    self.tts_pipeline.drain_queue(tts_queue)
                tts_queue.put(None)
                if tts_thread is not None:
                    tts_thread.join()
            with self.lock:
                if track_current_turn and self.current_llm_cancel_event is cancel_event:
                    self.current_llm_cancel_event = None
                    self.current_llm_user_text = ""
                    self.current_llm_reply_index = None

    def _read_router_prompt(self) -> str:
        try:
            if self.control_prompt_config_path.exists():
                loaded = yaml.safe_load(self.control_prompt_config_path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    section = loaded.get("RouterConfig")
                    if isinstance(section, dict):
                        raw = section.get("Raw")
                        if isinstance(raw, str) and raw.strip():
                            return raw.strip()
                        prompt = section.get("Router Prompt", section.get("Prompt", ""))
                        if isinstance(prompt, str) and prompt.strip():
                            return prompt.strip()
        except Exception as exc:
            self._push_error(f"control prompt router read error: {exc!r}")
            return ""
        for path in self.router_config_paths:
            try:
                if path.exists():
                    return path.read_text(encoding="utf-8-sig", errors="replace").strip()
            except Exception as exc:
                self._push_error(f"router config read error: {exc!r}")
                return ""
        return ""

    def _build_router_user_text(self, router_prompt: str, user_text: str, assistant_text: str) -> str:
        user = strip_control_codes(user_text)
        assistant = strip_control_codes(assistant_text)
        turn_record = f"User:\n{user}\n\nTalker:\n{assistant}".strip()
        replacements = {
            "{conversation_history}": turn_record,
            "{current_user_message}": user,
            "{current_assistant_message}": assistant,
            "{talker_reply}": assistant,
        }
        prompt = router_prompt
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        if prompt == router_prompt:
            prompt = (
                f"{router_prompt}\n\n"
                f"Conversation record:\n{turn_record}\n\n"
                "Router decision:"
            )
        return prompt

    def _run_router_after_turn(
        self,
        user_text: str,
        assistant_text: str,
        reset_generation: int | None = None,
    ) -> None:
        with self.lock:
            if not self.memory.is_current_generation(reset_generation):
                return
        router_prompt = self._read_router_prompt()
        if not router_prompt:
            return
        router_user_text = self._build_router_user_text(router_prompt, user_text, assistant_text)
        reply = ""
        codes: list[str] = []
        try:
            with self.lock:
                self.last_llm_temperature = 0.0
                self._push_log_locked(f"[LLM] send source=Router mode=control temp=0 chars={len(router_user_text)}")
            reply = "".join(self.llm_client.stream_reply(router_user_text, history=[], temperature=0.0)).strip()
        except Exception as exc:
            with self.lock:
                if not self.memory.is_current_generation(reset_generation):
                    return
                self.memory.append_router_record(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "user": strip_control_codes(user_text),
                        "talker": strip_control_codes(assistant_text),
                        "prompt": router_user_text,
                        "reply": "",
                        "codes": [],
                        "error": repr(exc),
                    }
                )
            self._push_error(f"router error: {exc!r}")
            return
        for code in CONTROL_CODE_TEXT.findall(reply):
            if code not in codes:
                codes.append(code)
        with self.lock:
            if not self.memory.is_current_generation(reset_generation):
                return
            self.memory.append_router_record(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "user": strip_control_codes(user_text),
                    "talker": strip_control_codes(assistant_text),
                    "prompt": router_user_text,
                    "reply": reply,
                    "codes": codes,
                }
            )
        if not codes:
            return
        route_text = "\n".join(codes)
        with self.lock:
            self.reply_results.append({"role": "router", "text": route_text})
            self.reply_results = self.reply_results[-400:]
            self._push_log_locked(f"[Router] {route_text.replace(chr(10), ' ')}")

    def _push_log(self, message: str) -> None:
        with self.lock:
            self._push_log_locked(message)

    def _push_log_locked(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.tts_logs.append(f"{timestamp} {message}")
        self.tts_logs = self.tts_logs[-200:]

    def _push_error(self, message: str) -> None:
        with self.lock:
            self.errors.append(message)
            self.errors = self.errors[-100:]
            if not self.running:
                self.status = "error"
