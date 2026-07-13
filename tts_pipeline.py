from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, List

import emoji
import numpy as np
import sounddevice as sd

from audio_utils import wav_result_to_audio

if TYPE_CHECKING:
    from mic_engine import MicEngine


TTS_CHOP_DELIMITERS = ".!?,;:\u3002\uFF01\uFF1F\uFF0C\u3001\uFF1B\uFF1A"
MARKDOWN_BOLD_TEXT = re.compile(r"\*\*(.*?)\*\*")
PARENTHETICAL_TEXT = re.compile(r"\([^()]*\)|\uFF08[^\uFF08\uFF09]*\uFF09")
UNFINISHED_PARENTHETICAL_TEXT = re.compile(r"\([^()]*$|\uFF08[^\uFF08\uFF09]*$")
CONTROL_CODE_TEXT = re.compile(r"%%[^%\r\n]{1,80}%%")
SPEAKABLE_TEXT = re.compile(r"[\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", re.UNICODE)


def build_tts_clause_chunks(chop_exceptions: set[str]) -> re.Pattern[str]:
    delimiters = "".join(char for char in TTS_CHOP_DELIMITERS if char not in chop_exceptions)
    if not delimiters:
        return re.compile(r"a^")
    escaped = re.escape(delimiters)
    return re.compile(rf"[^{escaped}]+[{escaped}]+")


def sanitize_tts_text(text: str, silence_types: set[str]) -> str:
    result = CONTROL_CODE_TEXT.sub("", str(text or "")).strip()
    result = emoji.replace_emoji(result, replace="")
    if "markdown_bold" in silence_types or "bold" in silence_types:
        result = MARKDOWN_BOLD_TEXT.sub(r"\1", result)
        result = result.replace("**", "")
    if "parentheses" in silence_types or "parenthesis" in silence_types:
        previous = None
        while previous != result:
            previous = result
            result = PARENTHETICAL_TEXT.sub("", result)
        result = UNFINISHED_PARENTHETICAL_TEXT.sub("", result)
    return re.sub(r"\s+", " ", result).strip()


def has_unclosed_parenthetical(text: str) -> bool:
    ascii_depth = 0
    wide_depth = 0
    for char in text:
        if char == "(":
            ascii_depth += 1
        elif char == ")" and ascii_depth > 0:
            ascii_depth -= 1
        elif char == "\uFF08":
            wide_depth += 1
        elif char == "\uFF09" and wide_depth > 0:
            wide_depth -= 1
    return ascii_depth > 0 or wide_depth > 0


class TtsPipeline:
    def __init__(self, engine: MicEngine):
        self.engine = engine

    @property
    def args(self):
        return self.engine.args

    def start_worker(
        self,
        tts_queue: "queue.Queue[str | None]",
        cancel_event: threading.Event,
    ) -> threading.Thread:
        thread = threading.Thread(
            target=self.worker,
            args=(tts_queue, cancel_event),
            name="omnivoice-worker",
            daemon=True,
        )
        thread.start()
        return thread

    def queue_sentences(
        self,
        tts_queue: "queue.Queue[str | None]",
        text: str,
        sentence_group: List[str],
    ) -> tuple[str, List[str]]:
        consumed_until = 0
        clause_chunks = build_tts_clause_chunks(getattr(self.args, "tts_chop_exceptions", set()))
        for match in clause_chunks.finditer(text):
            fragment = match.group(0).strip()
            if not fragment:
                consumed_until = match.end()
                continue
            sentence_group.append(fragment)
            consumed_until = match.end()
            group_text = " ".join(sentence_group)
            if (
                len(sentence_group) >= max(1, int(self.args.tts_group_sentences))
                and not has_unclosed_parenthetical(group_text)
            ):
                self.put_text(tts_queue, group_text)
                sentence_group = []
        return text[consumed_until:], sentence_group

    def flush_tail(
        self,
        tts_queue: "queue.Queue[str | None]",
        sentence_buffer: str,
        sentence_group: List[str],
    ) -> None:
        tail = sentence_buffer.strip()
        if tail:
            sentence_group.append(tail)
        if sentence_group:
            self.put_text(tts_queue, " ".join(sentence_group))

    def put_text(self, tts_queue: "queue.Queue[str | None]", text: str) -> None:
        tts_text = sanitize_tts_text(text, getattr(self.args, "tts_silence_types", set()))
        if tts_text and SPEAKABLE_TEXT.search(tts_text):
            tts_queue.put(tts_text)

    def worker(self, tts_queue: "queue.Queue[str | None]", cancel_event: threading.Event) -> None:
        max_workers = max(1, int(self.args.tts_queue_ahead))
        batch_no = 0
        next_play_no = 1
        closed = False
        last_wait_log_ts = 0.0
        futures: dict[int, Future[tuple[int, str, np.ndarray, int, int, float]]] = {}
        engine = self.engine
        with engine.lock:
            engine.tts_cancel_event = cancel_event

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="omnivoice-synth") as executor:
            while (not closed or futures) and not cancel_event.is_set():
                while not closed and len(futures) < max_workers and not cancel_event.is_set():
                    try:
                        text = tts_queue.get(timeout=0.05)
                    except queue.Empty:
                        break
                    if text is None:
                        closed = True
                        break
                    batch_no += 1
                    future = executor.submit(self.synth_batch, batch_no, text)
                    futures[batch_no] = future
                    self.push_tts_log(
                        f"#{batch_no} queued synth | converting={len(futures)} | "
                        f"queued={tts_queue.qsize()} | group={self.args.tts_group_sentences} | "
                        f"chars={len(text)}: {text[:80]}"
                    )

                next_future = futures.get(next_play_no)
                if next_future is None:
                    time.sleep(0.02)
                    continue
                if not next_future.done():
                    now = time.monotonic()
                    if now - last_wait_log_ts >= 0.5:
                        last_wait_log_ts = now
                        self.push_tts_log(
                            f"waiting play #{next_play_no} | converting={len(futures)} | queued={tts_queue.qsize()}"
                        )
                    time.sleep(0.05)
                    continue

                futures.pop(next_play_no, None)
                try:
                    seq_no, text, samples, sample_rate, api_ms, audio_sec = next_future.result()
                    self.push_tts_log(
                        f"#{seq_no} ready api={api_ms}ms audio={audio_sec:.2f}s sr={sample_rate} | "
                        f"converting={len(futures)} | queued={tts_queue.qsize()}"
                    )
                    if cancel_event.is_set():
                        break
                    play_started = time.monotonic()
                    with engine.lock:
                        engine.tts_playing = True
                        engine.prev_tts_chunk = engine.current_tts_chunk
                        engine.current_tts_chunk = text
                        engine.tts_early_release_active = False
                        engine.interupt_switch_until_ts = 0.0
                        engine.active_voice_start_volume = max(float(self.args.voice_start_volume), engine.interupt_volume)
                        engine.interupt_count = engine.interupt_threshold
                        engine.interupt_qasr_parts = []
                        engine.interupt_qasr_text = ""
                    engine._push_log(
                        f"[TTS] play start #{seq_no} interrupt_count={engine.interupt_count}/{engine.interupt_threshold} "
                        f"interrupt_volume={engine.interupt_volume:.4f}: {text[:80]}"
                    )
                    interrupted = self.play_audio_interruptible(samples, sample_rate, cancel_event)
                    play_ms = int((time.monotonic() - play_started) * 1000.0)
                    switch_delay = 0.0
                    with engine.lock:
                        engine.tts_playing = False
                        engine.tts_early_release_active = False
                        if not interrupted:
                            engine.interupt_switch_until_ts = time.monotonic() + engine.interupt_switch_delay
                            engine.active_voice_start_volume = max(float(self.args.voice_start_volume), engine.interupt_volume)
                            engine.interupt_count = engine.interupt_threshold
                            engine.interupt_qasr_parts = []
                            engine.interupt_qasr_text = ""
                            switch_delay = engine.interupt_switch_delay
                    if interrupted:
                        engine._push_log(f"[TTS] interrupted #{seq_no} played_ms={play_ms}: {text[:80]}")
                        break
                    engine._push_log(
                        f"[TTS] played #{seq_no} played_ms={play_ms} "
                        f"switch_delay={switch_delay:.2f}s: {text[:80]}"
                    )
                except Exception as exc:
                    self.push_tts_log(f"#{next_play_no} error: {exc!r}")
                next_play_no += 1
        with engine.lock:
            if engine.tts_cancel_event is cancel_event:
                engine.tts_cancel_event = None
                engine.tts_playing = False
                engine.tts_early_release_active = False

    def synth_batch(self, seq_no: int, text: str) -> tuple[int, str, np.ndarray, int, int, float]:
        lock = getattr(self.args, "tts_settings_lock", None)
        if lock is not None:
            with lock:
                packet_prefix = self.args.tts_packet_prefix
                packet_suffix = self.args.tts_packet_suffix
        else:
            packet_prefix = self.args.tts_packet_prefix
            packet_suffix = self.args.tts_packet_suffix
        api_text = f"{packet_prefix}{text}{packet_suffix}"
        started = time.monotonic()
        result = self.engine.tts_client.call_tts_api(api_text)
        api_ms = int((time.monotonic() - started) * 1000.0)
        samples, sample_rate = wav_result_to_audio(result)
        audio_sec = len(samples) / float(sample_rate)
        return seq_no, api_text, samples, sample_rate, api_ms, audio_sec

    def play_audio_interruptible(
        self,
        samples: np.ndarray,
        sample_rate: int,
        cancel_event: threading.Event,
    ) -> bool:
        chunk_frames = max(1, int(sample_rate * 0.02))
        offset = 0
        interrupted = False
        stream: sd.OutputStream | None = None
        engine = self.engine
        try:
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            stream.start()
            while offset < len(samples):
                if cancel_event.is_set():
                    interrupted = True
                    break
                remaining_sec = (len(samples) - offset) / float(sample_rate)
                if (
                    engine.interupt_early_release > 0
                    and remaining_sec <= engine.interupt_early_release
                    and not engine.tts_early_release_active
                ):
                    with engine.lock:
                        engine.tts_early_release_active = True
                        engine.active_voice_start_volume = float(self.args.voice_start_volume)
                    self.push_tts_log(f"early release active | remaining={remaining_sec:.2f}s")
                chunk = samples[offset : offset + chunk_frames]
                stream.write(chunk.reshape(-1, 1))
                offset += len(chunk)
        finally:
            if stream is not None:
                try:
                    if interrupted or cancel_event.is_set():
                        interrupted = True
                        stream.abort()
                    else:
                        stream.stop()
                finally:
                    stream.close()
        return interrupted or cancel_event.is_set()

    def drain_queue(self, tts_queue: "queue.Queue[str | None]") -> None:
        while True:
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                return

    def push_tts_log(self, message: str) -> None:
        self.engine._push_log(f"[TTS] {message}")
