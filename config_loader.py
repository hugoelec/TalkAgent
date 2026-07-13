from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "asr_base_url": "http://127.0.0.1:9000/v1",
    "model": "Qwen/Qwen3-ASR-1.7B",
    "method": "chat",
    "ui_host": "0.0.0.0",
    "ui_port": 7860,
    "min_sec": 2.0,
    "max_sec": 6.0,
    "silence_ms": 1000.0,
    "cut_rate": 2.0,
    "cut_tail_ms": 5.0,
    "voice_start_volume": 0.010,
    "sample_rate": 16000,
    "block_ms": 20,
    "max_tokens": 128,
    "timeout": 120.0,
    "long_silence_ms": 2000,
    "stutter_extend_enabled": True,
    "stutter_extend_mode": "hybrid_simple",
    "stutter_delay_ms": 80,
    "stutter_delay_max_ms": 600,
    "asr_phonetic_output": {},
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if re.match(r"^[+-]?\d+$", value):
            return int(value)
        if re.match(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$", value):
            return float(value)
    except Exception:
        pass
    return value


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not path.exists():
        return data

    raw_text = path.read_text(encoding="utf-8")
    clean_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", raw_text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(clean_text)
        return dict(loaded or {})
    except ImportError:
        pass

    raw_lines = clean_text.splitlines()
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, data)]
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    if isinstance(data.get("asr_phonetic_output"), dict) and not data["asr_phonetic_output"]:
        list_items = parse_simple_list_section(raw_lines, "asr_phonetic_output")
        if list_items:
            data["asr_phonetic_output"] = list_items
    tts_cfg = data.get("tts")
    if isinstance(tts_cfg, dict) and isinstance(tts_cfg.get("silence_types"), dict) and not tts_cfg["silence_types"]:
        list_items = parse_simple_nested_list_section(raw_lines, "tts", "silence_types")
        if list_items:
            tts_cfg["silence_types"] = list_items
    if isinstance(tts_cfg, dict) and isinstance(tts_cfg.get("chop_exceptions"), dict) and not tts_cfg["chop_exceptions"]:
        list_items = parse_simple_nested_list_section(raw_lines, "tts", "chop_exceptions")
        if list_items:
            tts_cfg["chop_exceptions"] = list_items
    return data


def parse_simple_list_section(lines: list[str], section_name: str) -> list[str]:
    section_indent: int | None = None
    items: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = stripped.split("#", 1)[0].rstrip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if section_indent is None:
            if header == f"{section_name}:":
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if stripped.startswith("- "):
            item = stripped[2:].split("#", 1)[0].strip()
            if item:
                items.append(str(parse_scalar(item)))
    return items


def parse_simple_nested_list_section(lines: list[str], parent_name: str, section_name: str) -> list[str]:
    parent_indent: int | None = None
    section_indent: int | None = None
    items: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = stripped.split("#", 1)[0].rstrip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if parent_indent is None:
            if header == f"{parent_name}:":
                parent_indent = indent
            continue
        if indent <= parent_indent:
            if section_indent is not None:
                break
            parent_indent = None
            continue
        if section_indent is None:
            if header == f"{section_name}:":
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if stripped.startswith("- "):
            item = stripped[2:].split("#", 1)[0].strip()
            if item:
                items.append(str(parse_scalar(item)))
    return items


def nested_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def config_value(config: Dict[str, Any], flat_key: str, nested_path: str | None = None) -> Any:
    if nested_path:
        nested_value = nested_get(config, nested_path, None)
        if nested_value is not None:
            return nested_value
    return config[flat_key]


def section_value(config: Dict[str, Any], section: str, key: str, flat_key: str, default: Any = None) -> Any:
    value = nested_get(config, f"{section}.{key}", None)
    if value is not None:
        return value
    return config.get(flat_key, default)


def merge_prompt_config(config: Dict[str, Any], prompt_config: Dict[str, Any]) -> None:
    for section_name in ("Control Prompt", "control_prompt", "EchoFilter"):
        section = prompt_config.get(section_name)
        if isinstance(section, dict):
            config[section_name] = section


def asr_base_url_from_config(config: Dict[str, Any]) -> str:
    nested_base_url = nested_get(config, "asr.base_url", None)
    nested_endpoint = nested_get(config, "asr.endpoint", None)
    if nested_base_url and nested_endpoint:
        endpoint = str(nested_endpoint).strip()
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint[: -len("/chat/completions")]
        elif endpoint.endswith("/audio/transcriptions"):
            endpoint = endpoint[: -len("/audio/transcriptions")]
        return f"{str(nested_base_url).rstrip('/')}/{endpoint.lstrip('/')}".rstrip("/")
    return str(config["asr_base_url"])


def parse_phonetic_output_languages(value: Any) -> set[str]:
    if not value:
        return set()

    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, list):
        items = value
    else:
        return set()

    result = set()
    for item in items:
        lang = str(item).strip().lower().rstrip(":")
        if lang:
            result.add(lang)
    return result


def parse_language_codes(value: Any) -> set[str]:
    if not value:
        return set()
    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.split(",")
    else:
        return set()

    result = set()
    for item in items:
        lang = str(item).strip().lower().rstrip(":")
        if lang:
            result.add(lang)
    return result


def parse_silence_types(value: Any) -> set[str]:
    if not value:
        return set()

    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.split(",")
    else:
        return set()

    result = set()
    for item in items:
        silence_type = str(item).strip().lower().replace("-", "_").rstrip(":")
        if silence_type:
            result.add(silence_type)
    return result


def parse_chop_exceptions(value: Any) -> set[str]:
    if not value:
        return set()

    if isinstance(value, dict):
        items = value.keys()
    elif isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value]
    else:
        return set()

    result = set()
    for item in items:
        text = str(item).strip().rstrip(":")
        if text:
            result.update(text)
    return result


def build_args() -> argparse.Namespace:
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default=None)
    known, _remaining = first.parse_known_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(known.config) if known.config else script_dir / "config.yaml"
    config = dict(DEFAULTS)
    config.update(load_simple_yaml(config_path))
    merge_prompt_config(config, load_simple_yaml(config_path.parent / "ControlPrompt.yaml"))

    turn_detection_silence_ms = config_value(config, "silence_ms", "turn_detection.silence_ms")
    long_silence_ms = config_value(config, "long_silence_ms", "turn_detection.long_silence_ms")
    stutter_extend_enabled = config_value(
        config,
        "stutter_extend_enabled",
        "turn_detection.stutter_extend.enabled",
    )
    stutter_extend_mode = config_value(
        config,
        "stutter_extend_mode",
        "turn_detection.stutter_extend.mode",
    )
    stutter_delay_ms = config_value(
        config,
        "stutter_delay_ms",
        "turn_detection.stutter_extend.stutter_delay_ms",
    )
    stutter_delay_max_ms = config_value(
        config,
        "stutter_delay_max_ms",
        "turn_detection.stutter_extend.stutter_delay_max_ms",
    )
    asr_phonetic_output_languages = parse_phonetic_output_languages(config.get("asr_phonetic_output"))
    llm_cfg = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    control_prompt_cfg = config.get("Control Prompt")
    if not isinstance(control_prompt_cfg, dict):
        control_prompt_cfg = config.get("control_prompt") if isinstance(config.get("control_prompt"), dict) else {}
    tts_cfg = config.get("tts") if isinstance(config.get("tts"), dict) else {}
    echo_filter_cfg = config.get("EchoFilter") if isinstance(config.get("EchoFilter"), dict) else {}
    persona_prompt = str(
        control_prompt_cfg.get(
            "persona",
            control_prompt_cfg.get("text", llm_cfg.get("control_prompt", llm_cfg.get("append_prompt", ""))),
        )
    )
    tool_prompt = str(control_prompt_cfg.get("tool prompt", control_prompt_cfg.get("tool_prompt", "")))

    parser = argparse.ArgumentParser(
        description="Qasr Listener Local Mic: local Python mic capture for an existing qwen-asr-serve."
    )
    parser.add_argument("--config", default=str(config_path), help="Path to config.yaml")
    parser.add_argument("--asr-base-url", default=asr_base_url_from_config(config), help="Existing qwen-asr-serve base URL, including /v1")
    parser.add_argument("--model", default=section_value(config, "asr", "model", "model"), help="Served model name")
    parser.add_argument("--method", choices=["chat", "transcriptions"], default=section_value(config, "asr", "method", "method"), help="Backend API method")
    parser.add_argument("--ui-host", default=section_value(config, "ui", "host", "ui_host"), help="UI host")
    parser.add_argument("--ui-port", type=int, default=int(section_value(config, "ui", "port", "ui_port")), help="UI port")
    parser.add_argument("--min-sec", type=float, default=float(section_value(config, "audio", "min_sec", "min_sec")), help="Do not cut before this segment length")
    parser.add_argument("--max-sec", type=float, default=float(section_value(config, "audio", "max_sec", "max_sec")), help="After this length, cut on the next low-volume block")
    parser.add_argument("--silence-ms", type=float, default=float(section_value(config, "audio", "silence_ms", "silence_ms")), help="Base silence cutter value")
    parser.add_argument("--turn-detection-silence-ms", type=float, default=float(turn_detection_silence_ms), help="ASR silence value shown with long-silence turn debug")
    parser.add_argument("--long-silence-ms", type=int, default=int(float(long_silence_ms)), help="No-ASR-update turn completion timeout")
    parser.add_argument("--stutter-extend-enabled", action=argparse.BooleanOptionalAction, default=bool(stutter_extend_enabled), help="Extend long-silence timing when repeated chars/tokens are detected")
    parser.add_argument("--stutter-extend-mode", default=str(stutter_extend_mode), help="Stutter detector mode")
    parser.add_argument("--stutter-delay-ms", type=int, default=int(float(stutter_delay_ms)), help="Long-silence extra delay per stutter count")
    parser.add_argument("--stutter-delay-max-ms", type=int, default=int(float(stutter_delay_max_ms)), help="Maximum long-silence stutter extension")
    parser.add_argument("--cut-rate", type=float, default=float(section_value(config, "audio", "cut_rate", "cut_rate")), help="Higher value cuts more aggressively")
    parser.add_argument("--cut-tail-ms", type=float, default=float(section_value(config, "audio", "cut_tail_ms", "cut_tail_ms")), help="Extra low-volume tail kept after entering tail mode, in milliseconds")
    parser.add_argument("--voice-start-volume", type=float, default=float(section_value(config, "audio", "voice_start_volume", "voice_start_volume")), help="Mic volume gate; below this value blocks are ignored before speech begins")
    interupt_volume = section_value(config, "audio", "interupt_volume", "interupt_volume", tts_cfg.get("interupt_volume", 0.0))
    parser.add_argument("--tts-interupt-volume", type=float, default=float(interupt_volume))
    parser.add_argument("--interupt-early-release", type=float, default=float(section_value(config, "audio", "interupt_early_release", "interupt_early_release", 0.0)))
    parser.add_argument("--interupt-switch-delay", type=float, default=float(section_value(config, "audio", "interupt_switch_delay", "interupt_switch_delay", 0.5)))
    parser.add_argument("--sample-rate", type=int, default=int(section_value(config, "audio", "sample_rate", "sample_rate")), help="Local microphone capture sample rate")
    parser.add_argument("--block-ms", type=int, default=int(section_value(config, "audio", "block_ms", "block_ms")), help="InputStream block size in milliseconds")
    parser.add_argument("--echo-filter-enabled", action=argparse.BooleanOptionalAction, default=bool(echo_filter_cfg.get("enabled", False)))
    parser.add_argument("--interrupt-languages", default=",".join(parse_language_codes(echo_filter_cfg.get("interrupt language", echo_filter_cfg.get("interrupt_language", "")))))
    parser.add_argument("--interrupt-analyze-prompt", default=str(echo_filter_cfg.get("interrupt analyze prompt", echo_filter_cfg.get("interrupt_analyze_prompt", ""))))
    parser.add_argument("--max-tokens", type=int, default=int(section_value(config, "asr", "max_tokens", "max_tokens")))
    parser.add_argument("--timeout", type=float, default=float(section_value(config, "asr", "timeout_sec", "timeout")))
    parser.add_argument("--llm-base-url", default=str(llm_cfg.get("base_url", "")))
    parser.add_argument("--llm-endpoint", default=str(llm_cfg.get("endpoint", "/chat/completions")))
    parser.add_argument("--llm-model", default=str(llm_cfg.get("model", "")))
    parser.add_argument("--llm-temperature", type=float, default=float(llm_cfg.get("temperature", 0.7)))
    parser.add_argument("--llm-max-tokens", type=int, default=int(llm_cfg.get("max_tokens", 2048)))
    parser.add_argument("--llm-stream", action=argparse.BooleanOptionalAction, default=bool(llm_cfg.get("stream", True)))
    parser.add_argument("--llm-disable-thinking", action=argparse.BooleanOptionalAction, default=bool(llm_cfg.get("disable_thinking", True)))
    parser.add_argument("--llm-timeout-sec", type=float, default=float(llm_cfg.get("timeout_sec", 120)))
    parser.add_argument("--llm-control-prompt", default=persona_prompt)
    parser.add_argument("--llm-persona-prompt", default=persona_prompt)
    parser.add_argument("--llm-tool-prompt", default=tool_prompt)
    parser.add_argument("--llm-control-prompt-inject-threshold", type=int, default=int(control_prompt_cfg.get("inject_threshold", llm_cfg.get("control_prompt_inject_threshold", 1000000))))
    parser.add_argument("--raw-history-rounds", type=int, default=int(control_prompt_cfg.get("raw history rounds", control_prompt_cfg.get("raw_history_rounds", 10))))
    parser.add_argument("--raw-recent-rounds", type=int, default=int(control_prompt_cfg.get("raw recent rounds", control_prompt_cfg.get("raw_recent_rounds", 3))))
    parser.add_argument("--memory-extract-freq", type=int, default=int(control_prompt_cfg.get("memory extract freq", control_prompt_cfg.get("memory_extract_freq", 5))))
    parser.add_argument("--memory-extract-rounds", type=int, default=int(control_prompt_cfg.get("extract rounds", control_prompt_cfg.get("extract_rounds", 10))))
    parser.add_argument("--llm-report-section-id", action=argparse.BooleanOptionalAction, default=bool(control_prompt_cfg.get("report section id", control_prompt_cfg.get("report_section_id", False))))
    parser.add_argument("--tts-enabled", action=argparse.BooleanOptionalAction, default=bool(tts_cfg.get("enabled", True)))
    parser.add_argument("--tts-base-url", default=str(tts_cfg.get("base_url", "http://127.0.0.1:8810")))
    parser.add_argument("--tts-endpoint", default=str(tts_cfg.get("endpoint", "/tts_file")))
    parser.add_argument("--tts-voice", default=str(tts_cfg.get("voice", "auto")))
    parser.add_argument("--tts-language", default=str(tts_cfg.get("language", "")))
    parser.add_argument("--tts-instruct", default=str(tts_cfg.get("instruct", "")))
    parser.add_argument("--tts-speed", type=float, default=float(tts_cfg.get("speed", 1.0)))
    parser.add_argument("--tts-duration", type=float, default=float(tts_cfg.get("duration", 0)))
    parser.add_argument("--tts-num-step", type=int, default=int(tts_cfg.get("num_step", 25)))
    parser.add_argument("--tts-guidance-scale", type=float, default=float(tts_cfg.get("guidance_scale", 2.0)))
    parser.add_argument("--tts-denoise", action=argparse.BooleanOptionalAction, default=bool(tts_cfg.get("denoise", True)))
    parser.add_argument("--tts-queue-ahead", type=int, default=int(tts_cfg.get("queue_ahead", 2)))
    parser.add_argument("--tts-group-sentences", type=int, default=int(tts_cfg.get("group_sentences", 1)))
    parser.add_argument("--tts-interupt-threshold", type=int, default=int(tts_cfg.get("interupt_threshold", 2)))
    parser.add_argument(
        "--tts-chop-exceptions",
        default=list(parse_chop_exceptions(tts_cfg.get("chop_exceptions"))),
        help="TTS chop delimiter exceptions from config.yaml.",
    )
    parser.add_argument(
        "--tts-silence-types",
        default=",".join(parse_silence_types(tts_cfg.get("silence_types"))),
        help="Comma-separated TTS text filters, e.g. markdown_bold,parentheses",
    )
    parser.add_argument("--tts-packet-prefix", default=str(tts_cfg.get("packet_prefix", "")))
    parser.add_argument("--tts-packet-suffix", default=str(tts_cfg.get("packet_suffix", "")))
    parser.add_argument("--tts-timeout-sec", type=float, default=float(tts_cfg.get("timeout_sec", 120)))
    args = parser.parse_args()
    if args.llm_control_prompt != persona_prompt and args.llm_persona_prompt == persona_prompt:
        args.llm_persona_prompt = args.llm_control_prompt
    else:
        args.llm_control_prompt = args.llm_persona_prompt
    args.asr_phonetic_output_languages = asr_phonetic_output_languages
    args.interrupt_languages = parse_language_codes(args.interrupt_languages)
    args.tts_chop_exceptions = parse_chop_exceptions(args.tts_chop_exceptions)
    args.tts_silence_types = parse_silence_types(args.tts_silence_types)
    args.llm_extra_body = llm_cfg.get("extra_body") if isinstance(llm_cfg.get("extra_body"), dict) else None
    return args
