from __future__ import annotations

import argparse
import io
import logging
import posixpath
from pathlib import Path
import re
import threading
from urllib.parse import quote
import zipfile
from typing import Any, Dict
import xml.etree.ElementTree as ET

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
import yaml
from llm_client import LLMClient
from mic_engine import MicEngine


def create_app(args: argparse.Namespace, engine: MicEngine) -> Flask:
    app = Flask(__name__)
    tts_settings_lock = threading.Lock()
    args.tts_settings_lock = tts_settings_lock
    config_path = Path(str(getattr(args, "config", "config.yaml"))).resolve()
    script_dir = Path(__file__).resolve().parent
    control_prompt_path = (script_dir / "ControlPrompt.yaml").resolve()
    reader_config_path = (script_dir / "ReaderConfig.yaml").resolve()
    editable_files = {
        "config": {
            "label": "config.yaml",
            "language": "yaml",
            "path": config_path,
        },
        "control_prompt": {
            "label": "ControlPrompt.yaml",
            "language": "yaml",
            "path": control_prompt_path,
        },
    }
    book_reader_files: Dict[str, Dict[str, Any]] = {}
    current_epub_items: list[dict[str, Any]] = []
    reader_llm_client = LLMClient(args)

    def config_control_prompts(source: str) -> tuple[str, str]:
        try:
            loaded = yaml.safe_load(source) or {}
        except Exception:
            return (
                str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                str(getattr(args, "llm_tool_prompt", "")),
            )
        if not isinstance(loaded, dict):
            return (
                str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                str(getattr(args, "llm_tool_prompt", "")),
            )
        section = loaded.get("Control Prompt")
        if not isinstance(section, dict):
            section = loaded.get("control_prompt") if isinstance(loaded.get("control_prompt"), dict) else {}
        persona = section.get(
            "persona",
            section.get("text", getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
        )
        tool_prompt = section.get("tool prompt", section.get("tool_prompt", getattr(args, "llm_tool_prompt", "")))
        return str(persona or ""), str(tool_prompt or "")

    def load_control_prompt_config() -> Dict[str, Any]:
        try:
            loaded = yaml.safe_load(control_prompt_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def extract_top_level_section_source(source: str, section_name: str) -> str:
        section_lines: list[str] = []
        in_section = False
        section_pattern = re.compile(rf"^{re.escape(section_name)}\s*:\s*(?:#.*)?$")
        for line in source.splitlines():
            if not in_section:
                if section_pattern.match(line.strip()):
                    in_section = True
                continue
            if line.strip() and re.match(r"^\S", line) and not line.lstrip().startswith("#"):
                break
            section_lines.append(line)
        return "\n".join(section_lines).strip("\n")

    def clean_yaml_scalar_text(value: str) -> str:
        text = str(value or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return text[1:-1].strip()
        return text.strip("\"' ")

    def parse_reader_config_fallback(source: str) -> Dict[str, Any]:
        section_source = extract_top_level_section_source(source, "ReaderConfig")
        if not section_source:
            return {}
        parsed: Dict[str, Any] = {}
        current_section = ""
        current_indent = 0
        for line in section_source.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.match(r"^(\s+)([^:#]+?)\s*:\s*(.*?)\s*$", line)
            if not match:
                continue
            indent = len(match.group(1).replace("\t", "    "))
            key = match.group(2).strip()
            value = match.group(3).strip()
            if not value:
                current_section = key
                current_indent = indent
                parsed.setdefault(current_section, {})
                continue
            if current_section and indent > current_indent and isinstance(parsed.get(current_section), dict):
                parsed[current_section][key] = clean_yaml_scalar_text(value)
            else:
                parsed[key] = clean_yaml_scalar_text(value)
                current_section = ""
                current_indent = 0
        return parsed

    def control_prompt_section(source: str, section_name: str) -> Dict[str, Any]:
        try:
            loaded = yaml.safe_load(source) or {}
        except Exception:
            return {}
        if not isinstance(loaded, dict):
            return {}
        section = loaded.get(section_name)
        return section if isinstance(section, dict) else {}

    def apply_prompt_runtime_config(source: str) -> None:
        control_section = control_prompt_section(source, "Control Prompt")
        echo_section = control_prompt_section(source, "EchoFilter")
        with engine.llm_control_prompt_lock:
            persona_prompt, tool_prompt = config_control_prompts(source)
            args.llm_persona_prompt = persona_prompt
            args.llm_control_prompt = persona_prompt
            args.llm_tool_prompt = tool_prompt
            if "inject_threshold" in control_section:
                engine.control_prompt_inject_threshold = max(0, int(control_section.get("inject_threshold", 0)))
                args.llm_control_prompt_inject_threshold = engine.control_prompt_inject_threshold
            if "raw history rounds" in control_section or "raw_history_rounds" in control_section:
                engine.raw_history_rounds = max(0, int(control_section.get("raw history rounds", control_section.get("raw_history_rounds", 10))))
                args.raw_history_rounds = engine.raw_history_rounds
            if "raw recent rounds" in control_section or "raw_recent_rounds" in control_section:
                engine.raw_recent_rounds = max(0, int(control_section.get("raw recent rounds", control_section.get("raw_recent_rounds", 3))))
                args.raw_recent_rounds = engine.raw_recent_rounds
            if "memory extract freq" in control_section or "memory_extract_freq" in control_section:
                engine.memory.set_memory_extract_config(
                    freq=int(control_section.get("memory extract freq", control_section.get("memory_extract_freq", 0)))
                )
            if "extract rounds" in control_section or "extract_rounds" in control_section:
                engine.memory.set_memory_extract_config(
                    rounds=int(control_section.get("extract rounds", control_section.get("extract_rounds", 0)))
                )
        if echo_section:
            engine.echo_filter_enabled = bool(echo_section.get("enabled", engine.echo_filter_enabled))
            lang_value = echo_section.get("interrupt language", echo_section.get("interrupt_language", ""))
            if isinstance(lang_value, str):
                engine.interrupt_languages = {item.strip().lower() for item in lang_value.split(",") if item.strip()}
            elif isinstance(lang_value, list):
                engine.interrupt_languages = {str(item).strip().lower() for item in lang_value if str(item).strip()}
            engine.interrupt_analyze_prompt = str(
                echo_section.get(
                    "interrupt analyze prompt",
                    echo_section.get("interrupt_analyze_prompt", engine.interrupt_analyze_prompt),
                )
                or ""
            )

    def safe_file_stem(name: str) -> str:
        stem = Path(name).stem.strip()
        stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", stem, flags=re.UNICODE).strip("._")
        return stem or "book"

    def register_book_file(file_id: str, label: str, path: Path, language: str = "plaintext") -> None:
        book_reader_files[file_id] = {
            "label": label,
            "language": language,
            "path": Path(path).resolve(),
        }

    def registered_book_file_path(file_id: str) -> Path:
        info = book_reader_files.get(file_id)
        if info is None:
            raise KeyError(file_id)
        return Path(info["path"]).resolve()

    def book_file_stem(name: str) -> str:
        stem = Path(name).stem.strip()
        return stem or safe_file_stem(name)

    def read_upload_text(uploaded_file: Any) -> str:
        raw = uploaded_file.read()
        try:
            return raw.decode("utf-8-sig")
        except Exception:
            return raw.decode("utf-8", errors="replace")

    SUMMARY_INDEX_MARKER = "===== SummaryIndex ====="
    ANALYZE_RESAULT_MARKER = "===== AnalyzeResault ====="

    def split_combined_book_reader_text(text: str) -> Dict[str, str]:
        source = str(text or "")
        summary_pos = source.find(SUMMARY_INDEX_MARKER)
        analyze_pos = source.find(ANALYZE_RESAULT_MARKER)
        if summary_pos < 0 or analyze_pos < 0 or analyze_pos <= summary_pos:
            return {}
        summary_start = summary_pos + len(SUMMARY_INDEX_MARKER)
        analyze_start = analyze_pos + len(ANALYZE_RESAULT_MARKER)
        return {
            "summary_index": source[summary_start:analyze_pos].strip("\r\n"),
            "analyze_resault": source[analyze_start:].strip("\r\n"),
        }

    def combined_book_reader_text(summary_index: str, analyze_resault: str) -> str:
        return (
            f"{SUMMARY_INDEX_MARKER}\n"
            f"{str(summary_index or '').rstrip()}\n\n"
            f"{ANALYZE_RESAULT_MARKER}\n"
            f"{str(analyze_resault or '').rstrip()}\n"
        )

    def sidecar_kind(filename: str) -> str:
        normalized = Path(filename).name.lower().replace(" ", "")
        if "analyzeresault" in normalized or "analyzeresult" in normalized:
            return "analyze_resault"
        if "summaryindex" in normalized:
            return "summary_index"
        return ""

    def uploaded_sidecar_contents(uploaded_files: list[Any]) -> tuple[Dict[str, str], Dict[str, str]]:
        contents: Dict[str, str] = {}
        names: Dict[str, str] = {}
        for item in uploaded_files:
            name = Path(str(item.filename or "")).name
            text = read_upload_text(item)
            combined = split_combined_book_reader_text(text)
            if combined:
                contents.update(combined)
                names.update({file_id: name for file_id in combined})
                continue
            kind = sidecar_kind(name)
            if not kind:
                continue
            contents[kind] = text
            names[kind] = name
        return contents, names

    def download_file_name(book_name: str) -> str:
        return f"{book_file_stem(book_name)}-BookReader.txt"

    def xml_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return " ".join(part.strip() for part in node.itertext() if part and part.strip())

    def zip_posix_join(base_dir: str, href: str) -> str:
        href_path = str(href or "").split("#", 1)[0]
        return posixpath.normpath(posixpath.join(base_dir, href_path)).lstrip("/")

    def epub_xhtml_text(raw: bytes) -> str:
        try:
            root = ET.fromstring(raw)
            parts: list[str] = []
            block_tags = {
                "p",
                "div",
                "section",
                "article",
                "header",
                "footer",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "li",
                "blockquote",
            }
            for element in root.iter():
                tag = str(element.tag).rsplit("}", 1)[-1].lower()
                if tag in {"script", "style", "svg"}:
                    continue
                if tag in block_tags:
                    text = xml_text(element)
                    if text:
                        parts.append(text)
            if parts:
                return "\n\n".join(parts)
            return xml_text(root)
        except Exception:
            text = raw.decode("utf-8", errors="replace")
            text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.I | re.S)
            text = re.sub(r"<[^>]+>", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

    def parse_epub_xhtml_index(raw: bytes) -> list[dict[str, Any]]:
        with zipfile.ZipFile(io.BytesIO(raw)) as epub:
            names = set(epub.namelist())
            container = ET.fromstring(epub.read("META-INF/container.xml"))
            rootfile = container.find(".//{*}rootfile")
            if rootfile is None or not rootfile.get("full-path"):
                raise ValueError("EPUB container.xml missing package.opf path")
            opf_path = rootfile.get("full-path", "")
            opf_dir = posixpath.dirname(opf_path)
            opf = ET.fromstring(epub.read(opf_path))

            manifest: dict[str, dict[str, str]] = {}
            for item in opf.findall(".//{*}manifest/{*}item"):
                item_id = item.get("id") or ""
                href = item.get("href") or ""
                if not item_id or not href:
                    continue
                full_path = zip_posix_join(opf_dir, href)
                manifest[item_id] = {
                    "id": item_id,
                    "href": href,
                    "path": full_path,
                    "media_type": item.get("media-type") or "",
                    "properties": item.get("properties") or "",
                }

            spine_ids = [
                itemref.get("idref") or ""
                for itemref in opf.findall(".//{*}spine/{*}itemref")
                if itemref.get("idref")
            ]
            spine_order = {item_id: index for index, item_id in enumerate(spine_ids, start=1)}

            toc_titles: dict[str, str] = {}
            spine = opf.find(".//{*}spine")
            toc_id = spine.get("toc") if spine is not None else ""
            toc_item = manifest.get(toc_id or "")
            if toc_item and toc_item["path"] in names:
                try:
                    ncx = ET.fromstring(epub.read(toc_item["path"]))
                    ncx_dir = posixpath.dirname(toc_item["path"])
                    for nav_point in ncx.findall(".//{*}navPoint"):
                        label = xml_text(nav_point.find(".//{*}navLabel/{*}text"))
                        content = nav_point.find(".//{*}content")
                        src = content.get("src") if content is not None else ""
                        if label and src:
                            toc_titles[zip_posix_join(ncx_dir, src)] = label
                except Exception:
                    pass

            nav_items = [
                item for item in manifest.values() if "nav" in item.get("properties", "").split()
            ]
            for nav_item in nav_items:
                if nav_item["path"] not in names:
                    continue
                try:
                    nav_root = ET.fromstring(epub.read(nav_item["path"]))
                    nav_dir = posixpath.dirname(nav_item["path"])
                    for link in nav_root.findall(".//{*}a"):
                        href = link.get("href") or ""
                        label = xml_text(link)
                        if href and label:
                            toc_titles.setdefault(zip_posix_join(nav_dir, href), label)
                except Exception:
                    pass

            items: list[dict[str, Any]] = []
            for item in manifest.values():
                path = item["path"]
                media_type = item["media_type"].lower()
                if path not in names:
                    continue
                if not (
                    media_type in {"application/xhtml+xml", "text/html"}
                    or path.lower().endswith((".xhtml", ".html", ".htm"))
                ):
                    continue
                item_id = item["id"]
                title = toc_titles.get(path) or toc_titles.get(f"{path}#") or Path(path).stem
                text = epub_xhtml_text(epub.read(path))
                items.append(
                    {
                        "id": item_id,
                        "title": title,
                        "path": path,
                        "href": item["href"],
                        "media_type": item["media_type"],
                        "spine_index": spine_order.get(item_id),
                        "in_spine": item_id in spine_order,
                        "chars": len(text),
                        "text": text,
                    }
                )
            items.sort(key=lambda row: (row["spine_index"] is None, row["spine_index"] or 999999, row["path"]))
            return items

    def reader_config_source() -> str:
        try:
            control_source = control_prompt_path.read_text(encoding="utf-8")
        except Exception:
            control_source = ""
        prompt_config = load_control_prompt_config()
        reader_section = prompt_config.get("ReaderConfig")
        if isinstance(reader_section, dict):
            raw = reader_section.get("Raw")
            if isinstance(raw, str) and raw.strip():
                return raw
            return yaml.safe_dump(reader_section, allow_unicode=True, sort_keys=False)
        section_source = extract_top_level_section_source(control_source, "ReaderConfig")
        if section_source:
            return section_source
        try:
            return reader_config_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def reader_config_dict() -> Dict[str, Any]:
        try:
            control_source = control_prompt_path.read_text(encoding="utf-8")
        except Exception:
            control_source = ""
        prompt_config = load_control_prompt_config()
        reader_section = prompt_config.get("ReaderConfig")
        if isinstance(reader_section, dict):
            raw = reader_section.get("Raw")
            if isinstance(raw, str) and raw.strip():
                try:
                    loaded = yaml.safe_load(raw) or {}
                except Exception:
                    return parse_reader_config_fallback(raw)
                return loaded if isinstance(loaded, dict) else {}
            return reader_section
        fallback = parse_reader_config_fallback(control_source)
        if fallback:
            return fallback
        try:
            loaded = yaml.safe_load(reader_config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def reader_event_prompt(event_name: str) -> str:
        loaded = reader_config_dict()
        event_config = loaded.get(event_name)
        if isinstance(event_config, dict):
            return str(event_config.get("Prompt", "") or "").strip()
        if isinstance(event_config, str):
            return clean_yaml_scalar_text(event_config)
        source = reader_config_source()
        in_event = False
        for line in source.splitlines():
            direct_match = re.match(rf"^\s*{re.escape(event_name)}\s*:\s*(.+?)\s*$", line)
            if direct_match:
                return clean_yaml_scalar_text(direct_match.group(1))
            if re.match(r"^\S", line):
                in_event = line.strip() == f"{event_name}:"
                continue
            if in_event:
                match = re.match(r"^\s+Prompt\s*:\s*(.*)$", line)
                if match:
                    raw_value = match.group(1).strip()
                    try:
                        parsed = yaml.safe_load(raw_value)
                        return str(parsed if parsed is not None else "").strip()
                    except Exception:
                        return raw_value.strip("\"' ")
        return ""

    def reader_on_loading_prompt() -> str:
        return reader_event_prompt("onLoading")

    def reader_on_loading_talker_text(book_name: str, prompt: str) -> str:
        title = Path(str(book_name or "book.epub")).name
        question = str(prompt or "").strip()
        if question:
            return f"上傳{title} {question}"
        return f"上傳{title}"

    def reader_analyze_config() -> Dict[str, Any]:
        loaded = reader_config_dict()
        analyze = loaded.get("Analyze")
        return analyze if isinstance(analyze, dict) else {}

    def reader_analyze_int_value(key: str, default: int = 0) -> int:
        analyze = reader_analyze_config()
        value = analyze.get(key)
        if value is None:
            source = reader_config_source()
            in_analyze = False
            for line in source.splitlines():
                if re.match(r"^\S", line):
                    in_analyze = line.strip() == "Analyze:"
                    continue
                if in_analyze:
                    match = re.match(rf"^\s+{re.escape(key)}\s*:\s*(.+?)\s*$", line)
                    if match:
                        value = match.group(1).strip().strip("\"'")
                        break
        try:
            match = re.search(r"\d+", str(value or ""))
            return max(0, int(match.group(0))) if match else max(0, int(default))
        except (TypeError, ValueError):
            return max(0, int(default))

    def reader_analyze_bool_value(key: str, default: bool = False) -> bool:
        analyze = reader_analyze_config()
        value = analyze.get(key)
        if value is None:
            source = reader_config_source()
            in_analyze = False
            for line in source.splitlines():
                if re.match(r"^\S", line):
                    in_analyze = line.strip() == "Analyze:"
                    continue
                if in_analyze:
                    match = re.match(rf"^\s+{re.escape(key)}\s*:\s*(.+?)\s*$", line)
                    if match:
                        value = match.group(1).strip().strip("\"'")
                        break
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
        return bool(default)

    def reader_llm_turn(user_text: str) -> str:
        return "".join(reader_llm_client.stream_reply(user_text, history=[])).strip()

    def current_tts_settings() -> Dict[str, Any]:
        with tts_settings_lock:
            return {
                "tts_voice": str(args.tts_voice),
                "tts_language": str(args.tts_language),
                "tts_instruct": str(args.tts_instruct),
                "tts_speed": float(args.tts_speed),
                "tts_duration": float(args.tts_duration),
                "tts_num_step": int(args.tts_num_step),
                "tts_guidance_scale": float(args.tts_guidance_scale),
                "tts_denoise": bool(args.tts_denoise),
                "tts_packet_prefix": str(args.tts_packet_prefix),
                "tts_packet_suffix": str(args.tts_packet_suffix),
            }

    def clamp_float(value: Any, minimum: float, maximum: float) -> float:
        parsed = float(value)
        return max(minimum, min(maximum, parsed))

    def clamp_int(value: Any, minimum: int, maximum: int) -> int:
        parsed = int(value)
        return max(minimum, min(maximum, parsed))

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            min_sec=float(args.min_sec),
            max_sec=float(args.max_sec),
            silence_ms=float(args.silence_ms),
            turn_detection_silence_ms=float(args.turn_detection_silence_ms),
            long_silence_ms=int(args.long_silence_ms),
            stutter_delay_ms=int(args.stutter_delay_ms),
            stutter_delay_max_ms=int(args.stutter_delay_max_ms),
            cut_rate=float(args.cut_rate),
            cut_tail_ms=float(args.cut_tail_ms),
            voice_start_volume=float(args.voice_start_volume),
            sample_rate=int(args.sample_rate),
            block_ms=int(args.block_ms),
            tts_engine="omnivoice",
            tts_enabled=bool(args.tts_enabled),
            tts_base_url=str(args.tts_base_url),
            tts_endpoint=str(args.tts_endpoint),
            tts_voice=str(args.tts_voice),
            tts_language=str(args.tts_language),
            tts_instruct=str(args.tts_instruct),
            tts_speed=float(args.tts_speed),
            tts_duration=float(args.tts_duration),
            tts_num_step=int(args.tts_num_step),
            tts_guidance_scale=float(args.tts_guidance_scale),
            tts_denoise=bool(args.tts_denoise),
            tts_packet_prefix=str(args.tts_packet_prefix),
            tts_packet_suffix=str(args.tts_packet_suffix),
            tts_queue_ahead=int(args.tts_queue_ahead),
            tts_group_sentences=int(args.tts_group_sentences),
            control_prompt=str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
            persona_prompt=str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
            tool_prompt=str(getattr(args, "llm_tool_prompt", "")),
            control_prompt_inject_threshold=int(engine.control_prompt_inject_threshold),
            raw_history_rounds=int(getattr(args, "raw_history_rounds", 10)),
            raw_recent_rounds=int(getattr(args, "raw_recent_rounds", 3)),
            memory_extract_freq=int(engine.memory.memory_extract_freq),
            memory_extract_rounds=int(engine.memory.memory_extract_rounds),
        )

    @app.get("/tts-settings")
    def get_tts_settings() -> Dict[str, Any]:
        return {"ok": True, **current_tts_settings()}

    @app.post("/tts-settings")
    def update_tts_settings() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            updates: Dict[str, Any] = {}
            if "tts_voice" in payload:
                updates["tts_voice"] = str(payload["tts_voice"]).strip() or "auto"
            if "tts_language" in payload:
                updates["tts_language"] = str(payload["tts_language"]).strip()
            if "tts_instruct" in payload:
                updates["tts_instruct"] = str(payload["tts_instruct"])
            if "tts_speed" in payload:
                updates["tts_speed"] = clamp_float(payload["tts_speed"], 0.1, 3.0)
            if "tts_duration" in payload:
                updates["tts_duration"] = max(0.0, float(payload["tts_duration"]))
            if "tts_num_step" in payload:
                updates["tts_num_step"] = clamp_int(payload["tts_num_step"], 1, 200)
            if "tts_guidance_scale" in payload:
                updates["tts_guidance_scale"] = clamp_float(payload["tts_guidance_scale"], 0.0, 20.0)
            if "tts_denoise" in payload:
                updates["tts_denoise"] = bool(payload["tts_denoise"])
            if "tts_packet_prefix" in payload:
                updates["tts_packet_prefix"] = str(payload["tts_packet_prefix"])
            if "tts_packet_suffix" in payload:
                updates["tts_packet_suffix"] = str(payload["tts_packet_suffix"])

            with tts_settings_lock:
                for key, value in updates.items():
                    setattr(args, key, value)

            return jsonify(ok=True, **current_tts_settings())
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc), **current_tts_settings()), 400

    @app.get("/control-prompt")
    def get_control_prompt() -> Dict[str, Any]:
        with engine.llm_control_prompt_lock:
            return {
                "ok": True,
                "control_prompt": str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                "persona_prompt": str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                "tool_prompt": str(getattr(args, "llm_tool_prompt", "")),
                "current_tokens": int(engine.memory.control_prompt_current_tokens),
                "token_since_inject": int(engine.memory.control_prompt_token_since_inject),
                "delta_tokens": int(engine.memory.control_prompt_delta_tokens),
                "inject_threshold": int(engine.control_prompt_inject_threshold),
                "manual_inject": bool(engine.memory.control_prompt_manual_inject),
                "raw_history_rounds": int(engine.raw_history_rounds),
                "raw_recent_rounds": int(engine.raw_recent_rounds),
                "memory_round_current": int(engine.memory.memory_round_current),
                "memory_round_since_extract": int(engine.memory.memory_round_since_extract),
                "memory_extract_freq": int(engine.memory.memory_extract_freq),
                "memory_extract_rounds": int(engine.memory.memory_extract_rounds),
            }

    @app.post("/control-prompt")
    def update_control_prompt() -> Response:
        payload = request.get_json(silent=True) or {}
        inject_threshold = payload.get("inject_threshold", None)
        raw_history_rounds = payload.get("raw_history_rounds", None)
        raw_recent_rounds = payload.get("raw_recent_rounds", None)
        memory_extract_freq = payload.get("memory_extract_freq", None)
        memory_extract_rounds = payload.get("memory_extract_rounds", None)
        with engine.llm_control_prompt_lock:
            if inject_threshold is not None:
                engine.control_prompt_inject_threshold = max(0, int(inject_threshold))
                args.llm_control_prompt_inject_threshold = engine.control_prompt_inject_threshold
        with engine.lock:
            if raw_history_rounds is not None:
                engine.raw_history_rounds = max(0, int(raw_history_rounds))
                args.raw_history_rounds = engine.raw_history_rounds
                engine.memory.trim_all()
            if raw_recent_rounds is not None:
                engine.raw_recent_rounds = max(0, int(raw_recent_rounds))
                args.raw_recent_rounds = engine.raw_recent_rounds
            engine.memory.set_memory_extract_config(
                freq=int(memory_extract_freq) if memory_extract_freq is not None else None,
                rounds=int(memory_extract_rounds) if memory_extract_rounds is not None else None,
            )
        with engine.llm_control_prompt_lock:
            return jsonify(
                ok=True,
                control_prompt=str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                persona_prompt=str(getattr(args, "llm_persona_prompt", args.llm_control_prompt)),
                tool_prompt=str(getattr(args, "llm_tool_prompt", "")),
                current_tokens=int(engine.memory.control_prompt_current_tokens),
                token_since_inject=int(engine.memory.control_prompt_token_since_inject),
                delta_tokens=int(engine.memory.control_prompt_delta_tokens),
                inject_threshold=int(engine.control_prompt_inject_threshold),
                manual_inject=bool(engine.memory.control_prompt_manual_inject),
                raw_history_rounds=int(engine.raw_history_rounds),
                raw_recent_rounds=int(engine.raw_recent_rounds),
                memory_round_current=int(engine.memory.memory_round_current),
                memory_round_since_extract=int(engine.memory.memory_round_since_extract),
                memory_extract_freq=int(engine.memory.memory_extract_freq),
                memory_extract_rounds=int(engine.memory.memory_extract_rounds),
            )

    @app.post("/control-prompt/inject")
    def manual_inject_control_prompt() -> Response:
        with engine.llm_control_prompt_lock:
            engine.memory.mark_manual_inject()
            return jsonify(
                ok=True,
                current_tokens=int(engine.memory.control_prompt_current_tokens),
                token_since_inject=int(engine.memory.control_prompt_token_since_inject),
                delta_tokens=int(engine.memory.control_prompt_delta_tokens),
                inject_threshold=int(engine.control_prompt_inject_threshold),
                manual_inject=bool(engine.memory.control_prompt_manual_inject),
                raw_history_rounds=int(engine.raw_history_rounds),
                raw_recent_rounds=int(engine.raw_recent_rounds),
                memory_round_current=int(engine.memory.memory_round_current),
                memory_round_since_extract=int(engine.memory.memory_round_since_extract),
                memory_extract_freq=int(engine.memory.memory_extract_freq),
                memory_extract_rounds=int(engine.memory.memory_extract_rounds),
            )

    @app.get("/editor-files")
    def editor_files() -> Dict[str, Any]:
        files = []
        for file_id, info in editable_files.items():
            files.append(
                {
                    "id": file_id,
                    "label": str(info["label"]),
                    "language": str(info["language"]),
                    "virtual": bool(info.get("virtual", False)),
                }
            )
        return {"ok": True, "files": files}

    @app.get("/editor-file/<file_id>")
    def get_editor_file(file_id: str) -> Response:
        info = editable_files.get(file_id)
        if info is None:
            return jsonify(ok=False, error="unknown file"), 404
        path = Path(info["path"]).resolve()
        allowed_paths = {config_path, control_prompt_path}
        if path not in allowed_paths:
            return jsonify(ok=False, error="file not allowed"), 403
        try:
            return jsonify(
                ok=True,
                id=file_id,
                label=str(info["label"]),
                language=str(info["language"]),
                content=path.read_text(encoding="utf-8"),
                path=str(path),
            )
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.post("/editor-file/<file_id>")
    def save_editor_file(file_id: str) -> Response:
        info = editable_files.get(file_id)
        if info is None:
            return jsonify(ok=False, error="unknown file"), 404
        payload = request.get_json(silent=True) or {}
        content = str(payload.get("content", ""))
        path = Path(info["path"]).resolve()
        allowed_paths = {config_path, control_prompt_path}
        if path not in allowed_paths:
            return jsonify(ok=False, error="file not allowed"), 403
        try:
            path.write_text(content, encoding="utf-8")
            if file_id == "control_prompt":
                apply_prompt_runtime_config(content)
            return jsonify(ok=True, id=file_id)
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.get("/book-reader/files")
    def list_book_reader_files() -> Dict[str, Any]:
        files = []
        for file_id, info in book_reader_files.items():
            files.append(
                {
                    "id": file_id,
                    "label": str(info["label"]),
                    "language": str(info["language"]),
                    "path": str(info["path"]),
                }
            )
        return {"ok": True, "files": files}

    @app.post("/book-reader/upload")
    def upload_book() -> Response:
        nonlocal current_epub_items
        uploaded_files = list(request.files.getlist("files"))
        legacy_book = request.files.get("book")
        if legacy_book is not None and legacy_book.filename:
            uploaded_files.insert(0, legacy_book)
        if not uploaded_files:
            return jsonify(ok=False, error="missing book file"), 400
        epub_upload = next(
            (item for item in uploaded_files if Path(str(item.filename or "")).suffix.lower() == ".epub"),
            None,
        )
        if epub_upload is None or not epub_upload.filename:
            return jsonify(ok=False, error="one .epub file is required"), 400
        raw_uploaded_filename = str(epub_upload.filename)
        filename = Path(raw_uploaded_filename).name
        stem = safe_file_stem(filename)
        sidecar_contents, sidecar_names = uploaded_sidecar_contents(
            [item for item in uploaded_files if item is not epub_upload]
        )
        raw_epub = epub_upload.read()
        xhtml_items = parse_epub_xhtml_index(raw_epub)
        current_epub_items = xhtml_items
        if "analyze_resault" in sidecar_contents:
            analyze_content = sidecar_contents["analyze_resault"]
            loaded_analyze_resault = True
        else:
            analyze_content = f"{filename}\n"
            loaded_analyze_resault = False
        if "summary_index" in sidecar_contents:
            summary_content = sidecar_contents["summary_index"]
            loaded_summary_index = True
        else:
            summary_content = ""
            loaded_summary_index = False
        book_reader_files.clear()
        with engine.lock:
            engine.reader_mode = True
            engine.reader_analyze_resault = analyze_content
            engine.reader_summary_index = summary_content
        on_loading_prompt = reader_on_loading_prompt()
        on_loading_talker_text = ""
        if on_loading_prompt:
            on_loading_talker_text = reader_on_loading_talker_text(filename, on_loading_prompt)
            try:
                engine.start_asr_text_turn(on_loading_talker_text, source="ReaderOnLoadingASR")
            except Exception:
                logging.exception("failed to start reader onLoading talker turn")
        xhtml_index = [
            {
                "id": item["id"],
                "title": item["title"],
                "path": item["path"],
                "href": item["href"],
                "spine_index": item["spine_index"],
                "in_spine": item["in_spine"],
                "chars": item["chars"],
                "checked": bool(item["in_spine"]),
            }
            for item in xhtml_items
        ]
        return jsonify(
            ok=True,
            saved=False,
            book_name=filename,
            stem=stem,
            storage_mode="uploaded_bundle" if sidecar_names else "draft",
            uploaded_sidecars=sidecar_names,
            loaded_analyze_resault=loaded_analyze_resault,
            loaded_summary_index=loaded_summary_index,
            on_loading_prompt=on_loading_prompt,
            on_loading_talker_text=on_loading_talker_text,
            xhtml_index=xhtml_index,
            files=[
                {
                    "id": "analyze_resault",
                    "label": "AnalyzeResault",
                    "language": "markdown",
                    "content": analyze_content,
                    "exists": loaded_analyze_resault,
                    "path": "",
                },
                {
                    "id": "summary_index",
                    "label": "SummaryIndex",
                    "language": "plaintext",
                    "content": summary_content,
                    "exists": loaded_summary_index,
                    "path": "",
                },
            ],
        )

    @app.post("/book-reader/save-workspace")
    def save_book_reader_workspace() -> Response:
        return jsonify(ok=False, error="workspace save disabled; use /book-reader/download"), 410

    @app.post("/book-reader/download")
    def download_book_reader_files() -> Response:
        payload = request.get_json(silent=True) or {}
        filename = Path(str(payload.get("book_name", "book.epub"))).name
        if Path(filename).suffix.lower() != ".epub":
            filename = f"{safe_file_stem(filename)}.epub"
        files_payload = payload.get("files") if isinstance(payload.get("files"), dict) else {}
        download_text = combined_book_reader_text(
            summary_index=str(files_payload.get("summary_index", "")),
            analyze_resault=str(files_payload.get("analyze_resault", "")),
        )
        output_name = download_file_name(filename)
        return Response(
            download_text,
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(output_name)}"},
        )

    @app.post("/reader-mode/context")
    def update_reader_mode_context() -> Response:
        payload = request.get_json(silent=True) or {}
        analyze_resault = str(payload.get("analyze_resault", "") or "")
        summary_index = str(payload.get("summary_index", "") or "")
        now_chapter = str(payload.get("now_chapter", "") or "")
        reading_mode_update = payload.get("reading_mode", None)
        with engine.lock:
            engine.reader_analyze_resault = analyze_resault
            engine.reader_summary_index = summary_index
            engine.reader_now_chapter = now_chapter
            if reading_mode_update is not None:
                engine.reading_mode = bool(reading_mode_update)
            reader_mode = bool(engine.reader_mode)
            reading_mode = bool(engine.reading_mode)
        return jsonify(
            ok=True,
            reader_mode=reader_mode,
            reading_mode=reading_mode,
            chars=len(analyze_resault),
            summary_index_chars=len(summary_index),
            now_chapter_chars=len(now_chapter),
        )

    @app.post("/book-reader/unload")
    def unload_book_reader_workspace() -> Dict[str, Any]:
        nonlocal current_epub_items
        dropped_turns = engine.cancel_speech_turns("reader unload")
        book_reader_files.clear()
        current_epub_items = []
        with engine.lock:
            engine.reader_mode = False
            engine.reading_mode = False
            engine.reader_analyze_resault = ""
            engine.reader_summary_index = ""
            engine.reader_now_chapter = ""
        return {"ok": True, "files": [], "dropped_speech_turns": dropped_turns}

    @app.post("/book-reader/xhtml-content")
    def get_xhtml_content() -> Response:
        payload = request.get_json(silent=True) or {}
        selected_ids = payload.get("ids")
        if not isinstance(selected_ids, list):
            return jsonify(ok=False, error="ids must be a list"), 400
        wanted = {str(item) for item in selected_ids}
        sections = []
        for item in current_epub_items:
            if str(item["id"]) not in wanted:
                continue
            sections.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "path": item["path"],
                    "text": item["text"],
                    "chars": item["chars"],
                }
            )
        if not sections:
            return jsonify(ok=False, error="no selected XHTML content"), 400
        return jsonify(ok=True, sections=sections)

    @app.post("/reader-llm/send")
    def send_reader_llm() -> Response:
        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("text", "")).strip()
        if not user_text:
            return jsonify(ok=False, error="empty text"), 400
        try:
            response_text = reader_llm_turn(user_text)
            return jsonify(ok=True, text=response_text)
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.post("/reader-llm/stream")
    def stream_reader_llm() -> Response:
        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("text", "")).strip()
        if not user_text:
            return jsonify(ok=False, error="empty text"), 400

        def generate():
            try:
                for delta in reader_llm_client.stream_reply(user_text, history=[]):
                    if not delta:
                        continue
                    yield delta
            except Exception as exc:
                yield f"\n[Analyzer error: {exc!r}]"

        return Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8")

    @app.post("/speech-llm/send")
    def send_speech_llm() -> Response:
        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("text", "")).strip()
        if not user_text:
            return jsonify(ok=False, error="empty text"), 400
        try:
            engine.start_manual_llm_turn(user_text)
            return jsonify(ok=True)
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.post("/reader-mode/asr-turn")
    def send_reader_mode_asr_turn() -> Response:
        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("text", "")).strip()
        context = str(payload.get("context", "") or "")
        if not user_text:
            return jsonify(ok=False, error="empty text"), 400
        try:
            engine.start_asr_text_turn(
                user_text,
                source="ReaderEventASR",
                reader_context_override=context,
            )
            return jsonify(ok=True, context_chars=len(context))
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.get("/reader-analyze-config")
    def get_reader_analyze_config() -> Dict[str, Any]:
        analyze = reader_analyze_config()
        return {
            "ok": True,
            "AnalyzePrompt": str(analyze.get("AnalyzePrompt", "") or ""),
            "AnalyzeAll": reader_analyze_bool_value("AnalyzeAll", False),
            "AutoAnalyze": reader_analyze_bool_value("AutoAnalyze", False),
            "ResaultPromptOn": reader_analyze_bool_value("ResaultPromptOn", False),
            "ResaultPrompt": str(analyze.get("ResaultPrompt", "") or ""),
            "CptShortRest": reader_analyze_int_value("CptShortRest", 0),
            "StartReadingPrompt": reader_event_prompt("StartReading"),
            "onReadingPrompt": reader_event_prompt("onReading"),
            "FinishReadingPrompt": reader_event_prompt("FinishReading"),
        }

    @app.get("/book-reader/file/<file_id>")
    def get_book_reader_file(file_id: str) -> Response:
        info = book_reader_files.get(file_id)
        if info is None:
            return jsonify(ok=False, error="unknown book reader file"), 404
        path = registered_book_file_path(file_id)
        try:
            return jsonify(
                ok=True,
                id=file_id,
                label=str(info["label"]),
                language=str(info["language"]),
                content=path.read_text(encoding="utf-8"),
            )
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.post("/book-reader/file/<file_id>")
    def save_book_reader_file(file_id: str) -> Response:
        info = book_reader_files.get(file_id)
        if info is None:
            return jsonify(ok=False, error="unknown book reader file"), 404
        payload = request.get_json(silent=True) or {}
        content = str(payload.get("content", ""))
        path = registered_book_file_path(file_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return jsonify(ok=True, id=file_id, path=str(path))
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc)), 500

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "asr_base_url": args.asr_base_url,
            "model": args.model,
            "method": args.method,
            "min_sec": args.min_sec,
            "max_sec": args.max_sec,
            "silence_ms": args.silence_ms,
            "turn_detection_silence_ms": args.turn_detection_silence_ms,
            "long_silence_ms": args.long_silence_ms,
            "stutter_extend_enabled": args.stutter_extend_enabled,
            "stutter_extend_mode": args.stutter_extend_mode,
            "stutter_delay_ms": args.stutter_delay_ms,
            "stutter_delay_max_ms": args.stutter_delay_max_ms,
            "asr_phonetic_output_languages": sorted(args.asr_phonetic_output_languages),
            "llm_base_url": args.llm_base_url,
            "llm_endpoint": args.llm_endpoint,
            "llm_model": args.llm_model,
            "llm_stream": args.llm_stream,
            "llm_control_prompt": args.llm_control_prompt,
            "llm_persona_prompt": getattr(args, "llm_persona_prompt", args.llm_control_prompt),
            "llm_tool_prompt": getattr(args, "llm_tool_prompt", ""),
            "control_prompt_current_tokens": engine.memory.control_prompt_current_tokens,
            "control_prompt_token_since_inject": engine.memory.control_prompt_token_since_inject,
            "control_prompt_inject_threshold": engine.control_prompt_inject_threshold,
            "raw_history_rounds": engine.raw_history_rounds,
            "raw_recent_rounds": engine.raw_recent_rounds,
            "interrupt_languages": sorted(engine.interrupt_languages),
            "memory_round_current": engine.memory.memory_round_current,
            "memory_round_since_extract": engine.memory.memory_round_since_extract,
            "memory_extract_freq": engine.memory.memory_extract_freq,
            "memory_extract_rounds": engine.memory.memory_extract_rounds,
            "tts_engine": "omnivoice",
            "tts_enabled": args.tts_enabled,
            "tts_base_url": args.tts_base_url,
            "tts_endpoint": args.tts_endpoint,
            "tts_voice": args.tts_voice,
            "tts_language": args.tts_language,
            "tts_instruct": args.tts_instruct,
            "tts_speed": args.tts_speed,
            "tts_duration": args.tts_duration,
            "tts_num_step": args.tts_num_step,
            "tts_guidance_scale": args.tts_guidance_scale,
            "tts_denoise": args.tts_denoise,
            "tts_queue_ahead": args.tts_queue_ahead,
            "tts_group_sentences": args.tts_group_sentences,
            "tts_packet_prefix": args.tts_packet_prefix,
            "tts_packet_suffix": args.tts_packet_suffix,
            "cut_rate": args.cut_rate,
            "cut_tail_ms": args.cut_tail_ms,
            "voice_start_volume": args.voice_start_volume,
            "sample_rate": args.sample_rate,
            "block_ms": args.block_ms,
        }

    @app.get("/state")
    def state() -> Dict[str, Any]:
        return engine.snapshot()

    @app.post("/start")
    def start() -> Response:
        try:
            engine.start()
            return jsonify(ok=True, **engine.snapshot())
        except Exception as exc:
            return jsonify(ok=False, error=repr(exc), **engine.snapshot()), 500

    @app.post("/stop")
    def stop() -> Dict[str, Any]:
        engine.stop()
        return {"ok": True, **engine.snapshot()}

    @app.post("/mem-reset")
    def mem_reset() -> Dict[str, Any]:
        engine.memory.reset_history()
        return {"ok": True, **engine.snapshot()}

    return app


class IgnoreStateLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "GET /state HTTP/" not in message
