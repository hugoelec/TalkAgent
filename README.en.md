# QasrGbxOmni

[English](README.en.md) | [Traditional Chinese](README.md)

QasrGbxOmni is a local voice conversation and reading-machine workbench that connects microphone capture, ASR, LLM replies, OmniVoice TTS, a Flask Web UI, conversation memory, and an EPUB reader mode in one Python application.

This version is closer to a local lab tool than a packaged product. You need to run the external ASR, LLM, and TTS services yourself, then point `config.yaml` and `ControlPrompt.yaml` at the correct endpoints and prompts.

## Features

- Local microphone capture with volume gating, min/max segment length, silence-based cutting, and tail buffering.
- OpenAI-compatible ASR client with support for `chat/completions` audio input and `audio/transcriptions`.
- OpenAI-compatible LLM client with streaming replies.
- OmniVoice TTS integration with sentence chunking, queued pre-synthesis, playback, and interrupt handling.
- Web UI for live ASR text, VAD/cutting status, LLM replies, TTS logs, errors, and runtime history.
- Runtime TTS controls for voice, language, speed, duration, inference steps, guidance scale, denoise, and packet prefix/suffix.
- Central prompt configuration in `ControlPrompt.yaml`, including persona, tool prompt, prompt injection threshold, raw history windows, memory counters, and EchoFilter.
- LLM-based interrupt judge to decide whether ASR captured during TTS playback is a real interruption or echo/noise.
- Router prompt support for post-turn control-code records.
- Reading Machine / BookReader / Reader Mode for uploading EPUB files, extracting XHTML chapters, analyzing book content, saving `AnalyzeResault` and `SummaryIndex`, and answering voice questions from book context only.
- Built-in config editor for `config.yaml` and `ControlPrompt.yaml`.

## Project Layout

```text
QasrBasic.py              # Entry point: loads config, creates MicEngine and Flask app
config_loader.py          # Loads config.yaml / ControlPrompt.yaml into an argparse Namespace
web_app.py                # Flask routes, Web UI state APIs, BookReader APIs
mic_engine.py             # Mic cutting, ASR queue, turn detection, LLM/TTS orchestration
asr_client.py             # ASR API client
llm_client.py             # OpenAI-compatible chat completions client
omnivoice_tts_client.py   # OmniVoice TTS API client
tts_pipeline.py           # TTS chunking, pre-synthesis, playback, interruption
conversation_memory.py    # Conversation history, prompt injection, memory state
talker_prompt_switch.py   # Normal mode / Reader Mode prompt builder
interrupt_judge.py        # TTS interruption decision helper
templates/index.html      # Single-page Web UI
config.yaml               # Main ASR / LLM / TTS / UI / audio settings
ControlPrompt.yaml        # Persona, EchoFilter, ReaderConfig, RouterConfig
requirement.txt           # Python dependencies
```

## Requirements

- Python 3.10 or newer is recommended.
- A usable microphone and speaker/output device.
- A local or LAN ASR service, currently expected to behave like a Qwen3-ASR style OpenAI-compatible API.
- A local or LAN LLM service with an OpenAI-compatible `/chat/completions` endpoint.
- OmniVoice server if TTS is enabled. The default endpoint is `http://127.0.0.1:8810/tts_file`.

The app uses `sounddevice` for direct system audio capture and playback. If startup fails with PortAudio or device errors, check your system audio driver and default input/output devices first.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirement.txt
```

## Configuration

Main service and runtime settings live in `config.yaml`:

- `asr`: ASR base URL, endpoint, model, method, and timeout.
- `llm`: LLM base URL, endpoint, model, temperature, streaming, and extra request body.
- `tts`: OmniVoice base URL, voice, language, speed, queue settings, chunking, and text filters.
- `ui`: Flask host and port.
- `audio`: microphone volume thresholds, cutting windows, sample rate, block size, and interrupt threshold.
- `turn_detection`: how long to wait after ASR stops updating before sending a turn to the LLM, plus stutter-based wait extension.
- `asr_phonetic_output`: languages whose ASR output should be converted to phonetic text.

Prompt and behavior settings live in `ControlPrompt.yaml`:

- `Control Prompt.persona`: the main persona / character prompt.
- `Control Prompt.tool prompt`: an optional per-turn behavior/tool prompt.
- `inject_threshold`: token delta threshold for re-injecting the persona prompt.
- `raw history rounds` / `raw recent rounds`: how many history rounds are shown in the UI and sent back to the LLM.
- `EchoFilter`: whether to use an LLM to judge ASR captured during TTS playback.
- `ReaderConfig`: BookReader prompts for onLoading, StartReading, onReading, FinishReading, and Analyze.
- `RouterConfig`: router prompt used after normal conversation turns.

You can override selected settings from the CLI:

```powershell
python QasrBasic.py --ui-port 7861 --no-tts-enabled
```

Show all CLI options:

```powershell
python QasrBasic.py --help
```

## Running

Start the external services first and make sure `config.yaml` points to them. The current defaults are roughly:

- ASR: `http://127.0.0.1:9000/v1/chat/completions`
- LLM: `http://192.168.66.146:8000/v1/chat/completions`
- TTS: `http://127.0.0.1:8810/tts_file`
- UI: `http://0.0.0.0:7861`

Run the app:

```powershell
python QasrBasic.py
```

Open the UI:

```text
http://127.0.0.1:7861
```

Press `Start mic` to begin microphone capture and `Stop` to stop. `MemReset` clears the current conversation, prompt injection state, and runtime logs.

## Web UI Notes

- `ASR Raw`: current accumulated ASR text.
- `Voice Active Detection`: volume, audio cutting state, ASR queue, TTS interruption state, stutter delay, and long-silence countdown.
- `LLM Reply`: user turns and assistant replies.
- `Config`: edit `config.yaml` and `ControlPrompt.yaml` from the browser.
- `BookReader`: the reading-machine workspace for uploading EPUB files, inspecting XHTML chapters, running analysis, and downloading BookReader text bundles.
- `OmniVoice Settings`: adjust runtime TTS parameters. Changes are posted back to the running app automatically.

## Reading Machine / BookReader / Reader Mode

The reading machine currently starts from an EPUB file. Its job is to split a book into analyzable chapters, then turn the analysis result into context for later voice Q&A:

1. Upload a `.epub` file in the UI.
2. The app extracts XHTML/HTML chapters and table-of-contents metadata.
3. Select chapters to analyze, then generate or update `AnalyzeResault` and `SummaryIndex`.
4. When Reader Mode is active, voice questions use the book analysis context and avoid normal conversation memory and router prompts.
5. Downloading produces a `BookName-BookReader.txt` file that combines `SummaryIndex` and `AnalyzeResault`. Upload it together with the EPUB later to restore the reader workspace.

Reading-machine prompts live in `ReaderConfig` inside `ControlPrompt.yaml`, including prompts for book loading, start reading, while reading, finish reading, and chapter analysis. Normal chat mode and reading-machine mode use different prompt-building paths so book Q&A does not accidentally mix with regular conversation memory.

## API Quick Reference

- `GET /health`: current configuration summary.
- `GET /state`: runtime state polled by the Web UI.
- `POST /start`: start microphone capture.
- `POST /stop`: stop microphone capture.
- `POST /mem-reset`: clear conversation and runtime memory.
- `GET/POST /tts-settings`: read or update runtime TTS settings.
- `GET/POST /control-prompt`: read or update prompt injection / memory settings.
- `POST /control-prompt/inject`: manually mark the next persona prompt injection.
- `GET /editor-files`, `GET/POST /editor-file/<file_id>`: config editor endpoints.
- `POST /book-reader/upload`: upload an EPUB and optional BookReader sidecar.
- `POST /book-reader/download`: download the combined BookReader text bundle.
- `POST /book-reader/xhtml-content`: get selected chapter text.
- `POST /reader-llm/send` / `POST /reader-llm/stream`: BookReader analyzer chat.
- `POST /speech-llm/send`: manually send a normal LLM turn from the UI.
- `POST /reader-mode/asr-turn`: send a Reader Mode voice event with reader context.

## Known Limitations

- This version depends on external ASR, LLM, and TTS services. It does not start those servers for you.
- `ControlPrompt.yaml` currently contains local experiment prompts. Clean up persona and ReaderConfig before sharing or deploying.
- `requirement.txt` lists the direct dependencies used by the app, but it does not lock the full transitive dependency graph.
- The Web UI is a single template file. It is featureful, but not yet split into a dedicated frontend project.
- The `AnalyzeResault` spelling is kept for compatibility with existing BookReader data. Avoid renaming it unless all callers and saved files are migrated.
- The `interupt` spelling is also preserved in existing config/API/runtime names. Changing it requires checking config, UI, and runtime fields together.

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE`.
