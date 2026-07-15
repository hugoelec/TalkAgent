# QasrGbxOmni

[English](README.en.md) | [繁體中文](README.md)

QasrGbxOmni 是一個本機語音對話實驗工具，把麥克風收音、ASR、LLM 回覆、OmniVoice TTS、Web UI、對話記憶與 EPUB 讀書模式串在同一個 Python/Flask 應用裡。

目前版本偏向「本機工作台」而不是打包好的產品：外部 ASR、LLM、TTS 服務需要自己先啟動，主要設定集中在 `config.yaml` 與 `ControlPrompt.yaml`。

## 目前功能

- 本機麥克風收音，使用音量門檻、最短/最長片段、靜音時間與 tail buffer 切分語音。
- 呼叫 OpenAI-compatible ASR 服務，支援 `chat/completions` 音訊輸入與 `audio/transcriptions` 兩種模式。
- 呼叫 OpenAI-compatible LLM 服務，支援 streaming 回覆。
- 串接 OmniVoice TTS，支援分句、佇列預合成、即時播放與播放中斷。
- Web UI 即時顯示 ASR raw、VAD/切段狀態、LLM 回覆、TTS log、錯誤與歷史紀錄。
- UI 內可即時調整部分 OmniVoice 設定，例如 voice、language、speed、duration、steps、guidance、denoise、packet prefix/suffix。
- `ControlPrompt.yaml` 管理 persona、tool prompt、prompt 注入門檻、raw history 長度、記憶回合設定與 EchoFilter。
- LLM interrupt judge 可在 TTS 播放時判斷新的 ASR 是真正打斷還是回音/誤收音。
- Router prompt 支援在一般對話回合後產生控制碼紀錄。
- BookReader/Reader Mode 支援上傳 EPUB，抽取 XHTML 章節、分析書籍內容、保存 `AnalyzeResault` 與 `SummaryIndex` 為可下載文字包，並讓語音問答只根據書籍分析內容回答。
- 內建 config/editor 面板，可從 UI 編輯 `config.yaml` 與 `ControlPrompt.yaml`。

## 專案結構

```text
QasrBasic.py              # 入口點，載入設定、建立 MicEngine 與 Flask app
config_loader.py          # 讀取 config.yaml / ControlPrompt.yaml，建立 argparse Namespace
web_app.py                # Flask routes、Web UI 狀態 API、BookReader API
mic_engine.py             # 麥克風切段、ASR queue、turn detection、LLM/TTS 串接核心
asr_client.py             # ASR API client
llm_client.py             # OpenAI-compatible chat completions client
omnivoice_tts_client.py   # OmniVoice TTS API client
tts_pipeline.py           # TTS 分句、預合成、播放與中斷
conversation_memory.py    # 對話歷史、prompt 注入與記憶狀態
talker_prompt_switch.py   # 一般模式 / Reader Mode prompt 組裝
interrupt_judge.py        # TTS 播放中斷判斷
templates/index.html      # 單頁 Web UI
config.yaml               # ASR / LLM / TTS / UI / audio 主設定
ControlPrompt.yaml        # persona、EchoFilter、ReaderConfig、RouterConfig
requirement.txt           # Python 套件需求
```

## 環境需求

- Python 3.10 以上建議。
- 可用的麥克風與喇叭裝置。
- 本機或區網內的 ASR 服務，預設為 Qwen3-ASR 風格的 OpenAI-compatible API。
- 本機或區網內的 LLM 服務，預設為 OpenAI-compatible `/chat/completions`。
- 如果啟用 TTS，需要 OmniVoice server，預設 `http://127.0.0.1:8810/tts_file`。

目前 `sounddevice` 會直接使用系統音訊裝置；如果安裝或啟動時遇到 PortAudio/音訊裝置錯誤，要先確認系統音訊 driver 與預設輸入/輸出裝置可用。

## 安裝

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
```

如果是在 Linux/macOS，啟動 venv 的指令改成：

```bash
source .venv/bin/activate
python -m pip install -r requirement.txt
```

## 設定

主要設定在 `config.yaml`：

- `asr`: ASR base URL、endpoint、model、method、timeout。
- `llm`: LLM base URL、endpoint、model、temperature、stream、extra body。
- `tts`: OmniVoice base URL、voice、language、speed、queue、分句與文字過濾。
- `ui`: Flask host 與 port。
- `audio`: 麥克風音量門檻、切段秒數、sample rate、block size、中斷音量門檻。
- `turn_detection`: ASR 停止更新多久後送出一輪 LLM、以及 stutter 延長等待。
- `asr_phonetic_output`: 指定某些 ASR 語言輸出轉拼音/音標。

Prompt 與對話行為設定在 `ControlPrompt.yaml`：

- `Control Prompt.persona`: 主要人格/角色 prompt。
- `Control Prompt.tool prompt`: 每輪可附加的工具/行為 prompt。
- `inject_threshold`: persona prompt 重新注入的 token delta 門檻。
- `raw history rounds` / `raw recent rounds`: UI 保留與送進 LLM 的歷史回合數。
- `EchoFilter`: TTS 播放中是否用 LLM 判斷 ASR 是否為真正中斷。
- `ReaderConfig`: BookReader 的 onLoading、StartReading、onReading、FinishReading、Analyze prompt。
- `RouterConfig`: 一般對話後的 router prompt。

也可以用 CLI override 部分設定，例如：

```powershell
python QasrBasic.py --ui-port 7861 --no-tts-enabled
```

查看完整參數：

```powershell
python QasrBasic.py --help
```

## 啟動

先確認外部服務已啟動，且 `config.yaml` 指向正確位置。預設值大致是：

- ASR: `http://127.0.0.1:9000/v1/chat/completions`
- LLM: `http://192.168.66.146:8000/v1/chat/completions`
- TTS: `http://127.0.0.1:8810/tts_file`
- UI: `http://0.0.0.0:7861`

啟動主程式：

```powershell
python QasrBasic.py
```

瀏覽器打開：

```text
http://127.0.0.1:7861
```

Web UI 上按 `Start mic` 開始收音，按 `Stop` 停止。`MemReset` 會清掉目前對話、prompt 注入與 runtime 記錄。

## Web UI 使用筆記

- `ASR Raw`: 顯示目前累積的 ASR 文字。
- `Voice Active Detection`: 顯示音量、切段、ASR queue、TTS 中斷狀態、stutter 延遲與 long silence 倒數。
- `LLM Reply`: 顯示使用者回合與 assistant 回覆。
- `Config`: 可在 UI 中編輯 `config.yaml` 與 `ControlPrompt.yaml`。
- `BookReader`: 上傳 EPUB、檢視 XHTML 章節、執行分析、下載 BookReader 文字包。
- `OmniVoice Settings`: 可即時調整部分 TTS 參數，修改後會自動送到 runtime 設定。

## BookReader / Reader Mode

BookReader 目前使用 EPUB 作為入口：

1. 在 UI 上傳 `.epub`。
2. 系統抽取 EPUB 裡的 XHTML/HTML 章節與目錄資訊。
3. 可選擇要分析的章節，產生或更新 `AnalyzeResault` 與 `SummaryIndex`。
4. Reader Mode 開啟後，語音問答會使用書籍分析內容作為上下文，不使用一般對話記憶與 router。
5. 下載會輸出一個 `書名-BookReader.txt`，裡面合併 `SummaryIndex` 與 `AnalyzeResault`，下次可跟 EPUB 一起上傳恢復工作狀態。

## API 端點速查

- `GET /health`: 查看目前設定摘要。
- `GET /state`: Web UI 輪詢的 runtime 狀態。
- `POST /start`: 開始麥克風收音。
- `POST /stop`: 停止麥克風收音。
- `POST /mem-reset`: 清空對話與 runtime 記憶。
- `GET/POST /tts-settings`: 讀取或更新 runtime TTS 設定。
- `GET/POST /control-prompt`: 讀取或更新 prompt 注入/記憶設定。
- `POST /control-prompt/inject`: 手動標記下次注入 persona prompt。
- `GET /editor-files`, `GET/POST /editor-file/<file_id>`: UI 設定檔編輯器。
- `POST /book-reader/upload`: 上傳 EPUB 與可選 BookReader sidecar。
- `POST /book-reader/download`: 下載 BookReader 合併文字包。
- `POST /book-reader/xhtml-content`: 取得選取章節文字。
- `POST /reader-llm/send` / `POST /reader-llm/stream`: BookReader 分析聊天。
- `POST /speech-llm/send`: 從 UI 手動送一輪一般 LLM。
- `POST /reader-mode/asr-turn`: 用 Reader Mode context 送一輪語音事件。

## 已知限制

- 這版依賴多個外部本機服務，沒有內建啟動 ASR/LLM/TTS server。
- `ControlPrompt.yaml` 目前含有本機實驗 prompt 內容；若要分享或部署，建議先整理 persona 與 ReaderConfig。
- `requirement.txt` 是目前實際使用的依賴清單，但沒有鎖定完整 transitive dependencies。
- Web UI 是單檔模板，功能集中但尚未拆成前端專案。
- `BookReader` 的 `AnalyzeResault` 拼字沿用既有資料格式，暫時不要改名，避免破壞相容性。
- 程式裡的 `interupt` 拼字也沿用既有設定/API 命名，修改前要同步檢查 config、UI 與 runtime 欄位。

## License

本專案使用 GNU Affero General Public License v3.0。詳見 `LICENSE`。
