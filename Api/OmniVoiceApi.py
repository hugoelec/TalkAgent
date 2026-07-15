import io
import uuid
import time
import threading
import logging
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device


# ============================================================
# 基本設定
# ============================================================

MODEL_ID = "k2-fsa/OmniVoice"

BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR / "cloneVoice"

VOICE_DIR.mkdir(parents=True, exist_ok=True)

# 單張 GPU 先不要併發 generate，避免 VRAM 抖動 / OOM / 延遲亂跳
generate_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
)


# ============================================================
# 載入模型：API 啟動時只載一次
# ============================================================

DEVICE = get_best_device()

logging.info(f"Loading OmniVoice model from {MODEL_ID} on {DEVICE} ...")

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    dtype=torch.float16,
)

logging.info("OmniVoice model loaded.")


# ============================================================
# Request Schema
# ============================================================

class TTSRequest(BaseModel):
    # 必填
    text: str = Field(..., description="要合成的文字")

    # cloneVoice/<voice>/ref.wav + prompt.txt
    voice: str = Field("default", description="聲音資料夾名稱")

    # 也可以直接指定 ref_audio / ref_text，優先於 voice
    ref_audio: Optional[str] = Field(None, description="參考音訊路徑")
    ref_text: Optional[str] = Field(None, description="參考音訊逐字稿")

    # Voice design / instruction
    instruct: Optional[str] = Field(None, description="聲音風格指令，例如 male, British accent")
    language: Optional[str] = Field(None, description="語言名稱或代碼，例如 English / en / zh")

    # Generation Settings
    speed: float = Field(1.0, description="語速，1.0 正常，>1 較快，<1 較慢。若 duration 有效則可能被忽略")
    duration: Optional[float] = Field(None, description="固定輸出秒數。None 或 <=0 代表不用固定長度")
    num_step: int = Field(32, description="推理步數。低一點較快，高一點品質較好")
    guidance_scale: float = Field(2.0, description="CFG guidance scale")
    denoise: bool = Field(True, description="是否啟用 denoise")
    postprocess_output: bool = Field(True, description="是否移除生成音訊中的長靜音")

    # infer.py 裡也有的進階參數
    t_shift: float = Field(0.1, description="t_shift")
    layer_penalty_factor: float = Field(5.0, description="layer_penalty_factor")
    position_temperature: float = Field(5.0, description="position_temperature")
    class_temperature: float = Field(0.0, description="class_temperature")

    # API 額外處理
    preprocess_prompt: bool = Field(
        True,
        description="目前只做 ref_text 結尾補標點；不做音訊靜音修剪",
    )
    output_name: Optional[str] = Field(None, description="指定輸出檔名，不含副檔名也可以")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="OmniVoice Local TTS API")


# ============================================================
# 工具函式
# ============================================================

def ensure_punctuation(text: Optional[str]) -> Optional[str]:
    """
    簡易 preprocess prompt：
    如果 ref_text 結尾沒有標點，補一個句號。
    demo UI 裡的 Preprocess Prompt 還可能包含音訊 trimming / silence removal，
    這版先不做音訊前處理。
    """
    if not text:
        return text

    text = text.strip()
    if not text:
        return text

    if text[-1] not in ".。!！?？,，;；:：":
        text += "。"

    return text


def get_voice_files(voice: str) -> tuple[Optional[str], Optional[str]]:
    """
    單層 voice 檔案結構：

    cloneVoice/
      jpChild.wav
      jpChild.txt
      girl_cute.wav
      girl_cute.txt

    voice="jpChild" 時讀：
      cloneVoice/jpChild.wav
      cloneVoice/jpChild.txt
    """
    wav_path = VOICE_DIR / f"{voice}.wav"
    txt_path = VOICE_DIR / f"{voice}.txt"

    ref_audio = str(wav_path) if wav_path.exists() else None

    ref_text = None
    if txt_path.exists():
        ref_text = txt_path.read_text(encoding="utf-8").strip()

    return ref_audio, ref_text


def make_output_filename(output_name: Optional[str]) -> str:
    if output_name:
        safe_name = Path(output_name).stem
        return f"{safe_name}.wav"

    return f"{uuid.uuid4().hex}.wav"


def normalize_duration(duration: Optional[float]) -> Optional[float]:
    """
    UI 裡 Duration 留空才代表不用固定長度。
    但有些前端可能會送 0，所以 API 這邊把 <=0 視為 None。
    """
    if duration is None:
        return None

    if duration <= 0:
        return None

    return duration


def resolve_reference(req: TTSRequest) -> tuple[Optional[str], Optional[str]]:
    """
    優先權：
    1. request 直接帶 ref_audio / ref_text
    2. cloneVoice/<voice>/ref.wav + prompt.txt
    """
    ref_audio = req.ref_audio
    ref_text = req.ref_text

    if not ref_audio:
        voice_ref_audio, voice_ref_text = get_voice_files(req.voice)
        ref_audio = voice_ref_audio

        if not ref_text:
            ref_text = voice_ref_text

    if ref_audio and not Path(ref_audio).exists():
        raise HTTPException(status_code=400, detail=f"ref_audio not found: {ref_audio}")

    if req.preprocess_prompt:
        ref_text = ensure_punctuation(ref_text)

    return ref_audio, ref_text


def generate_wav_memory(req: TTSRequest) -> tuple[io.BytesIO, dict]:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    ref_audio, ref_text = resolve_reference(req)
    duration = normalize_duration(req.duration)
    filename = make_output_filename(req.output_name)

    generate_kwargs = {
        "text": text,
        "language": req.language,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "instruct": req.instruct,
        "duration": duration,
        "num_step": req.num_step,
        "guidance_scale": req.guidance_scale,
        "speed": req.speed,
        "t_shift": req.t_shift,
        "denoise": req.denoise,
        "postprocess_output": req.postprocess_output,
        "layer_penalty_factor": req.layer_penalty_factor,
        "position_temperature": req.position_temperature,
        "class_temperature": req.class_temperature,
    }

    generate_kwargs = {
        key: value for key, value in generate_kwargs.items()
        if value is not None
    }

    logging.info(
        "Generating: text=%r voice=%s ref_audio=%s language=%s instruct=%s "
        "speed=%s duration=%s num_step=%s",
        text[:80],
        req.voice,
        ref_audio,
        req.language,
        req.instruct,
        req.speed,
        duration,
        req.num_step,
    )

    start = time.time()

    # 單張 GPU 只允許一次 generate；編碼與傳輸仍可由各請求獨立處理。
    with generate_lock:
        audios = model.generate(**generate_kwargs)

    elapsed = time.time() - start
    wav = audios[0] if isinstance(audios, (list, tuple)) else audios

    # WAV 只寫進 RAM，不建立 outputs 檔案。
    audio_buffer = io.BytesIO()
    sf.write(
        audio_buffer,
        wav,
        model.sampling_rate,
        format="WAV",
        subtype="PCM_16",
    )
    audio_buffer.seek(0)

    logging.info(
        "Generated in memory: filename=%s bytes=%d elapsed=%.3fs",
        filename,
        audio_buffer.getbuffer().nbytes,
        elapsed,
    )

    metadata = {
        "filename": filename,
        "sampling_rate": model.sampling_rate,
        "elapsed_sec": round(elapsed, 3),
    }
    return audio_buffer, metadata


def make_audio_response(req: TTSRequest) -> StreamingResponse:
    audio_buffer, metadata = generate_wav_memory(req)

    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{metadata["filename"]}"',
            "X-Sampling-Rate": str(metadata["sampling_rate"]),
            "X-Elapsed-Sec": str(metadata["elapsed_sec"]),
        },
    )


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "sampling_rate": model.sampling_rate,
    }


@app.get("/voices")
def voices():
    """
    掃描單層 cloneVoice/*.wav。
    例如：
      cloneVoice/jpChild.wav
      cloneVoice/jpChild.txt
    """
    result = []

    if VOICE_DIR.exists():
        for wav_path in sorted(VOICE_DIR.glob("*.wav")):
            voice_name = wav_path.stem
            txt_path = VOICE_DIR / f"{voice_name}.txt"

            prompt_preview = None
            if txt_path.exists():
                prompt_preview = txt_path.read_text(encoding="utf-8").strip()[:80]

            result.append({
                "voice": voice_name,
                "wav_path": str(wav_path),
                "txt_path": str(txt_path),
                "has_ref_wav": wav_path.exists(),
                "has_prompt_txt": txt_path.exists(),
                "prompt_preview": prompt_preview,
            })

    return {
        "voice_dir": str(VOICE_DIR),
        "voices": result,
    }


@app.post("/tts")
def tts(req: TTSRequest):
    """
    產生音訊並直接從記憶體回傳 WAV，不寫入硬碟。
    """
    return make_audio_response(req)


@app.post("/tts_file")
def tts_file(req: TTSRequest):
    """
    與 /tts 相同；保留舊端點名稱以相容既有客戶端。
    """
    return make_audio_response(req)
