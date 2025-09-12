# app.py
import os
import asyncio
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---- ここではモデルを作らない（Noneにしておく） ----
model = None                    # type: Optional["WhisperModel"]
model_ready = False
sem = asyncio.Semaphore(1)      # 同時実行を制限（CPU版は1~2推奨）

app = FastAPI()

# 静的/テンプレート（ディレクトリが存在することを確認）
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if os.path.isdir("templates") else None

@app.get("/")
async def root(request: Request):
    if templates is None:
        return {"service": "whisper-api", "ready": model_ready}
    return templates.TemplateResponse("whisper.html", {"request": request, "ready": model_ready})

@app.get("/health")
async def health():
    return {"status": "ok" if model_ready else "starting"}

# ---- 重い初期化はバックグラウンドで ----
def _load_model_sync():
    from faster_whisper import WhisperModel
    # GPU (CUDA) を利用、float16 推奨
    return WhisperModel("medium", device="cuda", compute_type="float32")


async def _load_model_task():
    global model, model_ready
    # 同期処理をスレッドプールに逃がす
    model = await asyncio.to_thread(_load_model_sync)
    model_ready = True

@app.on_event("startup")
async def on_startup():
    # 起動ブロックしないようにタスク化
    asyncio.create_task(_load_model_task())

# ---- アップロードを一時ファイルへ保存（非同期で少しずつ） ----
async def _save_upload_to_temp(upload: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)  # 1MBずつ
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            os.remove(path)
        finally:
            pass
        raise
    finally:
        await upload.close()
    return path

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file がありません")
    if not model_ready:
        # 起動直後はまだロード中の可能性
        raise HTTPException(status_code=503, detail="model is loading")

    tmp_path = await _save_upload_to_temp(file)
    try:
        async with sem:  # 同時実行を制限
            # CPUだとVAD有りは少し重い。必要に応じてfalse/beam_size調整
            segments, info = await asyncio.to_thread(
                model.transcribe,
                tmp_path,
                vad_filter=True,
                beam_size=1,
                language="ja"
            )
        segments = list(segments)
        text = "".join(seg.text for seg in segments)
        seg_list = [
            {"id": i, "start": seg.start, "end": seg.end, "text": seg.text}
            for i, seg in enumerate(segments)
        ]
        return JSONResponse(content={
            "language": info.language,
            "text": text,
            "segments": seg_list
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/transcribe_long")
async def transcribe_long(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file がありません")
    if not model_ready:
        raise HTTPException(status_code=503, detail="model is loading")

    tmp_path = await _save_upload_to_temp(file)
    try:
        async with sem:
            segments, info = await asyncio.to_thread(
                model.transcribe,
                tmp_path,
                beam_size=5,
                language="ja"
            )
        segments = list(segments)
        text = "".join(seg.text for seg in segments)
        seg_list = [
            {"id": i, "start": seg.start, "end": seg.end, "text": seg.text}
            for i, seg in enumerate(segments)
        ]
        return JSONResponse(content={
            "language": info.language,
            "text": text,
            "segments": seg_list
        })
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
