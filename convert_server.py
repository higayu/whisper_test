# convert_server.py
import os
import uuid
import asyncio
import tempfile
import subprocess
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ---- 許可する入力拡張子／出力フォーマット ----
ALLOWED_INPUT_EXTS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg", ".flac", ".aac", ".opus", ".wma"
}
ALLOWED_OUTPUT_FMTS = {"mp3", "wav", "flac", "ogg", "opus", "m4a"}

# ---- テンプレート設定 ----
templates = Jinja2Templates(directory="templates") if os.path.isdir("templates") else None

# ---- 同時変換を制限（負荷に合わせて調整）----
sem = asyncio.Semaphore(2)

app = FastAPI(title="ffmpeg-converter", version="1.0.0")


@app.get("/")
async def root(request: Request):
    if templates is None:
        return {"service": "ffmpeg-converter", "ready": True}
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ok = True
    except Exception:
        ok = False
    return {"status": "ok" if ok else "ffmpeg-missing"}


# ---- アップロードを一時保存 ----
async def _save_to_temp(src: UploadFile) -> str:
    _, ext = os.path.splitext(src.filename or "")
    ext = ext.lower() if ext else ".bin"
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await src.read(1024 * 1024)  # 1MBずつ
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
        await src.close()
    return path


# ---- ffmpeg コマンドを組み立て ----
def _build_ffmpeg_cmd(input_path, output_path, out_fmt, ar, ac, ab):
    ffmpeg_bin = "/usr/bin/ffmpeg"   # ← フルパス指定
    args = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]
    
    # サンプルレート／チャンネル
    if ar:
        args += ["-ar", str(ar)]
    if ac:
        args += ["-ac", str(ac)]

    # 出力コーデック
    if out_fmt == "mp3":
        args += ["-c:a", "libmp3lame"]
        if ab:
            args += ["-b:a", ab]
    elif out_fmt == "wav":
        args += ["-c:a", "pcm_s16le"]  # 16-bit PCM
    elif out_fmt == "flac":
        args += ["-c:a", "flac"]
    elif out_fmt == "ogg":
        args += ["-c:a", "libvorbis"]
        if ab:
            args += ["-b:a", ab]
    elif out_fmt == "opus":
        args += ["-c:a", "libopus"]
        if ab:
            args += ["-b:a", ab]
    elif out_fmt == "m4a":
        args += ["-c:a", "aac"]
        if ab:
            args += ["-b:a", ab]
    else:
        raise HTTPException(status_code=400, detail=f"未対応の出力形式: {out_fmt}")

    args += [output_path]
    return args


@app.post("/convert")
async def convert(
    file: UploadFile = File(..., description="入力ファイル"),
    fmt: str = Query("mp3", description="出力フォーマット"),
    ar: Optional[int] = Query(16000, description="サンプルレート(Hz)"),
    ac: Optional[int] = Query(1, description="チャンネル数"),
    ab: Optional[str] = Query("64k", description="ビットレート(例: 64k/96k/128k)"),
    download_name: Optional[str] = Query(None, description="返却ファイル名（拡張子は自動付与）"),
):
    # 入力バリデーション
    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext and ext.lower() not in ALLOWED_INPUT_EXTS:
            raise HTTPException(status_code=415, detail=f"対応外の拡張子: {ext}")
    if fmt.lower() not in ALLOWED_OUTPUT_FMTS:
        raise HTTPException(status_code=400, detail=f"対応出力: {sorted(ALLOWED_OUTPUT_FMTS)}")

    # 一時保存
    src_path = await _save_to_temp(file)
    out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.{fmt.lower()}")

    try:
        async with sem:
            cmd = _build_ffmpeg_cmd(src_path, out_path, fmt, ar, ac, ab)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60 * 30)
                if proc.returncode != 0:
                    raise HTTPException(status_code=400, detail=f"変換失敗: {stderr.decode(errors='ignore')[:500]}")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="変換タイムアウト")
    finally:
        # 入力側はすぐ削除
        try:
            os.remove(src_path)
        except Exception:
            pass

    # ストリーミング返却
    def _iter():
        with open(out_path, "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                yield b

    # ダウンロード名
    if not download_name:
        base = os.path.splitext(os.path.basename(file.filename or "converted"))[0]
        download_name = f"{base}.{fmt.lower()}"
    else:
        base, _ = os.path.splitext(download_name)
        download_name = f"{base}.{fmt.lower()}"

    # Content-Type
    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",  # 容器に依存
        "m4a": "audio/mp4",
    }
    media_type = mime_map.get(fmt.lower(), "application/octet-stream")

    async def _cleanup():
        try:
            os.remove(out_path)
        except Exception:
            pass

    from starlette.background import BackgroundTask
    resp = StreamingResponse(_iter(), media_type=media_type, background=BackgroundTask(_cleanup))
    resp.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return resp
