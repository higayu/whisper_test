from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import tempfile, os

app = FastAPI()

# 静的ファイルとテンプレート
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# モデルをロード（CPU int8）
model = WhisperModel("small", device="cpu", compute_type="int8")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("whisper.html", {"request": request})

def _save_upload_to_temp(upload: UploadFile) -> str:
    """UploadFile を安全に一時ファイルへ保存してパスを返す"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            # まとめて読む（アップロードが大きいなら分割読み書きに変更可）
            f.write(upload.file.read() if hasattr(upload.file, "read") else b"")
    except Exception:
        # 失敗時はファイルを消してから再送出
        try:
            os.remove(path)
        finally:
            pass
        raise
    return path

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file がありません")
    tmp_path = _save_upload_to_temp(file)
    try:
        segments, info = model.transcribe(
            tmp_path,
            vad_filter=True,
            beam_size=1,
            language="ja"  # 日本語に固定
        )
        segments = list(segments)
        text = "".join(seg.text for seg in segments)
        seg_list = [
            {"id": i, "start": seg.start, "end": seg.end, "text": seg.text}
            for i, seg in enumerate(segments)
        ]
        return JSONResponse(content={
            "language": info.language,  # 言語固定でもここは "ja" が返る想定
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
    tmp_path = _save_upload_to_temp(file)
    try:
        # 精度重視（重くなるので同時実行を増やしすぎない）
        segments, info = model.transcribe(tmp_path, beam_size=5, language="ja")
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
