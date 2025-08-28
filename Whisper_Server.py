from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import tempfile, os

app = FastAPI()

# 📌 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# モデルをあらかじめロード
model = WhisperModel("small", device="cpu", compute_type="int8")

# ルートに HTML を表示
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("whisper.html", {"request": request})

# 音声アップロード & 文字起こし
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_path = tempfile.mktemp(suffix=".wav")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Whisper で文字起こし（言語を日本語 "ja" に固定）
    segments, info = model.transcribe(
        tmp_path,
        vad_filter=True,
        beam_size=1,
        language="ja"   # ← 日本語に固定
    )
    segments = list(segments)

    text = "".join([seg.text for seg in segments])
    seg_list = [
        {"id": i, "start": seg.start, "end": seg.end, "text": seg.text}
        for i, seg in enumerate(segments)
    ]

    os.remove(tmp_path)

    return JSONResponse(content={
        "language": info.language,  # ← ここも "ja" になる
        "text": text,
        "segments": seg_list
    })
