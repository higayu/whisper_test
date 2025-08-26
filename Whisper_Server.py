# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile, os

app = FastAPI()

# モデルをあらかじめロード
model = WhisperModel("small", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 一時ファイルに保存
    tmp_path = tempfile.mktemp(suffix=".wav")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # 文字起こし（1回だけ呼ぶ）
    segments, info = model.transcribe(tmp_path, vad_filter=True, beam_size=1)
    segments = list(segments)  # ★ここでジェネレーターをリスト化

    text = "".join([seg.text for seg in segments])
    seg_list = [
        {"id": i, "start": seg.start, "end": seg.end, "text": seg.text}
        for i, seg in enumerate(segments)
    ]


    os.remove(tmp_path)
    return JSONResponse(content={
        "language": info.language,
        "text": text,
        "segments": seg_list
    })

# 起動:
# uvicorn server:app --host 0.0.0.0 --port 8000
