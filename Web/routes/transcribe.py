from flask import Blueprint, request, jsonify
import os, time, logging
from services.transcriber import WhisperService
from config import TMP_DIR

bp = Blueprint("transcribe", __name__)
whisper = WhisperService()
log = logging.getLogger("whisper_app")

@bp.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files["audio"]
    lang = request.form.get("language") or None
    os.makedirs(TMP_DIR, exist_ok=True)
    path = TMP_DIR / f"{int(time.time()*1000)}_{f.filename}"
    f.save(path)
    try:
        text, info = whisper.transcribe_path(str(path), language=lang)
        log.info("[TRANSCRIBE] ok lang=%s dur=%.2fs len=%d", info.language, info.duration, len(text))
        return jsonify({"language": info.language, "duration": info.duration, "text": text})
    except Exception as e:
        log.exception("[TRANSCRIBE] error: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except: pass
