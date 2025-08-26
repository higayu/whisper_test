from flask import Blueprint, jsonify
from services.sessions import SessionStore
import shutil, os

bp = Blueprint("health", __name__)
store: SessionStore | None = None  # app.py から注入

@bp.get("/healthz")
def healthz():
    return jsonify({
        "model_loaded": False,  # ここは必要なら services から取得
        "sessions": store.count() if store else 0,
        "cwd": os.getcwd(),
        "ffmpeg_in_path": shutil.which("ffmpeg") is not None
    })
