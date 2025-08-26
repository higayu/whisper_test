from flask import Flask
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from config import BASE_DIR, LOG_PATH, SESSION_DIR, TMP_DIR
from routes.ui import bp as ui_bp
from routes.health import bp as health_bp
from routes.transcribe import bp as transcribe_bp
from routes.realtime import bp as rt_bp
from routes import health as health_mod

from routes.recorder import bp as recorder_bp

def create_app():
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
    ...
    app.register_blueprint(ui_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(transcribe_bp)
    app.register_blueprint(rt_bp)
    
    ...
    return app

def create_app():
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

    # ログ
    logger = logging.getLogger("whisper_app")
    logger.setLevel(logging.DEBUG)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("=== app start ===")

    # ディレクトリ準備
    (SESSION_DIR).mkdir(exist_ok=True)
    (TMP_DIR).mkdir(exist_ok=True)

    # Blueprints
    app.register_blueprint(ui_bp)
    # health はセッション数を参照するのでストア注入（realtime の store を共有）
    from routes.realtime import store as shared_store
    health_mod.store = shared_store
    app.register_blueprint(health_bp)
    app.register_blueprint(transcribe_bp)
    app.register_blueprint(rt_bp)
    app.register_blueprint(recorder_bp)  # ← 追加
    logger.info("template_folder = %s", app.template_folder)
    return app

app = create_app()

