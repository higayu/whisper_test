from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
SESSION_DIR = BASE_DIR / "rt_sessions"
TMP_DIR = BASE_DIR / "tmp"
LOG_PATH = BASE_DIR / "server.log"

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
