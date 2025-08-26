# services/transcriber.py
from __future__ import annotations  # 任意（型注釈の評価を遅延）
import logging
from typing import Optional
from faster_whisper import WhisperModel        # ★ これが必要
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE

log = logging.getLogger("whisper_app")

class WhisperService:
    _model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            log.info(
                "Loading Whisper model size=%s device=%s compute=%s",
                WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE
            )
            self._model = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
            )
            log.info("Model loaded")
        return self._model

    def get_model(self) -> WhisperModel:
        return self._ensure_model()

    def transcribe_path(self, path: str, language: Optional[str] = None, **kwargs):
        model = self._ensure_model()
        segments, info = model.transcribe(path, language=language, **kwargs)
        text = "".join(seg.text for seg in segments)
        return text, info
