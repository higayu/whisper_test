from dataclasses import dataclass, field
from typing import Optional, Dict, List
import wave, threading, queue, os

@dataclass
class RTSession:
    sid: str
    language: Optional[str]
    wav_path: str
    wav: wave.Wave_write
    pcm_frames: int = 0
    q: "queue.Queue[None]" = field(default_factory=queue.Queue)
    stop_flag: bool = False
    last_decoded_sec: float = 0.0
    text_accum: str = ""
    sse_subs: List[queue.Queue] = field(default_factory=list)
    # ffmpeg pipe
    ff = None
    ff_lock: threading.Lock = field(default_factory=threading.Lock)
    ff_reader = None

class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, RTSession] = {}

    def add(self, sess: RTSession):
        with self._lock:
            self._data[sess.sid] = sess

    def get(self, sid: str) -> Optional[RTSession]:
        with self._lock:
            return self._data.get(sid)

    def pop(self, sid: str) -> Optional[RTSession]:
        with self._lock:
            return self._data.pop(sid, None)

    def count(self) -> int:
        with self._lock:
            return len(self._data)

def open_wav_writer(path: str) -> wave.Wave_write:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = wave.open(path, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    return w
