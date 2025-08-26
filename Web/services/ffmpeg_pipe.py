import subprocess, threading, logging

log = logging.getLogger("whisper_app")

class FFmpegPipe:
    """
    MediaRecorder が吐く webm/ogg/mp4 の“断片”を stdin で連結して
    PCM(s16le, 16k, mono) を stdout から取り出す。
    """
    def __init__(self, mtype: str, write_pcm_cb):
        self.mtype = (mtype or "").lower()
        self.write_pcm_cb = write_pcm_cb
        self.p: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None
        self.lock = threading.Lock()

    def _in_args(self):
        mt = self.mtype
        if "ogg" in mt:   return ["-f", "ogg"]
        if "webm" in mt:  return ["-f", "webm"]
        if "mp4" in mt or "mpeg" in mt: return ["-f", "mp4"]
        return []

    def start(self):
        if self.p: return
        cmd = ["ffmpeg","-hide_banner","-nostdin","-loglevel","error",
               *self._in_args(), "-i","-","-ac","1","-ar","16000","-f","s16le","-"]
        log.info("[FF] spawn: %s", " ".join(cmd))
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def _reader():
            try:
                while True:
                    buf = self.p.stdout.read(4096)
                    if not buf: break
                    self.write_pcm_cb(buf)
            except Exception as e:
                log.info("[FF] reader end: %s", e)

        self.reader = threading.Thread(target=_reader, daemon=True)
        self.reader.start()

    def write(self, data: bytes) -> bool:
        with self.lock:
            if not self.p or not self.p.stdin: return False
            try:
                self.p.stdin.write(data)
                self.p.stdin.flush()
                return True
            except Exception as e:
                log.info("[FF] write error: %s", e)
                return False

    def close(self, wait_timeout=3.0):
        try:
            if self.p and self.p.stdin:
                try: self.p.stdin.close()
                except: pass
            if self.p:
                try: self.p.wait(timeout=wait_timeout)
                except: pass
            if self.reader:
                try: self.reader.join(timeout=1.0)
                except: pass
        finally:
            self.p = None
            self.reader = None
