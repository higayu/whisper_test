"""
Tkinter live (manual-send) transcription using faster-whisper server, 3s chunks buffered locally.

Install in your venv (client):
    pip install sounddevice soundfile numpy requests

Server (separate):
    pip install fastapi uvicorn faster-whisper python-multipart
    uvicorn Whisper_Server:app --reload --host 0.0.0.0 --port 8000
"""

import os
import csv
import queue
import threading
import tempfile
from pathlib import Path
import requests
import datetime as dt
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf

# ====== 設定 ======
SERVER_URL = "http://localhost:8000/transcribe"

SAMPLE_RATE = 16000
CHANNELS = 1
SUBTYPE = "PCM_16"
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 0.5


def now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_tc(x: float, sep=",") -> str:
    ms = int(round(max(0.0, float(x)) * 1000))
    hh, rem = divmod(ms, 3600_000); mm, rem = divmod(rem, 60_000); ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ms:03d}"


class LiveRecorder:
    """マイク録音してWAVへ保存しつつ、メモリにも流す役"""
    def __init__(self, sr=SAMPLE_RATE, ch=CHANNELS, subtype=SUBTYPE):
        self.sr, self.ch, self.subtype = sr, ch, subtype
        self._stream: Optional[sd.InputStream] = None
        self._q_write: "queue.Queue[np.ndarray]" = queue.Queue()
        self._q_feed: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop = threading.Event()
        self._writer: Optional[threading.Thread] = None
        self._sf: Optional[sf.SoundFile] = None
        self.wav: Optional[Path] = None

    def _cb(self, indata, frames, time, status):
        if status:
            # 必要ならログに出す: print(status)
            pass
        data = indata.copy()  # float32 (-1..1)
        self._q_write.put(data)
        self._q_feed.put(data)

    def start(self, wav_path: Path):
        if self._stream is not None:
            raise RuntimeError("already recording")
        self.wav = wav_path
        self._stop.clear()
        self._sf = sf.SoundFile(str(wav_path), mode="w", samplerate=self.sr, channels=self.ch, subtype=self.subtype)
        self._stream = sd.InputStream(samplerate=self.sr, channels=self.ch, dtype="float32", blocksize=1024, callback=self._cb)
        self._stream.start()

        def writer():
            while not self._stop.is_set():
                try:
                    data = self._q_write.get(timeout=0.2)
                except queue.Empty:
                    continue
                i16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
                self._sf.write(i16)
            # flush
            while True:
                try:
                    data = self._q_write.get_nowait()
                    i16 = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
                    self._sf.write(i16)
                except queue.Empty:
                    break

        self._writer = threading.Thread(target=writer, daemon=True)
        self._writer.start()

    def stop(self):
        if self._stream is None:
            return None
        self._stop.set()
        try:
            self._stream.stop(); self._stream.close()
        finally:
            self._stream = None
        if self._writer:
            self._writer.join(timeout=2)
        if self._sf:
            self._sf.close(); self._sf = None
        return self.wav


class Chunker:
    """3秒(+0.5秒の先頭オーバーラップを付けた)チャンクを生成してキューへ流す役"""
    def __init__(self, sr=SAMPLE_RATE, chunk_s=CHUNK_SECONDS, overlap_s=OVERLAP_SECONDS):
        self.sr = sr
        self.chunk_n = int(round(chunk_s * sr))
        self.overlap_n = int(round(overlap_s * sr))
        self._buf = np.zeros((0, 1), dtype=np.float32)
        self._emitted = 0  # non-overlap samples emitted
        self.out_q: "queue.Queue[Tuple[float,float,np.ndarray]]" = queue.Queue(maxsize=3)

    def feed(self, data: np.ndarray):
        if data.ndim == 2 and data.shape[1] > 1:
            data = data[:, :1]
        elif data.ndim == 1:
            data = data[:, None]
        self._buf = np.concatenate([self._buf, data], axis=0)
        while self._buf.shape[0] >= self.chunk_n:
            chunk = self._buf[: self.chunk_n]
            self._buf = self._buf[self.chunk_n :]
            prefix = self._tail()
            audio = np.concatenate([prefix, chunk], axis=0)[:, 0].astype(np.float32)
            s = self._emitted / float(self.sr)
            e = (self._emitted + self.chunk_n) / float(self.sr)
            self._emitted += self.chunk_n
            # backpressure: drop oldest if full
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
            self.out_q.put((s, e, audio))

    def flush(self):
        if self._buf.shape[0] > int(0.8 * self.sr):  # emit if ~>=0.8s remains
            chunk = self._buf; self._buf = np.zeros((0,1), dtype=np.float32)
            prefix = self._tail()
            audio = np.concatenate([prefix, chunk], axis=0)[:, 0].astype(np.float32)
            s = self._emitted / float(self.sr)
            e = (self._emitted + chunk.shape[0]) / float(self.sr)
            self._emitted += chunk.shape[0]
            if self.out_q.full():
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
            self.out_q.put((s, e, audio))

    def _tail(self) -> np.ndarray:
        tail = np.zeros((self.overlap_n, 1), dtype=np.float32)
        take = min(self.overlap_n, self._buf.shape[0])
        if take > 0:
            tail[-take:] = self._buf[:take]
        return tail


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("faster-whisper LIVE (manual send, 3s chunks)")
        self.geometry("1000x720")
        self.minsize(900, 600)

        self.output_dir = Path.cwd() / "live_sessions"; self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_var = tk.StringVar(value="Ready")
        self.last_chunk: Optional[Tuple[float, float, np.ndarray]] = None  # (s, e, audio)

        self.rec: Optional[LiveRecorder] = None
        self.chunker = Chunker()
        self.session_tag = now_tag()
        self.wav: Optional[Path] = None
        self.csv: Optional[Path] = None
        self.srt: Optional[Path] = None
        self.vtt: Optional[Path] = None
        self.seg_idx = 0

        self._consumer: Optional[threading.Thread] = None

        # ===== UI =====
        top = ttk.Frame(self, padding=10); top.pack(fill=tk.X)
        ttk.Label(top, text="Output:").pack(side=tk.LEFT)
        self.out_var = tk.StringVar(value=str(self.output_dir))
        ttk.Entry(top, textvariable=self.out_var, width=70).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse", command=self.choose_output).pack(side=tk.LEFT)
        ttk.Button(top, text="Open", command=self.open_output).pack(side=tk.LEFT, padx=(6,0))

        ctrl = ttk.Frame(self, padding=(10,0,10,10)); ctrl.pack(fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ Start LIVE", command=self.on_start)
        self.btn_stop = ttk.Button(ctrl, text="■ Stop", command=self.on_stop, state=tk.DISABLED)
        self.btn_send_chunk = ttk.Button(ctrl, text="📤 Send Chunk", command=self.on_send_chunk, state=tk.DISABLED)
        self.btn_send_file = ttk.Button(ctrl, text="📁 Send WAV…", command=self.on_send_file, state=tk.NORMAL)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=8)
        self.btn_send_chunk.pack(side=tk.LEFT, padx=8)
        self.btn_send_file.pack(side=tk.LEFT, padx=8)

        ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(12,6)).pack(fill=tk.X)

        frame_text = ttk.Frame(self, padding=(10,0,10,10)); frame_text.pack(fill=tk.BOTH, expand=True)
        self.text = tk.Text(frame_text, wrap=tk.WORD); self.text.pack(fill=tk.BOTH, expand=True)

    # ---------- UI handlers ----------
    def choose_output(self):
        d = filedialog.askdirectory(initialdir=str(self.output_dir))
        if d:
            self.output_dir = Path(d); self.out_var.set(str(self.output_dir))

    def open_output(self):
        try:
            os.startfile(str(self.output_dir))
        except Exception:
            messagebox.showinfo("Open", str(self.output_dir))

    def on_start(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            base = self.output_dir / f"live_{now_tag()}"
            self.wav, self.csv, self.srt, self.vtt = base.with_suffix(".wav"), base.with_suffix(".csv"), base.with_suffix(".srt"), base.with_suffix(".vtt")
            with open(self.csv, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(["file_id","segment_index","start_sec","end_sec","start_timecode","end_timecode","text"])
            self.srt.write_text("", encoding="utf-8"); self.vtt.write_text("WEBVTT\n\n", encoding="utf-8")

            self.rec = LiveRecorder(); self.chunker = Chunker(); self.seg_idx = 0
            self.rec.start(self.wav)
            self.status_var.set(f"LIVE… writing: {self.wav.name}")
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_stop.configure(state=tk.NORMAL)
            self.btn_send_chunk.configure(state=tk.NORMAL)

            # consumer thread: チャンクを内部に貯めつつ、最新だけ保持
            def consumer():
                assert self.rec is not None
                while self.btn_stop['state'] == tk.NORMAL:
                    try:
                        data = self.rec._q_feed.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self.chunker.feed(data)
                    # out_qに溜まっている分を全て取り出し、最後の1個をlast_chunkに保持
                    while not self.chunker.out_q.empty():
                        try:
                            self.last_chunk = self.chunker.out_q.get_nowait()
                        except queue.Empty:
                            break
                # 停止時に残りを1本だけflushして最新に
                self.chunker.flush()
                while not self.chunker.out_q.empty():
                    try:
                        self.last_chunk = self.chunker.out_q.get_nowait()
                    except queue.Empty:
                        break

            self._consumer = threading.Thread(target=consumer, daemon=True)
            self._consumer.start()

        except Exception as e:
            messagebox.showerror("Start", f"Failed to start: {e}")

    def on_send_chunk(self):
        """直近の3秒チャンクをサーバに手動送信"""
        if not self.last_chunk:
            messagebox.showinfo("Send Chunk", "送信できる音声チャンクがまだありません（少し話してから押してください）")
            return
        s, e, audio = self.last_chunk
        # 一時WAVに書き出してPOST
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            tmp_path = Path(tmpf.name)
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype=SUBTYPE)
            with open(tmp_path, "rb") as f:
                files = {"file": (tmp_path.name, f, "audio/wav")}
                r = requests.post(SERVER_URL, files=files, timeout=120)
            if r.ok:
                data = r.json()
                # "text" が空なら segments から合成
                txt = (data.get("text") or
                       " ".join(seg.get("text","") for seg in data.get("segments", []))).strip()
                if not txt:
                    txt = "(no speech detected)"
                self.append_outputs(s, e, txt)
            else:
                messagebox.showerror("Send Chunk", f"Server error: {r.status_code}\n{r.text}")
        except Exception as ex:
            messagebox.showerror("Send Chunk", f"送信失敗: {ex}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def on_send_file(self):
        """任意のWAVファイルを選択して送信（録音停止後や既存ファイルの検証用）"""
        path = filedialog.askopenfilename(
            title="Select WAV to send",
            filetypes=[("WAV Files","*.wav"), ("All Files","*.*")]
        )
        if not path:
            return
        wav_path = Path(path)
        # 長さを算出（CSV/SRT/VTT用）
        try:
            with sf.SoundFile(str(wav_path), mode="r") as sfh:
                frames = len(sfh); sr = sfh.samplerate
            s, e = 0.0, frames / float(sr)
        except Exception:
            s, e = 0.0, 0.0
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (wav_path.name, f, "audio/wav")}
                r = requests.post(SERVER_URL, files=files, timeout=300)
            if r.ok:
                data = r.json()
                txt = (data.get("text") or
                       " ".join(seg.get("text","") for seg in data.get("segments", []))).strip()
                if not txt:
                    txt = "(no speech detected)"
                self.append_outputs(s, e, txt)
            else:
                messagebox.showerror("Send WAV", f"Server error: {r.status_code}\n{r.text}")
        except Exception as ex:
            messagebox.showerror("Send WAV", f"送信失敗: {ex}")

    def append_outputs(self, s: float, e: float, txt: str):
        def _apply():
            self.seg_idx += 1
            self.text.insert(tk.END, txt + " "); self.text.see(tk.END)
            # CSV
            with open(self.csv, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([self.wav.name if self.wav else "live", self.seg_idx, f"{s:.3f}", f"{e:.3f}", fmt_tc(s), fmt_tc(e), txt])
            # SRT
            with open(self.srt, "a", encoding="utf-8") as f:
                f.write(f"{self.seg_idx}\n{fmt_tc(s)} --> {fmt_tc(e)}\n{txt}\n\n")
            # VTT
            with open(self.vtt, "a", encoding="utf-8") as f:
                f.write(f"{fmt_tc(s,'.')} --> {fmt_tc(e,'.')}\n{txt}\n\n")
            self.status_var.set(f"Segments: {self.seg_idx}")
        self.after(0, _apply)

    def on_stop(self):
        try:
            if self.rec:
                self.rec.stop()
            self.btn_start.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)
            self.btn_send_chunk.configure(state=tk.DISABLED)
            self.status_var.set("Stopped")
        except Exception as e:
            messagebox.showerror("Stop", f"Failed to stop: {e}")


if __name__ == "__main__":
    app = App()
    try:
        style = ttk.Style()
        style.theme_use("vista")
    except Exception:
        pass
    app.mainloop()
