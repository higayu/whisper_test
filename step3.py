# -*- coding: utf-8 -*-
"""
Step3: Realtime transcription (client) – send 3s chunks to a Faster-Whisper server
- Non-blocking GUI (Tkinter): all I/O (recording, posting) done in threads
- Auto device & sample-rate picker (tries device default, 48k, 44.1k, 32k, 16k)
- Level meter, energy gate, and debug chunk saving
- Manual send of the last chunk + optional auto-send of every chunk

Server (example):
    pip install fastapi uvicorn faster-whisper python-multipart
    uvicorn Whisper_Server:app --reload --host 0.0.0.0 --port 8000

Expected server endpoint: POST {SERVER_URL}
  form-data: file=<wav>
  response JSON: {"text": str, "segments": [{"text": str}, ...]} (either is fine)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading, queue, time, requests
from pathlib import Path
import datetime as dt

# ====== Config ======
SERVER_URL_DEFAULT = "http://localhost:8000/transcribe"
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 0.5
OUTDIR = Path.cwd() / "live_sessions"
OUTDIR.mkdir(exist_ok=True)
SAVE_CHUNKS_DIR = OUTDIR / "chunks"
SAVE_CHUNKS_DIR.mkdir(exist_ok=True)

# Energy gates (tune for your mic): if both below, treat as silence
ENERGY_RMS_GATE = 1e-4
ENERGY_PEAK_GATE = 1e-3
# Force language (None = server decides). If your server supports it, you can send along.
FORCE_LANGUAGE = "ja"  # or None

# ====== Utils ======

def now_tag():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def list_input_devices():
    devs = sd.query_devices(); apis = sd.query_hostapis(); out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) >= 1:
            api = apis[d["hostapi"][0] if isinstance(d["hostapi"], (list, tuple)) else d["hostapi"]]["name"] \
                  if 0 <= d["hostapi"] < len(apis) else "unknown"
            out.append({
                "index": i,
                "name": d["name"],
                "label": f"[{i}] {d['name']} — {api} (in={d['max_input_channels']}, default_sr={d.get('default_samplerate')})",
                "default_sr": d.get("default_samplerate", None),
            })
    return out


def try_open(device_index, sr, seconds=0.15):
    try:
        s = sd.InputStream(device=device_index, samplerate=int(sr), channels=1, dtype="float32", blocksize=0)
        s.start(); time.sleep(seconds); s.stop(); s.close()
        return True, ""
    except Exception as e:
        return False, str(e)


def auto_pick_device_and_sr(devices):
    for d in devices:
        cand = []
        if d["default_sr"]:
            cand.append(int(round(d["default_sr"])) )
        cand += [48000, 44100, 32000, 16000]
        # dedup
        seen, sr_list = set(), []
        for s in cand:
            s = int(s)
            if s not in seen:
                seen.add(s); sr_list.append(s)
        for sr in sr_list:
            ok, _ = try_open(d["index"], sr)
            if ok:
                return d["index"], sr, f"[{d['index']}] {d['name']} @ {sr} Hz"
    return None, None, "どのデバイス/サンプルレートでも開けませんでした"


# ====== Chunker ======
class Chunker:
    def __init__(self, sr, chunk_s, overlap_s):
        self.sr = sr
        self.chunk_n = int(round(chunk_s * sr))
        self.overlap_n = int(round(overlap_s * sr))
        self.buf = np.zeros((0, 1), dtype=np.float32)
        self.emitted = 0
        self.out_q = queue.Queue(maxsize=8)

    def feed(self, data):
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim == 2 and data.shape[1] > 1:
            data = data[:, :1]
        self.buf = np.concatenate([self.buf, data], axis=0)
        while self.buf.shape[0] >= self.chunk_n:
            chunk = self.buf[: self.chunk_n]
            self.buf = self.buf[self.chunk_n :]
            prefix = self._tail()
            audio = np.concatenate([prefix, chunk], axis=0)[:, 0].astype(np.float32)
            s = self.emitted / float(self.sr)
            e = (self.emitted + self.chunk_n) / float(self.sr)
            self.emitted += self.chunk_n
            if self.out_q.full():
                try: self.out_q.get_nowait()
                except queue.Empty: pass
            self.out_q.put((s, e, audio))

    def flush(self):
        if self.buf.shape[0] > int(0.8 * self.sr):
            chunk = self.buf; self.buf = np.zeros((0, 1), dtype=np.float32)
            prefix = self._tail()
            audio = np.concatenate([prefix, chunk], axis=0)[:, 0].astype(np.float32)
            s = self.emitted / float(self.sr)
            e = (self.emitted + chunk.shape[0]) / float(self.sr)
            self.emitted += chunk.shape[0]
            if self.out_q.full():
                try: self.out_q.get_nowait()
                except queue.Empty: pass
            self.out_q.put((s, e, audio))

    def _tail(self):
        tail = np.zeros((self.overlap_n, 1), dtype=np.float32)
        take = min(self.overlap_n, self.buf.shape[0])
        if take > 0:
            tail[-take:] = self.buf[:take]
        return tail


# ====== App ======
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Step3: Realtime → Server POST (non-blocking)")
        self.geometry("980x640")

        # State
        self.devices = list_input_devices()
        self.device_index = None
        self.sr = None
        self.stream = None
        self.chunker = None
        self.run_ev = threading.Event()
        self.last_chunk = None
        self.q_jobs = queue.Queue()
        self._last_auto_sent_s = None
        self._last_peak = 0.0

        self._build_ui()
        self._start_worker()

    # --- UI ---
    def _build_ui(self):
        top = ttk.Frame(self, padding=10); top.pack(fill=tk.X)
        ttk.Label(top, text="Server URL:").pack(side=tk.LEFT)
        self.server_var = tk.StringVar(value=SERVER_URL_DEFAULT)
        ttk.Entry(top, textvariable=self.server_var, width=55).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Ping (silent)", command=self.ping_silent).pack(side=tk.LEFT)
        ttk.Button(top, text="Ping (sine)", command=self.ping_sine).pack(side=tk.LEFT, padx=6)

        dev = ttk.Frame(self, padding=(10, 4)); dev.pack(fill=tk.X)
        ttk.Label(dev, text="Input device:").pack(side=tk.LEFT)
        self.combo = ttk.Combobox(dev, width=80, state="readonly",
                                  values=[d["label"] for d in self.devices])
        self.combo.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        if self.devices: self.combo.current(0)
        ttk.Button(dev, text="Auto-pick", command=self._auto_pick).pack(side=tk.LEFT, padx=6)
        ttk.Button(dev, text="Test", command=self._test_open).pack(side=tk.LEFT)

        ctrl = ttk.Frame(self, padding=(10, 4)); ctrl.pack(fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ Start", command=self.start_live)
        self.btn_stop  = ttk.Button(ctrl, text="■ Stop", command=self.stop_live, state=tk.DISABLED)
        self.btn_send  = ttk.Button(ctrl, text="📤 Send last chunk", command=self.send_last, state=tk.DISABLED)
        self.autosend  = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(ctrl, text="Auto-send every chunk", variable=self.autosend)
        self.btn_start.pack(side=tk.LEFT); self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_send.pack(side=tk.LEFT, padx=6); chk.pack(side=tk.RIGHT)

        self.txt = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # level meter
        lf = ttk.Frame(self, padding=(10,0)); lf.pack(fill=tk.X)
        ttk.Label(lf, text="Input level:").pack(side=tk.LEFT)
        self.level_var = tk.DoubleVar(value=0.0)
        self.level_bar = ttk.Progressbar(lf, orient="horizontal", length=250, mode="determinate", maximum=1.0, variable=self.level_var)
        self.level_bar.pack(side=tk.LEFT, padx=6)
        self.after(100, self._tick_level)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12, pady=(0,8))

    # --- Device helpers ---
    def _auto_pick(self):
        if not self.devices:
            messagebox.showerror("Auto-pick", "入力デバイスがありません"); return
        dev, sr, why = auto_pick_device_and_sr(self.devices)
        if dev is None:
            messagebox.showerror("Auto-pick", f"失敗: {why}")
        else:
            self.device_index, self.sr = dev, sr
            self.status.set(f"Auto-picked {why}")

    def _test_open(self):
        idx = self.device_index
        if idx is None:
            i = self.combo.current()
            if i >= 0 and self.devices:
                idx = self.devices[i]["index"]
                sr  = int(round(self.devices[i]["default_sr"] or 16000))
            else:
                messagebox.showerror("Test", "デバイス未選択"); return
        else:
            sr = self.sr or int(round(sd.query_devices()[idx].get("default_samplerate", 16000)))
        ok, err = try_open(idx, sr)
        if ok: messagebox.showinfo("Test", f"Open OK (device={idx}, sr={sr})")
        else:  messagebox.showerror("Test", f"Open failed: {err}")

    # --- Recording / chunking ---
    def _cb(self, indata, frames, time_info, status_):
        if status_:
            print("PortAudio:", status_)
        peak = float(np.max(np.abs(indata))) if indata.size else 0.0
        self._last_peak = 0.8*self._last_peak + 0.2*peak
        self.chunker.feed(indata.copy())

    def start_live(self):
        if not self.devices:
            messagebox.showerror("Start", "入力デバイスがありません"); return
        if self.device_index is None or self.sr is None:
            dev, sr, _ = auto_pick_device_and_sr(self.devices)
            if dev is None:
                messagebox.showerror("Start", "デバイスを開けませんでした"); return
            self.device_index, self.sr = dev, sr
        self.chunker = Chunker(self.sr, CHUNK_SECONDS, OVERLAP_SECONDS)
        sd.default.device = (self.device_index, None)
        try:
            self.stream = sd.InputStream(device=self.device_index, samplerate=self.sr,
                                         channels=1, dtype="float32", blocksize=0,
                                         latency="high", callback=self._cb)
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Start", f"InputStream失敗: {e}"); return
        self.run_ev.set()
        threading.Thread(target=self._consumer, daemon=True).start()
        self.status.set(f"LIVE… device={self.device_index}, sr={self.sr}")
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_send.config(state=tk.NORMAL)

    def _consumer(self):
        while self.run_ev.is_set():
            while not self.chunker.out_q.empty():
                try:
                    s, e, audio = self.chunker.out_q.get_nowait()
                except queue.Empty:
                    break
                self.last_chunk = (s, e, audio)
                if self.autosend.get():
                    if self._last_auto_sent_s != s and self.q_jobs.qsize() < 4:
                        self.q_jobs.put(("transcribe_remote", (s, e, audio, self.sr)))
                        self._last_auto_sent_s = s
            time.sleep(0.03)
        # flush on stop
        self.chunker.flush()
        while not self.chunker.out_q.empty():
            try: self.last_chunk = self.chunker.out_q.get_nowait()
            except queue.Empty: break

    def stop_live(self):
        self.run_ev.clear()
        try:
            if self.stream: self.stream.stop(); self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.status.set("Stopped")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    # --- Worker (server POST) ---
    def _start_worker(self):
        def worker():
            while True:
                kind, payload = self.q_jobs.get()
                if kind == "transcribe_remote":
                    s, e, audio, sr = payload
                    # energy gate
                    if audio.size == 0:
                        self._append_text(s, e, "(empty chunk)"); continue
                    rms  = float(np.sqrt(np.mean(audio**2)))
                    peak = float(np.max(np.abs(audio)))
                    # save for debug
                    fname = SAVE_CHUNKS_DIR / f"chunk_{now_tag()}_{s:.2f}_{e:.2f}_r{rms:.6f}_p{peak:.6f}.wav"
                    try:
                        sf.write(fname, audio, sr, subtype="PCM_16")
                    except Exception:
                        pass
                    if rms < ENERGY_RMS_GATE and peak < ENERGY_PEAK_GATE:
                        self._append_text(s, e, f"(silence: rms={rms:.2e}, peak={peak:.2e})"); continue

                    url = self.server_var.get().strip() or SERVER_URL_DEFAULT
                    # send file
                    try:
                        t0 = time.perf_counter()
                        with open(fname, "rb") as f:
                            files = {"file": (fname.name, f, "audio/wav")}
                            data = {}
                            if FORCE_LANGUAGE:
                                # only if your server supports; otherwise harmless
                                data["language"] = FORCE_LANGUAGE
                            r = requests.post(url, files=files, data=data, timeout=60)
                        ms = (time.perf_counter() - t0) * 1000.0
                        if r.ok:
                            j = r.json()
                            text = (j.get("text") or " ".join(seg.get("text", "") for seg in j.get("segments", []))).strip()
                            text = text if text else "(no speech detected)"
                            self._append_text(s, e, f"{text}")
                            self.status.set(f"POST {int(ms)} ms OK")
                        else:
                            self._append_text(s, e, f"(server {r.status_code})")
                            self.status.set(f"HTTP {r.status_code}")
                    except Exception as ex:
                        self._append_text(s, e, f"(error: {ex})")
        threading.Thread(target=worker, daemon=True, name="PostWorker").start()

    # --- Actions ---
    def send_last(self):
        if not self.last_chunk:
            messagebox.showinfo("Send", "まだチャンクがありません"); return
        s, e, audio = self.last_chunk
        if self.q_jobs.qsize() >= 6:
            messagebox.showinfo("Send", "処理中が多いためスキップしました"); return
        self.q_jobs.put(("transcribe_remote", (s, e, audio, self.sr)))
        self.status.set(f"Queued chunk {s:.2f}-{e:.2f}s")

    def _append_text(self, s, e, text):
        def apply():
            self.txt.insert(tk.END, text + " ")
            self.txt.see(tk.END)
        self.after(0, apply)

    # --- Level meter tick ---
    def _tick_level(self):
        self.level_var.set(min(1.0, self._last_peak * 4.0))
        self.after(100, self._tick_level)

    # --- Ping helpers ---
    def _post_wav_bytes(self, wav_path):
        url = self.server_var.get().strip() or SERVER_URL_DEFAULT
        with open(wav_path, "rb") as f:
            files = {"file": (Path(wav_path).name, f, "audio/wav")}
            data = {"language": FORCE_LANGUAGE} if FORCE_LANGUAGE else None
            t0 = time.perf_counter()
            r = requests.post(url, files=files, data=data, timeout=20)
            ms = (time.perf_counter() - t0) * 1000.0
        return r, ms

    def ping_silent(self):
        try:
            sr = 16000; dur = 0.5
            x = np.zeros(int(sr*dur), dtype=np.float32)
            tmp = OUTDIR / f"ping_silent_{now_tag()}.wav"
            sf.write(tmp, x, sr, subtype="PCM_16")
            r, ms = self._post_wav_bytes(tmp)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            if r.ok:
                j = r.json(); text = (j.get("text") or " ".join(seg.get("text", "") for seg in j.get("segments", []))).strip()
                messagebox.showinfo("Ping", f"OK {ms:.1f} ms\ntext='{text[:60]}'")
            else:
                messagebox.showerror("Ping", f"HTTP {r.status_code}\n{r.text[:200]}")
        except Exception as ex:
            messagebox.showerror("Ping", f"Ping failed: {ex}")

    def ping_sine(self):
        try:
            sr = 16000; dur = 1.0; t = np.arange(int(sr*dur))/sr
            x = 0.6*np.sin(2*np.pi*440*t).astype(np.float32)
            tmp = OUTDIR / f"ping_sine_{now_tag()}.wav"
            sf.write(tmp, x, sr, subtype="PCM_16")
            r, ms = self._post_wav_bytes(tmp)
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
            if r.ok:
                j = r.json(); text = (j.get("text") or " ".join(seg.get("text", "") for seg in j.get("segments", []))).strip()
                messagebox.showinfo("Ping", f"OK {ms:.1f} ms\ntext='{text[:60]}'")
            else:
                messagebox.showerror("Ping", f"HTTP {r.status_code}\n{r.text[:200]}")
        except Exception as ex:
            messagebox.showerror("Ping", f"Ping failed: {ex}")


if __name__ == "__main__":
    App().mainloop()
