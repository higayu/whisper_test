# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading, queue, time
from pathlib import Path
import datetime as dt

# ==== 設定 ====
CHUNK_SECONDS = 3.0
OVERLAP_SECONDS = 0.5
OUTDIR = Path.cwd() / "live_sessions"
OUTDIR.mkdir(exist_ok=True)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # エラーメッセージで案内する

# ==== ユーティリティ ====
def now_tag():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def list_input_devices():
    devs = sd.query_devices(); apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) >= 1:
            api = apis[d["hostapi"]]["name"] if 0 <= d["hostapi"] < len(apis) else "unknown"
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
            cand.append(int(round(d["default_sr"])))
        cand += [48000, 44100, 32000, 16000]
        # 重複除去
        seen = set(); sr_list = []
        for s in cand:
            s = int(s)
            if s not in seen:
                seen.add(s); sr_list.append(s)
        for sr in sr_list:
            ok, _ = try_open(d["index"], sr)
            if ok:
                return d["index"], sr, f"[{d['index']}] {d['name']} @ {sr} Hz"
    return None, None, "どのデバイス/サンプルレートでも開けませんでした"

# ==== チャンクャ ====
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

# ==== アプリ ====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Step2: Local Realtime Transcribe (no server)")
        self.geometry("940x600")

        # 状態
        self.devices = list_input_devices()
        self.device_index = None
        self.sr = None
        self.stream = None
        self.chunker = None
        self.run_ev = threading.Event()
        self.last_chunk = None
        self.q_jobs = queue.Queue()
        self._last_auto_sent_s = None

        # モデル
        if WhisperModel is None:
            messagebox.showerror("faster-whisper が未インストール",
                                 "pip install faster-whisper を実行してください。")
            self.destroy(); return
        self.model = WhisperModel("small", device="cpu", compute_type="int8")

        self._build_ui()
        self._start_worker()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10); frm.pack(fill=tk.X)
        ttk.Label(frm, text="Input device:").pack(side=tk.LEFT)
        self.combo = ttk.Combobox(frm, width=80, state="readonly",
                                  values=[d["label"] for d in self.devices])
        self.combo.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        if self.devices: self.combo.current(0)
        ttk.Button(frm, text="Auto-pick", command=self._auto_pick).pack(side=tk.LEFT, padx=6)
        ttk.Button(frm, text="Test", command=self._test_open).pack(side=tk.LEFT)

        ctrl = ttk.Frame(self, padding=(10, 4)); ctrl.pack(fill=tk.X)
        self.btn_start = ttk.Button(ctrl, text="▶ Start", command=self.start_live)
        self.btn_stop  = ttk.Button(ctrl, text="■ Stop", command=self.stop_live, state=tk.DISABLED)
        self.btn_send  = ttk.Button(ctrl, text="📤 Send last chunk", command=self.send_last, state=tk.DISABLED)
        self.autosend  = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(ctrl, text="Auto-send every chunk", variable=self.autosend)
        self.btn_start.pack(side=tk.LEFT); self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_send.pack(side=tk.LEFT, padx=6); chk.pack(side=tk.RIGHT)

        self.txt = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=18)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12, pady=(0,8))

    def _auto_pick(self):
        if not self.devices:
            messagebox.showerror("Error", "入力デバイスがありません")
            return
        dev, sr, why = auto_pick_device_and_sr(self.devices)
        if dev is None:
            messagebox.showerror("Auto-pick", f"失敗: {why}")
        else:
            self.device_index, self.sr = dev, sr
            self.status.set(f"Auto-picked {why}")

    def _test_open(self):
        idx = self.device_index
        if idx is None:
            # comboboxから仮選択
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

    # ==== 録音・チャンク ====
    def _cb(self, indata, frames, time_info, status_):
        if status_:
            print("PortAudio:", status_)
        self.chunker.feed(indata.copy())

    def start_live(self):
        if not self.devices:
            messagebox.showerror("Start", "入力デバイスがありません"); return
        # デバイス/SR未決なら自動選択
        if self.device_index is None or self.sr is None:
            dev, sr, _ = auto_pick_device_and_sr(self.devices)
            if dev is None:
                messagebox.showerror("Start", "デバイスを開けませんでした"); return
            self.device_index, self.sr = dev, sr

        self.chunker = Chunker(self.sr, CHUNK_SECONDS, OVERLAP_SECONDS)
        sd.default.device = (self.device_index, None)
        try:
            self.stream = sd.InputStream(device=self.device_index, samplerate=self.sr,
                                         channels=1, dtype="float32",
                                         blocksize=0, latency="high", callback=self._cb)
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
            # 生成済みチャンクをすべて取り出す
            while not self.chunker.out_q.empty():
                try:
                    s, e, audio = self.chunker.out_q.get_nowait()
                except queue.Empty:
                    break
                self.last_chunk = (s, e, audio)
                # Auto-send
                if self.autosend.get():
                    # 連投しすぎ防止：同じ開始時刻はスキップ
                    if self._last_auto_sent_s != s and self.q_jobs.qsize() < 3:
                        self.q_jobs.put(("transcribe_local", (s, e, audio, self.sr)))
                        self._last_auto_sent_s = s
            time.sleep(0.03)
        # stop時 flush
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

    # ==== 送信（ローカル推論） ====
    def _start_worker(self):
        def worker():
            while True:
                kind, payload = self.q_jobs.get()
                if kind == "transcribe_local":
                    s, e, audio, sr = payload
                    # 一時WAVに書いてから faster-whisper に渡す（SR差も吸収）
                    tmp = OUTDIR / f"tmp_{now_tag()}.wav"
                    try:
                        sf.write(tmp, audio, sr, subtype="PCM_16")
                        segs, info = self.model.transcribe(str(tmp), beam_size=5)
                        text = "".join(seg.text for seg in segs).strip() or "(no speech detected)"
                    except Exception as ex:
                        text = f"(error: {ex})"
                    finally:
                        try: tmp.unlink(missing_ok=True)
                        except Exception: pass
                    self._append_text(s, e, text)
        threading.Thread(target=worker, daemon=True, name="TranscribeWorker").start()

    def send_last(self):
        if not self.last_chunk:
            messagebox.showinfo("Send", "まだチャンクがありません"); return
        s, e, audio = self.last_chunk
        # キューが溢れていたら落とす
        if self.q_jobs.qsize() >= 5:
            messagebox.showinfo("Send", "処理中が多いためスキップしました"); return
        self.q_jobs.put(("transcribe_local", (s, e, audio, self.sr)))
        self.status.set(f"Queued chunk {s:.2f}-{e:.2f}s")

    def _append_text(self, s, e, text):
        def apply():
            self.txt.insert(tk.END, text + " ")
            self.txt.see(tk.END)
            self.status.set(f"chunk {s:.2f}–{e:.2f}s → appended")
        self.after(0, apply)

if __name__ == "__main__":
    App().mainloop()
