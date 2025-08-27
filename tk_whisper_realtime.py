# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from pathlib import Path
import datetime as dt

OUTDIR = Path.cwd() / "live_sessions"
OUTDIR.mkdir(exist_ok=True)

def now_tag():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def list_input_devices():
    devs = sd.query_devices()
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) >= 1:
            api = apis[d["hostapi"]]["name"] if 0 <= d["hostapi"] < len(apis) else "unknown"
            out.append({
                "index": i,
                "label": f"[{i}] {d['name']} — {api} (in={d['max_input_channels']}, default_sr={d.get('default_samplerate')})",
                "name": d["name"],
                "default_sr": d.get("default_samplerate", None),
            })
    return out

def try_open(device_index, sr, seconds=0.2):
    """デバイスを指定SRで開けるか軽く試す（成功なら True）"""
    try:
        stream = sd.InputStream(device=device_index, samplerate=int(sr), channels=1, dtype="float32", blocksize=0)
        stream.start()
        time.sleep(seconds)
        stream.stop()
        stream.close()
        return True, ""
    except Exception as e:
        return False, str(e)

def auto_pick_device_and_sr(devices):
    """最初に開けた (device_index, sr, reason) を返す。見つからなければ (None, None, msg)"""
    for d in devices:
        cand = []
        if d["default_sr"]:
            cand.append(int(round(d["default_sr"])))
        # よく使うサンプルレートを追加
        cand += [48000, 44100, 32000, 16000]
        # 重複除去＆整数化
        sr_list = []
        seen = set()
        for s in cand:
            s = int(s)
            if s not in seen:
                seen.add(s)
                sr_list.append(s)
        for sr in sr_list:
            ok, err = try_open(d["index"], sr)
            if ok:
                return d["index"], sr, f"OK: [{d['index']}] {d['name']} @ {sr} Hz"
    return None, None, "どのデバイス/サンプルレートでも開けませんでした"

def record_5s(device_index, sr):
    """指定のデバイス/SRで5秒録音してWAV保存"""
    sd.default.device = (device_index, None)
    buf = []
    def cb(indata, frames, time_info, status_):
        if status_:
            print("Status:", status_)
        buf.append(indata.copy())

    with sd.InputStream(device=device_index, samplerate=sr, channels=1, dtype="float32", blocksize=0, callback=cb):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 5.0:
            root.update()
            time.sleep(0.01)

    if not buf:
        raise RuntimeError("音声が取得できませんでした")

    data = np.concatenate(buf, axis=0)[:, 0]
    out = OUTDIR / f"rec_{now_tag()}.wav"
    sf.write(out, data, sr, subtype="PCM_16")
    secs = len(data) / sr
    return out, secs

# ===== GUI =====
root = tk.Tk()
root.title("Step1: Auto-select & Record 5s")
root.geometry("780x220")

frame = ttk.Frame(root, padding=10); frame.pack(fill=tk.BOTH, expand=True)

# 手動用の一覧（念のため残す）
ttk.Label(frame, text="Input Device (manual):").grid(row=0, column=0, sticky="w")
devices = list_input_devices()
combo = ttk.Combobox(frame, width=90, state="readonly", values=[d["label"] for d in devices])
combo.grid(row=0, column=1, sticky="ew", columnspan=2)
if devices: combo.current(0)
frame.columnconfigure(1, weight=1)

def on_manual_record():
    i = combo.current()
    if i < 0 or not devices:
        messagebox.showerror("Error", "デバイスを選択してください")
        return
    dev = devices[i]
    sr = int(round(dev["default_sr"] or 16000))
    status.set(f"Manual: open [{dev['index']}] {dev['name']} @ {sr} Hz → 録音中…")
    root.update_idletasks()
    try:
        out, secs = record_5s(dev["index"], sr)
        status.set(f"Saved {out.name} ({secs:.2f}s @ {sr} Hz)")
        messagebox.showinfo("Done", f"Saved:\n{out}\n{secs:.2f}s @ {sr} Hz")
    except Exception as e:
        status.set("Ready")
        messagebox.showerror("Error", f"録音失敗: {e}")

def on_auto_record():
    status.set("Auto: 入力デバイス/サンプルレートを探索中…")
    root.update_idletasks()
    if not devices:
        messagebox.showerror("Error", "入力デバイスが見つかりませんでした")
        status.set("Ready"); return
    dev_index, sr, reason = auto_pick_device_and_sr(devices)
    if dev_index is None:
        status.set("Ready")
        messagebox.showerror("Error", f"オープンに失敗: {reason}")
        return
    # 見つかった組合せで録音
    status.set(f"{reason} → 5秒録音中…")
    root.update_idletasks()
    try:
        out, secs = record_5s(dev_index, sr)
        status.set(f"Saved {out.name} ({secs:.2f}s @ {sr} Hz)")
        messagebox.showinfo("Done", f"Auto selected:\nDevice #{dev_index}, {sr} Hz\n\nSaved:\n{out}\n{secs:.2f}s")
    except Exception as e:
        status.set("Ready")
        messagebox.showerror("Error", f"録音失敗: {e}")

ttk.Button(frame, text="🎙️ Auto 5秒録音 → WAV保存", command=on_auto_record).grid(row=1, column=0, columnspan=3, pady=(10, 6), sticky="ew")
ttk.Button(frame, text="🎛️ Manual 5秒録音 → WAV保存", command=on_manual_record).grid(row=2, column=0, columnspan=3, pady=6, sticky="ew")

status = tk.StringVar(value="Ready")
ttk.Label(frame, textvariable=status).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6,0))

root.mainloop()
