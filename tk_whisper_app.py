import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from faster_whisper import WhisperModel
import threading

# モデルのロード
model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_file(file_path, text_widget):
    try:
        segments, info = model.transcribe(file_path, beam_size=5)
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, f"言語: {info.language}\n")
        for segment in segments:
            text_widget.insert(tk.END, f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n")
    except Exception as e:
        messagebox.showerror("エラー", str(e))

def open_file_and_transcribe(text_widget):
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac")]
    )
    if file_path:
        threading.Thread(target=transcribe_file, args=(file_path, text_widget), daemon=True).start()

# GUIセットアップ
root = tk.Tk()
root.title("Whisper Tkinter Transcriber")
root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(pady=10)

select_button = tk.Button(frame, text="音声ファイルを選択して文字起こし", command=lambda: open_file_and_transcribe(result_box))
select_button.pack()

result_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
result_box.pack(pady=10)

root.mainloop()
