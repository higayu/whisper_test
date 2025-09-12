from faster_whisper import WhisperModel

# モデルをロード（GTX 1650なら medium が無難）
model = WhisperModel("medium", device="cuda", compute_type="float32")

# ohayou.wav を日本語指定で文字起こし
segments, info = model.transcribe("ohayou.wav", language="ja")

print("言語指定: 日本語 (ja)")
print("モデルが推定した言語:", info.language)
print("推定確率:", info.language_probability)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
