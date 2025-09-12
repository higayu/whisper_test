import time
from faster_whisper import WhisperModel

audio_file = "ohayou.wav"

for device in ["cpu", "cuda"]:
    print("="*40)
    print(f"--- {device.upper()} 実行 ---")

    # モデルをロード（mediumが無難、tiny/smallに変えてもOK）
    model = WhisperModel("medium", device=device ,compute_type="float32")

    # 計測開始
    start = time.time()

    segments, info = model.transcribe(audio_file, language="ja")

    elapsed = time.time() - start

    print(f"{device.upper()} での処理時間: {elapsed:.2f} 秒")
    print(f"推定言語: {info.language} (確率 {info.language_probability:.2f})")

    for seg in segments:
        print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
