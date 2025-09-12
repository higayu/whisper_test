# Faster-Whisper セットアップ手順記録

## 環境
- OS: Windows
- GPU: NVIDIA GeForce GTX 1650 (VRAM 4GB)
- CUDA Toolkit: 12.2
- cuDNN: 9.12
- Python: 3.11
- ライブラリ: faster-whisper, onnxruntime-gpu, ctranslate2

---

## 手順の記録

### 1. faster-whisper のインストール
```powershell
pip install faster-whisper
```

### 2. エラー: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'faster_whisper'
```
→ `uvicorn` が別の Python を使っていた可能性あり。仮想環境の python/uvicorn を使うように修正。

---

### 3. エラー: cudnn_ops64_9.dll が見つからない
```
Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!
```
- CUDA Toolkit 12.2 には cuDNN が含まれていない。
- NVIDIA サイトから cuDNN 9.x を入手し、
  - `cudnn_ops64_9.dll` を含む DLL を
    ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin
    ```
    にコピー。
- PATH 環境変数に cuDNN フォルダを追加。

---

### 4. GPU が利用可能か確認
```powershell
nvidia-smi
```
- `python.exe` プロセスが GPU を使用していることを確認。

---

### 5. 動作確認 (日本語指定)
```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float32")
segments, info = model.transcribe("ohayou.wav", language="ja")

print("言語:", info.language)
print("確率:", info.language_probability)
for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
```

- 出力例:
```
言語: ja
確率: 1
[0.00s -> 5.36s] おはようございます 今日もお仕事頑張るのだ
```

---

### 6. CPU と GPU の性能比較
計測コード:
```python
import time
from faster_whisper import WhisperModel

audio_file = "ohayou.wav"

for device in ["cpu", "cuda"]:
    model = WhisperModel("medium", device=device, compute_type="float32")
    start = time.time()
    segments, info = model.transcribe(audio_file, language="ja")
    elapsed = time.time() - start
    print(f"{device} 実行時間: {elapsed:.2f} 秒")
```

結果:
- CPU: 約 0.03 秒
- GPU: 約 0.03 秒 (短い音声なので差は出ない)

---

## まとめ
- `faster-whisper` を GPU で利用するには CUDA + cuDNN の正しい DLL 配置が必要。
- GTX 1650 では `compute_type="float32"` を指定すると安定して動作。
- 短い音声では CPU と GPU の速度差は小さいが、長い音声では GPU の方が有利。
