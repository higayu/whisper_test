# Whisper + Tkinter ローカル利用手順

この手順では、OpenAI Whisper をローカル PC 上で Python + Tkinter GUI から直接使えるようにします。
Apache や Flask は不要で、完全にローカル完結型です。

| 項目     | OpenAI公式 `whisper`           | `faster_whisper`              |
| ------ | ---------------------------- | ----------------------------- |
| 実行速度   | 普通                           | **最大 4〜6倍高速**                 |
| メモリ使用量 | 多め                           | 少なめ（量子化も可能）                   |
| GPU対応  | あり                           | **GPU最適化（ONNX, CTranslate2）** |
| インストール | `pip install openai-whisper` | `pip install faster-whisper`  |

```bash
pip install python-multipart
```

### 実行する時のコマンド
```bash
uvicorn Whisper_Server:app --host 0.0.0.0 --port 8000

& C:/Users/Higashiyama/Documents/_python/whisper_test/venv/Scripts/Activate.ps1
```

---

## 1. 前提
- Python 3.9 以上
- ffmpeg がインストールされていること（音声形式変換用）
  - Windows: https://ffmpeg.org/download.html から取得し、PATHを通す
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

---

## 2. 仮想環境の作成と有効化
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

---

## 3. 必要パッケージのインストール
```bash
pip install faster-whisper tkinterdnd2
```

`faster-whisper` は高速な Whisper 推論ライブラリです。

---

## 4. Tkinter Whisper アプリの実行
`sandbox/tk_whisper_app.py` のようなサンプルを作成し、以下のように実行します。

```bash
python tk_whisper_app.py
```

GUIが立ち上がり、音声ファイルを選択して文字起こしできます。

---

## 5. 使い方
1. 「音声ファイルを選択」ボタンから mp3/wav/m4a などの音声を選ぶ
2. 「文字起こし開始」を押す
3. 下のテキストボックスに結果が表示される

---

## 6. 注意
- 初回実行時はモデル（例: small）が自動ダウンロードされます。
- モデルサイズは `WhisperModel("small")` の部分で変更可能（tiny, base, small, medium, large-v2）
- 長時間ファイルは変換に時間がかかります。

