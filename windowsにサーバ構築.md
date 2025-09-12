# 仮想環境作成
```bash
# 仮想環境作成
python -m venv venv
# 有効化
venv\Scripts\activate
```

# pipのインストール
```bash
pip install --upgrade pip

# FastAPI & サーバ
pip install fastapi uvicorn[standard]

# 音声処理ライブラリ
pip install faster-whisper

# オプション（日本語文字化け対策など）
pip install python-multipart jinja2
```

# 起動
```bash
uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

# テスト環境
```bash
http://127.0.0.1:8004/
```

# GPU版に修正
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```