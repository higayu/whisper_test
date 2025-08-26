import requests

url = "http://localhost:8000/transcribe"
with open(r"C:\Users\Higashiyama\Documents\_python\ohayou.wav", "rb") as f:
    files = {"file": ("ohayou.wav", f, "audio/wav")}
    r = requests.post(url, files=files)

print(r.json())
