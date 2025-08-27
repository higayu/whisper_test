import requests

#url = "http://localhost:8081/transcribe"
url = "http://192.168.1.221:8081/transcribe"
with open(r"ohayou.wav", "rb") as f:
    files = {"file": ("ohayou.wav", f, "audio/wav")}
    r = requests.post(url, files=files)

print(r.json())
