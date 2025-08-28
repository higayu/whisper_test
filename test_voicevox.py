import ctypes
from pathlib import Path

# VOICEVOX Core ライブラリのロード
lib_path = "/opt/voicevox_setup/voicevox_core/c_api/lib/libvoicevox_core.so"
vvx = ctypes.cdll.LoadLibrary(lib_path)

# エンジン初期化 (辞書などを読み込み)
vvx.voicevox_initialize.argtypes = []
vvx.voicevox_initialize.restype = ctypes.c_int
ret = vvx.voicevox_initialize()
print("initialize:", ret)

# 音声合成 (speaker=1 の場合: 四国めたん)
vvx.voicevox_tts.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
vvx.voicevox_tts.restype = ctypes.c_int

output_wav_ptr = ctypes.c_void_p()
output_size = ctypes.c_size_t()
text = "おはようございます".encode("utf-8")
ret = vvx.voicevox_tts(text, 1, ctypes.byref(output_wav_ptr), ctypes.byref(output_size))
print("tts:", ret, "size:", output_size.value)

# wav データをファイル保存
buf = (ctypes.c_char * output_size.value).from_address(output_wav_ptr.value)
Path("out.wav").write_bytes(buf)

# メモリ解放
vvx.voicevox_free_waveform(output_wav_ptr)
