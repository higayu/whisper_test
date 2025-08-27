from pydub import AudioSegment

mp3_path = r"ohayou.mp3"
wav_path = r"ohayou.wav"

sound = AudioSegment.from_mp3(mp3_path)
sound.export(wav_path, format="wav")
print("変換完了:", wav_path)
