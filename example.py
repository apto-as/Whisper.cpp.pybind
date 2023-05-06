from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import pywhisper

if __name__ == "__main__":
    # Load whisper
    assert pywhisper.init(model_path="../whisper.cpp/models/ggml-medium.bin"), (
        "Failed to initialize pywhisper. Check the logs above for more details."
    )
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone(sample_rate=16_000) as source:
            print("なにか話してください")
            audio = recognizer.listen(source)

        print("音声処理中 ...")
        # 「音声データをWhisperの入力形式に変換」参照
        wav_bytes = audio.get_wav_data()
        wav_stream = BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_fp32 = audio_array.astype(np.float32)

        transcription = pywhisper.transcribe(audio_fp32, "ja")
        print(transcription[0][2])