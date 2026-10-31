import numpy as np
from openwakeword.model import Model # wake-word detection model
import os
import wave
import whisper
import ollama

import time 
import re

from piper import PiperVoice
import sounddevice as sd

from router import answer_question

# --- 0. Initialize Piper Neural TTS ---
MODEL_PATH = "en_US-amy-medium.onnx"
print("[BIMO Engine] Loading Piper neural voice model...")
voice = PiperVoice.load(MODEL_PATH)

def speak(text):
    import subprocess
    import tempfile
    import numpy as np

    print(f"[BIMO SPEAKING]: {text}")
    audio_bytes = bytearray()

    for chunk in voice.synthesize(text):
        audio_bytes.extend(chunk.audio_int16_bytes)

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        with wave.open(temp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            wf.writeframes(bytes(audio_bytes))

        subprocess.run(["pw-play", temp.name], check=True)

WAKE_WORD_MODEL = "./wakeword.onnx"
owwModel = Model(wakeword_model_paths=[WAKE_WORD_MODEL])

# Transcribe Audio using Local Whisper Model
print("[BIMO Engine] Loading local Whisper model (tiny.en)...")
model = whisper.load_model("tiny.en")

# 2. Audio Capture Settings, here we configure microphone and tell python how to capture audio
import subprocess

AUDIO_FILE = "recorded_prompt.wav"
RATE = 16000
CHUNK = 1280
MAX_RECORD_SECONDS = 10
SILENCE_SECONDS = 1.0
START_TIMEOUT_SECONDS = 5
MIC_TARGET = "85"

SILENCE_THRESHOLD = 500

mic_process = subprocess.Popen(
    [
        "pw-record",
        "--target", MIC_TARGET,
        "--rate", str(RATE),
        "--channels", "1",
        "--format", "s16",
        "--raw",
        "-"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=0
)

def read_exact(size):
    data = bytearray()
    while len(data) < size:
        part = mic_process.stdout.read(size - len(data))
        if not part:
            raise RuntimeError("Bose microphone stream stopped.")
        data.extend(part)
    return bytes(data)

# silence detection function
def record_until_silence():
    recorded_audio = []
    speech_started = False
    silence_duration = 0.0
    elapsed = 0.0
    speech_elapsed = 0.0

    print("[BIMO] Waiting for you to speak...")

    while True:
        raw_data = read_exact(CHUNK * 2)
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)

        rms = np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2))

        elapsed += CHUNK / RATE

        if speech_started:
            speech_elapsed += CHUNK / RATE

        if not speech_started:
            if rms >= SILENCE_THRESHOLD:
                speech_started = True
                print("[BIMO] Speech detected!")
                recorded_audio.append(raw_data)
            elif elapsed >= START_TIMEOUT_SECONDS:
                print("[BIMO] No speech detected. Returning to listening.")
                return None

        else:
            recorded_audio.append(raw_data)

            if speech_elapsed >= MAX_RECORD_SECONDS:
                print("[BIMO] Maximum recording time reached.")
                break

            if rms < SILENCE_THRESHOLD:
                silence_duration += CHUNK / RATE
            else:
                silence_duration = 0.0

            if silence_duration >= SILENCE_SECONDS:
                print("[BIMO] Silence detected. Stopping recording.")
                break

    if not speech_started:
        return None

    return recorded_audio

def clean_for_speech(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()

try:
    while True:
        print("\n[BIMO] Listening. Say 'Hey BMO'...")
        raw_data = read_exact(CHUNK * 2)
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)
        prediction = owwModel.predict(audio_frame)

        for key, value in prediction.items():
            print(f"\r[BIMO] Wake-word score: {value:.2f}", end="")

        if prediction.get("wakeword", 0) > 0.6:
            print("\n[BIMO] Wake word detected! Say your question...")
            record_start = time.perf_counter()

            recorded_audio = record_until_silence()

            record_time = time.perf_counter() - record_start
            print(f"[TIMING] Recording: {record_time:.2f} seconds")

            if recorded_audio is None:
                continue

            with wave.open(AUDIO_FILE, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(b"".join(recorded_audio))

            print("[BIMO] Transcribing...")
            transcribe_start = time.perf_counter()
            result = model.transcribe(AUDIO_FILE, fp16=False, condition_on_previous_text=False)
            prompt = result["text"].strip()
            transcribe_time = time.perf_counter() - transcribe_start
            print(f"[TIMING] Whisper transcription: {transcribe_time:.2f}s")

            if not prompt:
                print("[BIMO] Didn't catch any speech.")
                continue

            print(f"[BIMO PROMPT]: {prompt}")

            answer_start = time.perf_counter()
            answer_genereted = answer_question(prompt)
            answer = clean_for_speech(answer_genereted)
            answer_time = time.perf_counter() - answer_start


            print(f"[BIMO AI]: {answer}")
            print(f"[TIMING] AI answer: {answer_time:.2f} seconds")
            
            speak_start = time.perf_counter()
            speak(answer)
            speak_time = time.perf_counter() - speak_start

            print(f"[TIMING] Piper speech: {speak_time:.2f} seconds")
            print( 
                f"[TIMING] TOTAL: "
                f"{record_time + transcribe_time + answer_time + speak_time:.2f} seconds"
            )

            break

finally:
    mic_process.terminate()
    mic_process.wait()
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)
