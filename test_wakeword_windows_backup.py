import pyaudiowpatch as pyaudio # handles audio input/output from microphone
import numpy as np
from openwakeword.model import Model # wake-word detection model
import os
import wave
import whisper
import ollama

import imageio_ffmpeg
from piper import PiperVoice
import sounddevice as sd

from router import answer_question

# --- 0. Initialize Piper Neural TTS ---
MODEL_PATH = "en_US-amy-medium.onnx"
print("[BIMO Engine] Loading Piper neural voice model...")
voice = PiperVoice.load(MODEL_PATH)

def speak(text):
    print(f"[BIMO SPEAKING]: {text}")
    
    # Generate audio chunks using Piper
    audio_bytes = bytearray()
    for chunk in voice.synthesize(text):
        audio_bytes.extend(chunk.audio_int16_bytes)
    
    # Convert raw bytes to a 16-bit PCM numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Play through speakers using the model's sample rate
    sd.play(audio_np, samplerate=voice.config.sample_rate)
    sd.wait()

WAKE_WORD_MODEL = "./wakeword.onnx"
owwModel = Model(wakeword_models=[WAKE_WORD_MODEL])

# Transcribe Audio using Local Whisper Model
print("[BIMO Engine] Loading local Whisper model (tiny.en)...")
model = whisper.load_model("tiny.en")

# ---- Dynamic FFmpeg path setup ----
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# 2. Audio Capture Settings, here we configure microphone and tell python how to capture audio
AUDIO_FILE = "recorded_prompt.wav"
FORMAT = pyaudio.paInt16 # how each audio sample should be represented in memory (16-bit signed integer), these numbers describe the shape of the sound wave
CHANNELS = 1 # record one audio channel
RATE = 16000 # openWakeWord requiers audio sampled 16000 times per second
CHUNK = 1280 # it cuts microphone steam into 1280-sample frames (80ms chunks) for analysis
RECORD_SECONDS = 10

audio = pyaudio.PyAudio() # initialize the audio system
# open a stream to the microphone, this is where we will read audio input data from
mic_stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

try:
    while True:
        print("\n[BMO Engine] Wake word listener active! Say 'Hey BMO' to test...")

        # --- Read microphone stream - captures 80ms(1280 samples) of sound from mic, it's a raw binary audio data ---
        raw_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            
        # --- Convert raw binary audio buffer into numpy array, so we go from [10101001 01011010 00101001 ...] to [-1234 5678 -2345 ...] ---
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)

        # --- predict wake word ---
        prediction = owwModel.predict(audio_frame)
        for key,value in prediction.items():
            print(f"\r[BMO Engine] Wake word detection score: {value:.2f}", end="")

        # --- if detected -> record promt ---
        if prediction['wakeword'] > 0.8:
            print("\n[BMO Engine] Wake word detected! Recording your prompt...")
            recorded_audio = [] # here we will store the audio frames captured from the microphone

            # !!! add a delay befor start to record !!!

            # --- record audio for 10 seconds, each iteration(125) new microphone data ---
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                promt = mic_stream.read(CHUNK) # raw data, good for wave
                recorded_audio.append(promt)

            print("[BIMO Engine] Recording complete. Saving audio file...")

            # Save recorded audio buffer into a temporary .wav file
            wf = wave.open(AUDIO_FILE, 'wb')

            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)

            wf.writeframes(b''.join(recorded_audio)) # recorded audio is a list of byte strings, here we connect them into one byte stream
            wf.close()

            print("[BIMO Engine] Transcribing spoken audio...")
            # fp16=False suppresses the CPU warning and processes audio on CPU directly
            result = model.transcribe(AUDIO_FILE, fp16=False)

            prompt = result['text'].strip()

            if not prompt:
                print("[BIMO Engine] Didn't catch any speech.")
                continue

            print(f"[BIMO PROMPT]: {prompt}")

            # # send prompt to gemma 
            # response = ollama.chat(
            #     model = 'gemma:2b',
            #     messages = [
            #         {
            #             'role': 'system',
            #             'content': 'You are BIMO, a cheerful offline assistant. Keep responses under 2 sentences.'
            #         },
            #         {
            #             'role': 'user',
            #             'content': prompt
            #         }
            #     ]
            # )

            # answer = response['message']['content'].strip()

            # ----- HERE WE ADD THE ROUTER LOGIC -----
            # Send the transcript to the router
            answer = answer_question(prompt)
            
            print(f"[BIMO AI]: {answer}")

            # --- ANNOUNCE RESPONSE VIA TTS ---
            speak(answer)

            break

finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

    # Clean up local temp file
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)