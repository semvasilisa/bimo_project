import os
import re
import select
import subprocess
import tempfile
import time
import wave

import numpy as np

from openwakeword.model import Model  # wake-word detection model
from piper import PiperVoice

from router import answer_question, ROUTER_FAILURE_SENTENCE
from face_display import FaceDisplay

try:
    face = FaceDisplay()
    face.start()          # starts in "warmup"
except Exception as e:     # a broken screen must not kill BIMO
    print(f"[FACE] Disabled: {e}")
    face = None

def set_face(state):
    if face is not None:
        face.set_state(state)

from touch_input import TouchInput

try:
    touch = TouchInput()
    touch.start()
except Exception as e:
    print(f"[TOUCH] Disabled: {e}")
    touch = None

import whisper

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
AUDIO_FILE = "recorded_prompt.wav"
RATE = 16000
CHUNK = 1280                  # 80 ms of audio at 16 kHz
WAKE_WORD_THRESHOLD = 0.6

MAX_RECORD_SECONDS = 10       # max recording duration after speech starts
SILENCE_SECONDS = 1.0         # stop recording after this much silence
START_TIMEOUT_SECONDS = 5     # give up if nobody starts speaking
SILENCE_THRESHOLD = 500       # RMS level that separates speech from silence

MIC_TARGET = "bluez_input.78:2B:64:DD:D1:56"
MIC_READ_TIMEOUT = 5.0        # no audio for this long -> the mic is considered dead
MIC_START_RETRIES = 3         # how many times we try to (re)start the microphone
MIC_RETRY_DELAY = 2.0         # pause between start attempts
WARMUP_SECONDS = 1.5          # audio thrown away after each mic start (echo cooldown)

FALLBACK_SENTENCE = "I can't reach my brain right now, please try again in a moment."

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_PATH = "en_US-amy-medium.onnx"
print("[BIMO Engine] Loading Piper neural voice model...")
voice = PiperVoice.load(MODEL_PATH)

WAKE_WORD_MODEL = "./wakeword.onnx"
owwModel = Model(wakeword_model_paths=[WAKE_WORD_MODEL])

print("[BIMO Engine] Loading local Whisper model (tiny.en)...")
whisper_model = whisper.load_model("tiny.en")
set_face("idle") # state: IDLE

# ---------------------------------------------------------------------------
# Speech output
# ---------------------------------------------------------------------------
def speak(text, face_state="speaking"):
    print(f"[BIMO SPEAKING]: {text}")

    # measure Piper synthesis
    synthesis_start = time.perf_counter()

    audio_bytes = bytearray()
    total_samples = 0

    for chunk in voice.synthesize(text):
        audio_bytes.extend(chunk.audio_int16_bytes)
        total_samples += len(chunk.audio_int16_bytes) // 2

    synthesis_time = time.perf_counter() - synthesis_start

    # calculate the real length of the generated audio
    sample_rate = voice.config.sample_rate
    audio_duration = total_samples / sample_rate

    print(f"[PIPER] Synthesis: {synthesis_time:.2f}s")
    print(f"[PIPER] Generated audio length: {audio_duration:.2f}s")

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        with wave.open(temp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(audio_bytes))

        set_face(face_state) # state: SPEAKING
        # measure the real playback time
        playback_start = time.perf_counter()
        subprocess.run(["pw-play", temp.name], check=True)
        playback_time = time.perf_counter() - playback_start

    print(f"[PIPER] Playback: {playback_time:.2f}s")
    print(f"[PIPER] Total speak time: {synthesis_time + playback_time:.2f}s")


def say_fallback():
    """Speak the fixed sentence; never let this crash the loop."""
    try:
        speak(FALLBACK_SENTENCE, face_state="error")
    except Exception as e:
        print(f"[BIMO] Could not speak the fallback sentence: {e!r}")


def clean_for_speech(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Microphone (pw-record is started and stopped around every turn)
# ---------------------------------------------------------------------------
class MicrophoneError(Exception):
    """The microphone stream is missing, stopped or silent for too long."""


mic_process = None


def stop_microphone():
    """Stop pw-record. Safe to call when it is not running."""
    global mic_process

    if mic_process is not None:
        if mic_process.poll() is None:
            mic_process.terminate()
            try:
                mic_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                mic_process.kill()
                mic_process.wait()
        mic_process = None


def read_exact(size, timeout=MIC_READ_TIMEOUT):
    """Read exactly `size` bytes from pw-record, but never wait forever."""
    if mic_process is None or mic_process.poll() is not None:
        raise MicrophoneError("Microphone process is not running.")

    data = bytearray()
    deadline = time.monotonic() + timeout

    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MicrophoneError(f"No microphone audio for {timeout:.0f} s.")

        ready, _, _ = select.select([mic_process.stdout], [], [], remaining)
        if not ready:
            raise MicrophoneError(f"No microphone audio for {timeout:.0f} s.")

        part = mic_process.stdout.read(size - len(data))
        if not part:
            raise MicrophoneError("Microphone stream stopped.")
        data.extend(part)

    return bytes(data)


def rms_of(raw):
    frame = np.frombuffer(raw, dtype=np.int16)
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def discard_audio(seconds):
    """Read and throw away `seconds` of audio. Returns the RMS of each chunk."""
    levels = []
    for _ in range(int(seconds * RATE / CHUNK)):
        levels.append(rms_of(read_exact(CHUNK * 2)))
    return levels


def start_microphone():
    """Start pw-record and wait until real audio arrives (with retries)."""
    global mic_process

    for attempt in range(1, MIC_START_RETRIES + 1):
        stop_microphone()
        started = time.perf_counter()

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

        try:
            # first chunk: how long does the Bluetooth link need to deliver audio?
            levels = [rms_of(read_exact(CHUNK * 2))]
            first_audio = time.perf_counter() - started

            # throw away the next moments: stale data and the echo of our own voice
            levels += discard_audio(WARMUP_SECONDS)

        except MicrophoneError as e:
            print(f"[MIC] Start attempt {attempt}/{MIC_START_RETRIES} failed: {e}")
            stop_microphone()
            time.sleep(MIC_RETRY_DELAY)
            continue

        print(
            f"[MIC] Ready. First audio after {first_audio:.2f}s | "
            f"noise RMS min/avg/max: {min(levels):.0f}/"
            f"{sum(levels) / len(levels):.0f}/{max(levels):.0f}"
        )
        if max(levels) == 0:
            print("[MIC] WARNING: the stream delivers only zeros (silent mic).")
        return

    raise MicrophoneError("Could not start the microphone.")


# ---------------------------------------------------------------------------
# Listening
# ---------------------------------------------------------------------------
def wait_for_wake_word():
    print("\n[BIMO] Listening. Say 'Hey BMO'...")

    if touch is not None: 
        touch.clear()

    while True:
        raw_data = read_exact(CHUNK * 2)
        audio_frame = np.frombuffer(raw_data, dtype=np.int16)
        prediction = owwModel.predict(audio_frame)

        if touch is not None and touch.was_tapped():
            print("\n[BIMO] Touch detected! Say your question...")
            return

        score = prediction.get("wakeword", 0)
        print(f"\r[BIMO] Wake-word score: {score:.2f}", end="")

        if score > WAKE_WORD_THRESHOLD:
            print("\n[BIMO] Wake word detected! Say your question...")
            return


def record_until_silence():
    recorded_audio = []
    speech_started = False
    silence_duration = 0.0
    elapsed = 0.0
    speech_elapsed = 0.0

    print("[BIMO] Waiting for you to speak...")

    while True:
        raw_data = read_exact(CHUNK * 2)
        rms = rms_of(raw_data)

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

    return recorded_audio


# ---------------------------------------------------------------------------
# One full turn: record -> Whisper -> answer -> speak
# ---------------------------------------------------------------------------
def handle_turn():
    set_face("listening") # state: LISTENING
    record_start = time.perf_counter()
    recorded_audio = record_until_silence()
    record_time = time.perf_counter() - record_start
    print(f"[TIMING] Recording: {record_time:.2f} seconds")

    # The microphone is switched OFF here and stays off until the answer is spoken.
    # No stale audio piles up, and BIMO cannot hear its own voice.
    stop_microphone()

    if recorded_audio is None:
        return

    with wave.open(AUDIO_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(recorded_audio))

    set_face("thinking") # state: THINKING
    print("[BIMO] Transcribing...")
    transcribe_start = time.perf_counter()
    result = whisper_model.transcribe(
        AUDIO_FILE, fp16=False, condition_on_previous_text=False
    )
    prompt = result["text"].strip()
    transcribe_time = time.perf_counter() - transcribe_start
    print(f"[TIMING] Whisper transcription: {transcribe_time:.2f}s")

    if not prompt:
        print("[BIMO] Didn't catch any speech.")
        return

    print(f"[BIMO PROMPT]: {prompt}")

    answer_start = time.perf_counter()

    raw_answer = answer_question(prompt)
    answer = clean_for_speech(raw_answer)

    answer_time = time.perf_counter() - answer_start

    print(f"[BIMO AI]: {answer}")
    print(f"[TIMING] AI answer: {answer_time:.2f} seconds")

    # state: SPEAKING
    speak_start = time.perf_counter()

    face_state = "error" if raw_answer == ROUTER_FAILURE_SENTENCE else "speaking"
    speak(answer, face_state=face_state)

    speak_time = time.perf_counter() - speak_start

    print(f"[TIMING] Piper speech: {speak_time:.2f} seconds")
    print(
        f"[TIMING] TOTAL: "
        f"{record_time + transcribe_time + answer_time + speak_time:.2f} seconds"
    )


def resume_listening():
    """Start the microphone again and forget the old wake-word audio."""
    restart_start = time.perf_counter()
    start_microphone()
    owwModel.reset()
    print(f"[TEST] Microphone restart took {time.perf_counter() - restart_start:.2f} s")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    turns = 0

    try:
        start_microphone()

        while True:
            wait_for_wake_word()
            turns += 1
            print(f"[TEST] ===== Turn {turns} =====")

            try:
                handle_turn()
            except MicrophoneError:
                raise
            except Exception as e:
                # Whisper, the network, Piper or pw-play failed: stay alive
                print(f"[BIMO] Turn failed: {e!r}")
                say_fallback()

            set_face("idle") # state: IDLE
            resume_listening()

    except MicrophoneError as e:
        print(f"\n[BIMO MIC] ERROR: {e} Stopping BIMO.")
    except KeyboardInterrupt:
        print("\n[BIMO] Stopped by user.")
    finally:
        stop_microphone()
        if os.path.exists(AUDIO_FILE):
            os.remove(AUDIO_FILE)
        print(f"[TEST] Finished after {turns} turns.")

        if face is not None:
            face.stop() # clear the screen

        if touch is not None: 
            touch.stop()


if __name__ == "__main__":
    main()