import queue, threading, time, tempfile, os, sys
import numpy as np
import sounddevice as sd
import webrtcvad
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_TIMEOUT_MS = 700

vad = webrtcvad.Vad(2)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

audio_queue = queue.Queue()
result_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠", status)
    audio_queue.put(bytes(indata))

def process_audio():
    voiced_frames = []
    silence_ms = 0

    while True:
        frame = audio_queue.get()

        if vad.is_speech(frame, SAMPLE_RATE):
            voiced_frames.append(frame)
            silence_ms = 0
        else:
            silence_ms += FRAME_MS
            if voiced_frames and silence_ms > SILENCE_TIMEOUT_MS:
                segment = b"".join(voiced_frames)
                audio_np = (
                    np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
                )

                try:
                    audio_np = nr.reduce_noise(y=audio_np, sr=SAMPLE_RATE)
                except:
                    pass

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_np, SAMPLE_RATE)
                    path = tmp.name

                segments, _ = model.transcribe(path)
                os.remove(path)

                text = " ".join(seg.text for seg in segments).strip()
                if text:
                    print("🗣", text)          # DEBUG
                    result_queue.put(text)    # PUSH CONTINUOUSLY

                voiced_frames = []
                silence_ms = 0

def start_listener():
    threading.Thread(target=process_audio, daemon=True).start()

    print("🎙 Listening continuously...")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)

# ---- standalone test ----
if __name__ == "__main__":
    start_listener()
