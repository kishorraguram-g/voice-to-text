import queue
import threading
import time
import tempfile
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import torch
import librosa

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

# =====================================================
# CONFIG (WINDOWS-STABLE)
# =====================================================

# ✅ Use MME microphone (stable on Windows)
MIC_DEVICE_INDEX = 1   # Microphone Array (Intel Smart Sound, MME)

# ✅ Native device rate (MME supports this)
DEVICE_SAMPLE_RATE = 44100

# ✅ Whisper requirement
TARGET_SAMPLE_RATE = 16000

# Chunking (Silero VAD needs >= 0.3s)
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_SECONDS)

# Silence duration to finalize sentence
SILENCE_TIMEOUT = 1.0  # seconds

# =====================================================
# LOAD MODELS
# =====================================================

vad_model = load_silero_vad()

whisper_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

# =====================================================
# QUEUES
# =====================================================

audio_queue = queue.Queue()
result_queue = queue.Queue()

# =====================================================
# AUDIO CALLBACK (DEVICE RATE)
# =====================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠", status)

    # Flatten to mono float32
    audio_queue.put(indata.copy().flatten())


# =====================================================
# AUDIO PROCESSOR
# =====================================================

def process_audio():
    buffer_16k = np.zeros(0, dtype=np.float32)
    voiced_chunks = []
    last_voice_time = time.time()

    while True:
        # Get audio captured at DEVICE_SAMPLE_RATE
        raw_audio = audio_queue.get()

        # 🔁 Resample to 16kHz (REQUIRED)
        audio_16k = librosa.resample(
            raw_audio,
            orig_sr=DEVICE_SAMPLE_RATE,
            target_sr=TARGET_SAMPLE_RATE
        )

        buffer_16k = np.concatenate([buffer_16k, audio_16k])

        while len(buffer_16k) >= CHUNK_SAMPLES:
            chunk = buffer_16k[:CHUNK_SAMPLES]
            buffer_16k = buffer_16k[CHUNK_SAMPLES:]

            # Noise reduction (safe)
            try:
                chunk = nr.reduce_noise(
                    y=chunk,
                    sr=TARGET_SAMPLE_RATE
                )
            except:
                pass

            # Voice Activity Detection
            speech = get_speech_timestamps(
                torch.from_numpy(chunk),
                vad_model,
                sampling_rate=TARGET_SAMPLE_RATE
            )

            if speech:
                voiced_chunks.append(chunk)
                last_voice_time = time.time()
            else:
                if voiced_chunks and (time.time() - last_voice_time) >= SILENCE_TIMEOUT:
                    transcribe_sentence(voiced_chunks)
                    voiced_chunks = []


# =====================================================
# TRANSCRIPTION
# =====================================================

def transcribe_sentence(chunks):
    audio = np.concatenate(chunks)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, TARGET_SAMPLE_RATE)
        path = tmp.name

    try:
        segments, _ = whisper_model.transcribe(
            path,
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(seg.text for seg in segments).strip()
        if text:
            print("🗣", text)
            result_queue.put(text)

    finally:
        os.remove(path)


# =====================================================
# START LISTENER
# =====================================================

def start_listener():
    threading.Thread(
        target=process_audio,
        daemon=True
    ).start()

    print("🎙 Listening continuously...")

    with sd.InputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=1,
        device=MIC_DEVICE_INDEX,
        callback=audio_callback,
    ):
        while True:
            time.sleep(0.1)


# =====================================================
# STANDALONE MODE
# =====================================================

if __name__ == "__main__":
    start_listener()
