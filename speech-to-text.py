import os
import sys
import queue
import threading

# Fix for WinError 1314: download Hugging Face models without symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import av
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Patch av.open to strip 'metadata_errors' on older PyAV versions
_original_av_open = av.open


def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)


av.open = _patched_av_open

# Settings
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5  # seconds
CHUNK_DURATION = 2    # seconds
CHANNELS = 1

FRAMES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_DURATION)
FRAMES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

model = WhisperModel("medium.en", device="cpu", compute_type="int8")


def transcribe_file(path):
    segments, _ = model.transcribe(path, language="en", beam_size=5)
    for segment in segments:
        print(segment.text)


def transcribe_live():
    audio_queue = queue.Queue()
    audio_buffer = []

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    def recorder():
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=FRAMES_PER_BLOCK,
            callback=audio_callback,
        ):
            print("Listening... Press Ctrl+C to stop")
            while True:
                sd.sleep(100)

    threading.Thread(target=recorder, daemon=True).start()

    while True:
        audio_buffer.append(audio_queue.get())

        if sum(len(b) for b in audio_buffer) >= FRAMES_PER_CHUNK:
            audio_data = np.concatenate(audio_buffer)[:FRAMES_PER_CHUNK]
            audio_buffer.clear()
            audio_data = audio_data.flatten().astype(np.float32)

            segments, _ = model.transcribe(audio_data, language="en", beam_size=1)
            for segment in segments:
                print(segment.text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1])  # python transcribe.py audio.mp3
    else:
        transcribe_live()             # python transcribe.py