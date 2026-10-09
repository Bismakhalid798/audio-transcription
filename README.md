# Audio Transcription Pipeline

A fast, lightweight speech-to-text service built with **FastAPI**, **`faster-whisper`** (CTranslate2), and **PyAV**. Supports both synchronous batch audio file transcription via REST API and live audio streaming via WebSockets.

---

## Repository Structure

* **`app.py`**: FastAPI service handling batch file transcription (`POST /transcribe`) and documentation redirects (`GET /`).
* **`speech-to-text.py`**: Real-time speech transcription script using WebSockets for live audio chunk processing.
* **`requirements.txt`**: Project dependencies.

---

## System Architecture

```text
                   +-----------------------------------+
                   |    Client (Browser / API Client)  |
                   +-----------------+-----------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
  [ HTTP POST /transcribe ]                           [ WebSocket /ws/transcribe ]
           |                                                   |
           v                                                   v
  +-----------------+                                 +-----------------+
  |     app.py      |                                 | speech-to-text  |
  +--------+--------+                                 +--------+--------+
           |                                                   |
           v                                                   v
  +-----------------+                                 +-----------------+
  |  Temp Storage   |                                 |  2s  Buffer     |
  +--------+--------+                                 +--------+--------+
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                                     v
                           +-------------------+
                           | PyAV Audio Decode |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           |  faster-whisper   |
                           |   (CTranslate2)   |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           | Structured Output |
                           |   (JSON / Text)   |
                           +-------------------+
```

---

## Key Design Decisions in `app.py`

1. **CTranslate2 Engine (`faster-whisper`)**: Replaces standard PyTorch implementations to achieve up to 4x faster inference and low memory usage through `int8` CPU quantization.
2. **Environment & Compatibility Patches**:
   * Sets `HF_HUB_DISABLE_SYMLINKS="1"` to avoid privilege issues on Windows machines.
   * Monkey-patches `av.open` to strip `metadata_errors` parameters, fixing version mismatches between CTranslate2 and PyAV.
3. **Stateless File Cleanup**: Saves uploaded files temporarily with Python's `tempfile` module and ensures deletion inside a `finally` block to prevent disk clutter and memory leaks.
4. **Accuracy & VAD Filtering**: Enables Voice Activity Detection (`vad_filter=True`) and `beam_size=5` on batch uploads to trim background silence and eliminate hallucination loops.

---

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Batch API (app.py)

```bash
python app.py
```

Interactive API Docs: http://localhost:8000/docs

Transcribe Audio File:

```bash
curl -X 'POST' \
  'http://localhost:8000/transcribe' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.mp3;type=audio/mpeg'
```

### 3. Run Live Streaming (speech-to-text.py)

```bash
python speech-to-text.py
```

---

## API Output Example

```json
{
  "language": "en",
  "duration": 12.45,
  "segments": [
    {
      "start": 0.0,
      "end": 2.14,
      "text": "Hello and welcome to the transcription service."
    }
  ]
}
```
