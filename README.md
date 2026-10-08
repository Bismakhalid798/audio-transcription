# Audio Transcription & Real-Time Streaming Pipeline

A high-performance, dual-mode speech-to-text service built with **FastAPI**, **`faster-whisper`** (CTranslate2), and **PyAV**. It supports both synchronous batch transcription of audio files (with per-segment timestamps) and low-latency live streaming over WebSockets.

---

## Table of Contents

- [Key Features](#key-features)
- [Architectural & Design Decisions](#architectural--design-decisions)
- [API Reference](#api-reference)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Production Scaling Strategy](#production-scaling-strategy)

---

## Key Features

- **Batch File Processing (`POST /transcribe`)**: Accepts audio file uploads (`.mp3`, `.wav`, `.m4a`, `.flac`), extracts spoken text, and returns structured JSON with precise per-segment timestamps.
- **Real-Time Streaming (`WebSocket /ws/transcribe`)**: Ingests continuous binary PCM audio over WebSockets and performs chunked sliding-window inference with minimal latency.
- **Format Agnostic**: Uses FFmpeg bindings via PyAV to automatically decode and resample incoming audio, with no external pre-conversion needed.
- **Resilient Architecture**: Includes monkey-patches for PyAV C-binding compatibility and environment-level overrides for cross-platform deployment (Windows/Linux).

---

## Architectural & Design Decisions

### 1. Inference Engine (`faster-whisper`)

**Why CTranslate2?** Standard PyTorch implementations of OpenAI's Whisper suffer from high memory consumption and slower throughput. `faster-whisper` uses CTranslate2, a custom inference engine for Transformer models that delivers up to **4x faster inference** and significantly lower VRAM usage through weight quantization (`int8` and `float16`).

**Model configurations**

| Endpoint | Settings | Purpose |
|----------|----------|---------|
| Batch (`/transcribe`) | `beam_size=5`, `vad_filter=True` | Voice Activity Detection strips background silence, maximizes accuracy, and eliminates hallucination loops on long audio files. |
| Streaming (`/ws/transcribe`) | `beam_size=1` (greedy decoding), 2-second sliding window (`chunk_samples = 32000`) | Minimizes processing latency. |

### 2. Dual Ingestion Pipeline (Batch vs. Stream)

- **Batch I/O Safety**: Uploaded files are written to isolated temporary files using Python's `tempfile` module and cleaned up in a `finally` block. This keeps the application stateless and prevents memory exhaustion from holding large files in RAM.
- **WebSocket Streaming Buffer**: Clients send raw 16 kHz `float32` PCM arrays. The service accumulates chunks in an in-memory queue until a 2-second buffer threshold is reached, flattens the tensor, and dispatches it directly to the model.

### 3. Cross-Platform & Dependency Compatibility Patches

To ensure reliable execution across varying Windows environments and library updates:

- **Hugging Face Symlink Bypassing**: `HF_HUB_DISABLE_SYMLINKS="1"` is injected at runtime to prevent `WinError 1314` privilege crashes on Windows machines where Developer Mode or Administrator privileges are restricted.
- **PyAV Keyword Interception**: A dynamic monkey-patch wraps `av.open` to strip unsupported `metadata_errors` arguments, neutralizing version mismatches between CTranslate2 and the underlying PyAV version.

---

## API Reference

### 1. Root Redirect

**`GET /`**

Automatically redirects clients to `/docs` for interactive Swagger API testing.

### 2. Batch Audio Transcription

**`POST /transcribe`**

- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (audio file: `.mp3`, `.wav`, `.m4a`, etc.)

**Example response**

```json
{
  "language": "en",
  "duration": 12.45,
  "segments": [
    {
      "start": 0.0,
      "end": 2.14,
      "text": "Hello and welcome to the transcription pipeline."
    },
    {
      "start": 2.14,
      "end": 5.82,
      "text": "This model processes audio with per-segment timestamps."
    }
  ]
}
```

### 3. Real-Time Streaming

**`WebSocket /ws/transcribe`**

- **Input protocol**: Binary frames containing raw 16,000 Hz `float32` little-endian PCM audio.
- **Output protocol**: Text JSON frames returning partial/incremental transcriptions.

**Example output frame**

```json
{
  "text": "Hello and welcome to the live stream"
}
```

---

## Installation & Setup

### Prerequisites

- **Python**: 3.10 or higher
- **FFmpeg**: Required for audio decoding (handled automatically via `av`)
- **Hardware**: CPU (default `int8`) or NVIDIA GPU (`cuda` with `float16` compute type)

### 1. Repository Setup

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Virtual Environment & Dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn faster-whisper numpy av python-multipart
```

---

## Running the Application

Start the server locally:

```bash
python app.py
```

- **Interactive OpenAPI/Swagger docs**: <http://localhost:8000/docs>
- **Root endpoint (redirects to docs)**: <http://localhost:8000/>

---

## Production Scaling Strategy

To evolve this single-instance service into a distributed, production-grade architecture:

```
[ Client ] ---> [ API Gateway / Load Balancer ]
                        |
            +-----------+-----------+
            |                       |
    [ FastAPI Nodes ]       [ WebSocket Nodes ]
            |                       |
            +----------+------------+
                       |
                 [ Redis Queue ]
                       |
               [ Celery Workers ] (GPU Pool)
                       |
            +----------+------------+
            |                       |
     [ AWS S3 Storage ]     [ PostgreSQL / MongoDB ]
    (Raw Audio Files)     (Transcripts & Timestamps)
```

1. **Message Queue Offloading (Celery + Redis)**: For batch files, the web API should immediately return an HTTP `202 Accepted` response with a `job_id`. Heavy matrix operations are offloaded to asynchronous background GPU workers pulling from a Redis queue.
2. **Object & Document Storage**: Audio files are persisted in Amazon S3 or Google Cloud Storage. Completed JSON transcriptions and metadata are indexed in MongoDB or PostgreSQL (`JSONB`) for fast temporal queries.
3. **Resiliency & Dead Letter Queues (DLQ)**: If a worker hits an Out-Of-Memory (OOM) error, tasks automatically retry with exponential backoff. Tasks that fail consistently are routed to a Dead Letter Queue for developer auditing.
