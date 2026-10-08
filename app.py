import os
import tempfile
import numpy as np
import av

# 1. Disable symlinks for Hugging Face on Windows environments
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 2. Patch PyAV compatibility issue (fixes 'metadata_errors' TypeError)
_original_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)
av.open = _patched_av_open

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Real-Time & Batch Audio Transcription API")

# Global Whisper model setup
MODEL_SIZE = "medium.en"
DEVICE = "cpu"        # Change to "cuda" if running on NVIDIA GPU
COMPUTE_TYPE = "int8" # Change to "float16" if running on CUDA

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# --- ROOT REDIRECT (Fixes 404 on localhost:8000) ---
@app.get("/", include_in_schema=False)
async def root():
    """Redirects root URL to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

# --- REST ENDPOINT (Batch File Uploads) ---
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    temp_path = None
    try:
        # Save temporary file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Perform inference
        segments, info = model.transcribe(
            temp_path,
            language="en",
            beam_size=5,
            vad_filter=True  # Filter out silence to prevent hallucinations
        )

        results = []
        for segment in segments:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })

        return JSONResponse(content={
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- WEBSOCKET ENDPOINT (Real-Time Audio Streaming) ---
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = []
    chunk_samples = 16000 * 2  # 2 seconds of audio at 16kHz

    try:
        while True:
            # Expecting raw binary PCM float32 data from client
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer.append(chunk)

            total_samples = sum(len(c) for c in audio_buffer)
            if total_samples >= chunk_samples:
                audio_data = np.concatenate(audio_buffer)[:chunk_samples]
                audio_buffer = []  # Clear buffer

                segments, _ = model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=1  # Fast greedy decoding for real-time stream
                )

                transcript = " ".join([seg.text.strip() for seg in segments if seg.text])
                if transcript:
                    await websocket.send_json({"text": transcript})

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="localhost", port=8000)
# http://localhost:8000/docs