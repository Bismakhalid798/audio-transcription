import os
import tempfile
import av

# 1. Disable symlinks for Hugging Face on Windows environments
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 2. Patch PyAV compatibility issue (fixes 'metadata_errors' TypeError)
_original_av_open = av.open
def _patched_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _original_av_open(*args, **kwargs)
av.open = _patched_av_open

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Audio Transcription API")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

