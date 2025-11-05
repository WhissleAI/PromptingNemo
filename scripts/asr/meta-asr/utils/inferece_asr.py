#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import yaml
import json
import time
from pathlib import Path

import torch
import uvicorn
import librosa
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
# --- Project Setup ---
# Add the project root to the Python path to allow importing custom modules.
try:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.asr.meta_asr.nemo_adapter_with_langid import CustomEncDecCTCModelBPE
except (ImportError, IndexError):
    print(
        "Could not import custom NeMo model. "
        "Please ensure this script is run from its original location "
        "within the 'PromptingNemo/scripts/asr/meta_asr/' directory."
    )
    sys.exit(1)

# --- Arg parsing for configuration ---
# We parse a subset of args here to configure globals BEFORE the app is fully built.
# The full parsing happens in main().
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use for inference.")
_parser.add_argument("--model-path", type=str, default=None, help="Path to the .nemo checkpoint file.")
_args, _ = _parser.parse_known_args()

# --- Globals ---
MODEL = None
DEVICE = f"cuda:{_args.gpu_id}" if torch.cuda.is_available() else "cpu"
# This global will be populated by the pre-parser and used by the startup event.
MODEL_PATH = _args.model_path
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_BUSTER = str(int(time.time())) # Unique value on each server start

# --- FastAPI App ---
app = FastAPI(
    title="NeMo ASR Inference API",
    description="Transcribe audio files using a fine-tuned NeMo Parakeet model.",
    version="1.0.0",
)

# Mount static files directory (for CSS, JS)
app.mount("/static", StaticFiles(directory=SCRIPT_DIR / "static"), name="static")

# --- Model Loading ---
def load_model(model_path: str):
    """
    Loads the ASR model from a .nemo checkpoint file.
    The model is loaded into a global variable `MODEL`.
    """
    global MODEL
    if MODEL is not None:
        print("Model is already loaded.")
        return

    print(f"Attempting to load model from: {model_path}")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            f"Please provide a valid path using the --model-path argument."
        )

    # Use the custom model class for restoration
    model = CustomEncDecCTCModelBPE.restore_from(model_path, map_location=torch.device(DEVICE))
    model.eval()
    MODEL = model
    model_name = Path(model_path).name
    print(f"Model '{model_name}' loaded successfully on {DEVICE}.")


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    Determines config path and loads the model.
    """
    # Validate the selected device
    if "cuda" in DEVICE:
        try:
            gpu_id = int(DEVICE.split(':')[-1])
            if gpu_id >= torch.cuda.device_count():
                raise IndexError(f"GPU ID {gpu_id} is out of range.")
            torch.cuda.set_device(gpu_id)
            print(f"Successfully set active device to GPU {gpu_id}")
        except (IndexError, ValueError, RuntimeError) as e:
            available_gpus = torch.cuda.device_count()
            print(f"FATAL: Invalid GPU ID specified for device '{DEVICE}'. Available GPUs: {available_gpus}. Error: {e}")
            return  # Prevent model loading attempt

    if not MODEL_PATH:
        print("FATAL: --model-path argument is required. The server will start but will be non-functional.")
        return

    try:
        load_model(MODEL_PATH)
    except Exception as e:
        print(f"FATAL: Could not load model during startup. Error: {e}")
        # The server will still start, but endpoints will fail.
        # This is often better than crashing in a containerized environment.


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_index():
    """Serves the main HTML user interface with cache busting."""
    index_path = SCRIPT_DIR / "templates" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<html><body><h1>Error</h1><p>index.html not found.</p></body></html>", status_code=500)
    
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Dynamically add cache-busting query parameters to static file URLs
    content = content.replace('/static/style.css', f'/static/style.css?v={CACHE_BUSTER}')
    content = content.replace('/static/script.js', f'/static/script.js?v={CACHE_BUSTER}')
    
    return HTMLResponse(content=content)


@app.post("/transcribe", tags=["ASR"])
async def transcribe_audio(file: UploadFile = File(..., description="Audio file to transcribe.")):
    """
    Transcribes a given audio file.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503, detail="Model is not available. Check server logs for errors."
        )

    # Use temporary files for audio and manifest
    tmp_audio_path = None
    tmp_manifest_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_audio:
            content = await file.read()
            tmp_audio.write(content)
            tmp_audio_path = tmp_audio.name

        # Get audio duration using librosa
        duration = librosa.get_duration(path=tmp_audio_path)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_manifest:
            manifest_entry = {
                "audio_filepath": tmp_audio_path,
                "duration": duration,
                "lang": "EN",
                "lang_family": "ENGLISH"
            }
            tmp_manifest.write(json.dumps(manifest_entry) + '\n')
            tmp_manifest_path = tmp_manifest.name

        print(f"Transcribing file: {file.filename} (temp audio: {tmp_audio_path}, temp manifest: {tmp_manifest_path})")

        # Pass the manifest path via the 'audio' argument
        transcriptions = MODEL.transcribe(audio=[tmp_manifest_path], batch_size=1)
        
        # Handle complex hypothesis objects vs. plain strings returned by NeMo
        if isinstance(transcriptions, tuple) and len(transcriptions) > 0:
            result = transcriptions[0]
        else:
            result = transcriptions

        if isinstance(result, list) and result:
            first_hyp = result[0]
            if hasattr(first_hyp, 'text'):
                transcription = first_hyp.text
            else:
                transcription = str(first_hyp)
        else:
            transcription = "Transcription failed or produced no output."

        print(f"Transcription result: {transcription}")

        return {"filename": file.filename, "transcription": transcription}

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed due to an internal error: {str(e)}")
    finally:
        # Clean up the temporary files
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)
        if tmp_manifest_path and os.path.exists(tmp_manifest_path):
            os.unlink(tmp_manifest_path)
# --- Main execution ---
def main():
    """
    Main function to run the server.
    Parses command-line arguments for server configuration.
    """
    parser = argparse.ArgumentParser(description="FastAPI server for NeMo ASR Inference")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    # Re-add the arguments so they appear in the help message
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="GPU device ID to use for inference. Default: 0."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the .nemo checkpoint file for ASR. Example: /path/to/your/model.nemo",
    )
    args = parser.parse_args()

    # The global DEVICE and MODEL_PATH are already set using the pre-parser.
    print(f"Starting FastAPI server...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    # To run this server:
    # 1. Ensure you have the necessary packages:
    #    pip install fastapi uvicorn python-multipart
    # 2. Make sure your NeMo environment is active.
    # 3. Run the script from the command line:
    #    python PromptingNemo/scripts/asr/meta_asr/inferece_asr.py --model-path YOUR_MODEL.nemo --gpu-id 0
    main()
