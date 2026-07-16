"""Minimal HTTP API for IndexTTS2.

Run:  uv run api_server.py  (listens on 0.0.0.0:8000)

  GET  /voices                      -> list of available voice names
  POST /tts  (multipart form)       -> audio/wav
       text        (required)  text to speak
       voice       (optional)  name from /voices
       voice_file  (optional)  uploaded wav sample (overrides voice)
       emo_vector  (optional)  JSON list of 8 floats, each 0.0..1.0, in this
                               exact order:
                                 [happy, angry, sad, afraid, disgusted,
                                  melancholic, surprised, calm]
                               e.g. mostly sad + a little calm:
                                 [0, 0, 0.8, 0, 0, 0, 0, 0.2]
                               Note: the model rescales vectors whose sum
                               exceeds 0.8, so keep the total modest.
       emo_file    (optional)  emotion reference wav (copies its emotion)
       emo_alpha   (optional)  emotion strength 0.0..1.0 (default 1.0)
       emo_text    (optional)  free-text emotion description, auto-converted
                               to an emotion vector (e.g. "furious shouting")

  Emotion params are mutually exclusive; precedence: emo_vector > emo_text
  > emo_file. Omit all for neutral speech.
  Interactive docs (same info): http://HOST:8000/docs

Example:
  curl -o out.wav -F text="hello there" -F voice=voice_01 \
       -F emo_vector='[0,0,0.8,0,0,0,0,0.2]' http://HOST:8000/tts
"""
import json
import os
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "examples"))
EMOTION_ORDER = ["happy", "angry", "sad", "afraid", "disgusted",
                 "melancholic", "surprised", "calm"]

app = FastAPI(title="IndexTTS2 API")
tts = None
lock = threading.Lock()  # ponytail: model isn't thread-safe; one request at a time


@app.on_event("startup")
def load_model():
    global tts
    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")


@app.get("/voices")
def voices():
    """List available voice names for the `voice` form field."""
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


@app.get("/emotions")
def emotions():
    """Emotion names in emo_vector index order."""
    return EMOTION_ORDER


def _save_upload(f: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(f.file.read())
        return tmp.name


@app.post("/tts")
def synthesize(
    text: str = Form(..., description="Text to synthesize."),
    voice: str = Form(None, description="Voice name from GET /voices."),
    voice_file: UploadFile = None,
    emo_vector: str = Form(None, description=(
        "JSON list of 8 floats, each 0.0-1.0, in order "
        "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. "
        "Example: [0,0,0.8,0,0,0,0,0.2] (sad + a little calm). "
        "Sums over 0.8 are rescaled by the model.")),
    emo_file: UploadFile = None,
    emo_alpha: float = Form(1.0, ge=0.0, le=1.0,
                            description="Emotion strength, 0.0-1.0."),
    emo_text: str = Form(None, description=(
        "Free-text emotion description, auto-converted to an emotion "
        "vector. Example: 'furious shouting'.")),
):
    """Synthesize speech. Returns audio/wav.

    Emotion params are mutually exclusive; precedence:
    emo_vector > emo_text > emo_file. Omit all for neutral speech.
    """
    tmp_files = []
    if voice_file is not None:
        spk = _save_upload(voice_file)
        tmp_files.append(spk)
    elif voice:
        spk = str(VOICES_DIR / f"{voice}.wav")
        if not os.path.exists(spk):
            raise HTTPException(404, f"unknown voice: {voice}")
    else:
        raise HTTPException(422, "provide voice or voice_file")

    vec = None
    if emo_vector:
        try:
            vec = json.loads(emo_vector)
        except json.JSONDecodeError:
            raise HTTPException(422, "emo_vector must be a JSON list, e.g. [0,0,0.8,0,0,0,0,0.2]")
        if not (isinstance(vec, list) and len(vec) == 8
                and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in vec)):
            raise HTTPException(
                422,
                f"emo_vector must be 8 floats in 0.0-1.0, order: {EMOTION_ORDER}")

    emo_audio = None
    if emo_file is not None:
        emo_audio = _save_upload(emo_file)
        tmp_files.append(emo_audio)

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tmp_files.append(out)
    try:
        with lock:
            tts.infer(
                spk_audio_prompt=spk,
                text=text,
                output_path=out,
                emo_vector=vec,
                emo_audio_prompt=emo_audio,
                emo_alpha=emo_alpha,
                use_emo_text=emo_text is not None,
                emo_text=emo_text,
            )
        return Response(Path(out).read_bytes(), media_type="audio/wav")
    finally:
        for f in tmp_files:
            os.unlink(f)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
