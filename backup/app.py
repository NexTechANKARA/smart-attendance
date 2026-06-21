"""
app.py
------
FastAPI service exposing the face-recognition engine to the backend.

This is the local "door" Spring Boot knocks on. It runs on localhost,
makes no internet calls, and works fully offline.

Endpoints:
  GET  /health                 -> {"status": "ok", "enrolled": N}
  POST /enroll                 -> register a face under a person_id
  POST /recognize              -> identify the face in an image (the check-in)
  GET  /people                 -> list enrolled person_ids
  DELETE /people/{person_id}   -> remove an enrolled person

Tunable:
  MATCH_THRESHOLD  cosine similarity cut-off for "same person".
                   Higher = stricter. ~0.40 is a good start for buffalo_s.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from face_engine import FaceEngine
from store import EmbeddingStore

# --- recognition threshold (the engineering knob) --------------------------
MATCH_THRESHOLD = 0.40

app = FastAPI(title="Smart Attendance - Face Recognition Service", version="1.0")

# Allow the demo page / backend on other ports to call us during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: FaceEngine | None = None
store = EmbeddingStore()


@app.on_event("startup")
def _startup() -> None:
    # Load the model once when the service boots (slow part happens here, not per-request).
    global engine
    engine = FaceEngine()


@app.get("/demo", include_in_schema=False)
def demo_page():
    # Serve the webcam demo page from localhost so the browser grants camera access.
    return FileResponse(Path(__file__).with_name("demo.html"))


@app.get("/health")
def health():
    return {"status": "ok", "enrolled": len(store)}


@app.get("/people")
def people():
    return {"count": len(store), "person_ids": store.ids()}


@app.delete("/people/{person_id}")
def delete_person(person_id: str):
    removed = store.remove(person_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{person_id} not enrolled")
    return {"status": "removed", "person_id": person_id}


@app.post("/enroll")
async def enroll(person_id: str = Form(...), image: UploadFile = File(...)):
    """Register a face: image -> embedding -> stored under person_id."""
    raw = await image.read()
    try:
        result = engine.embed_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result["status"] == "no_face":
        return JSONResponse(status_code=422,
                            content={"status": "no_face",
                                     "message": "No face detected in the image."})

    store.add(person_id, result["embedding"])
    return {
        "status": "enrolled",
        "person_id": person_id,
        "det_score": round(result["det_score"], 3),
        "faces_seen": result["num_faces"],
        "total_enrolled": len(store),
    }


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    """The live check-in: image -> embedding -> nearest enrolled person."""
    raw = await image.read()
    try:
        result = engine.embed_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Edge case 1: nobody in frame.
    if result["status"] == "no_face":
        return {"status": "no_face", "person_id": None, "confidence": 0.0}

    probe = result["embedding"]
    best_id, score = FaceEngine.best_match(probe, store.all())

    # Edge case 2: gallery empty -> nothing to match against.
    if best_id is None:
        return {"status": "no_enrolled", "person_id": None, "confidence": 0.0}

    # Edge case 3: closest match still too far -> unknown, don't force a wrong identity.
    if score < MATCH_THRESHOLD:
        return {"status": "unknown", "person_id": None,
                "confidence": round(score, 3),
                "faces_seen": result["num_faces"]}

    return {
        "status": "match",
        "person_id": best_id,
        "confidence": round(score, 3),
        "faces_seen": result["num_faces"],
    }
