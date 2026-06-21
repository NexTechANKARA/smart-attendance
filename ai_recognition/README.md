# AI/ML Face Recognition Service

Local, pretrained, fast face-recognition microservice for the Smart Attendance system.
It answers one question for the backend: **"who is this face?"**

- **Local only** — runs on CPU, no internet, no cloud API (works WiFi-off).
- **Pretrained** — InsightFace `buffalo_s` (MobileFaceNet / ArcFace, ~tens of MB). No training.
- **Fast** — embeddings cached in RAM; matching is microseconds. ~2 KB per enrolled face.

## Run

```bash
conda activate face-attend
cd ai_recognition
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

## API contract (for the Spring Boot teammate)

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | `/health` | — | `{status, enrolled}` |
| POST | `/enroll` | form: `person_id`, `image` (file) | `{status:"enrolled", person_id, det_score, ...}` |
| POST | `/recognize` | form: `image` (file) | `{status, person_id, confidence, ...}` |
| GET | `/people` | — | `{count, person_ids}` |
| DELETE | `/people/{person_id}` | — | `{status:"removed", person_id}` |

`/recognize` `status` is one of: `match`, `unknown`, `no_face`, `no_enrolled`.

## How recognition works

1. Detect face(s) in the image. No face → `no_face`. Multiple → pick the largest (closest).
2. Align + embed the face → 512-d vector.
3. Cosine-compare against enrolled vectors in RAM → closest match + score.
4. Score ≥ `MATCH_THRESHOLD` (default **0.40**) → `match`; otherwise → `unknown`.

## Engineering decisions

- **No fine-tuning.** Face recognition is metric learning: enrolling a person is a one-shot
  embedding, not a training problem. Fine-tuning would only help under domain shift
  (e.g. thermal cameras) and risks overfitting on small data.
- **Stateless by design.** The backend DB is the source of truth for embeddings; this service
  caches them in RAM for speed and persists a local `embeddings.json` so demos survive restarts.
- **Threshold is the tuned knob.** `MATCH_THRESHOLD` in `app.py` trades false accepts vs.
  false rejects.
- **Edge cases handled:** no face, multiple faces, unknown (un-enrolled) person, empty gallery.

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI endpoints (the service) |
| `face_engine.py` | Model load, embed, cosine matching |
| `store.py` | In-RAM embedding store + JSON persistence |
| `prove_embed.py` | Standalone proof that the model embeds faces |
| `requirements.txt` | Dependencies |
