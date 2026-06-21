"""
face_engine.py
--------------
Wraps the pretrained InsightFace (buffalo_s / MobileFaceNet) model.

Responsibilities:
  * load the model once
  * turn an image into a 512-d face embedding (detect -> align -> embed)
  * compare an embedding against a set of enrolled embeddings (cosine similarity)

Everything here runs locally on CPU. No internet, no cloud API.
"""
from __future__ import annotations

import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, model_name: str = "buffalo_s", det_size: int = 640):
        # Load the pretrained model (downloaded once to ~/.insightface).
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    # ---- image decoding -------------------------------------------------
    @staticmethod
    def decode_image(raw: bytes) -> np.ndarray:
        """Decode raw image bytes (jpg/png) into a BGR numpy array."""
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        return img

    # ---- core: image -> single embedding --------------------------------
    def embed(self, img: np.ndarray):
        """
        Detect faces and return the embedding for the most prominent one.

        Returns a dict describing the outcome so the caller can react to
        edge cases instead of guessing:
          {"status": "ok",        "embedding": np.ndarray, "det_score": float, "num_faces": int}
          {"status": "no_face",   "num_faces": 0}
        """
        faces = self.app.get(img)
        if not faces:
            return {"status": "no_face", "num_faces": 0}

        # Edge case: multiple faces -> pick the largest bounding box
        # (closest person to the camera), which is the natural check-in subject.
        def area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)

        face = max(faces, key=area)
        return {
            "status": "ok",
            "embedding": face.normed_embedding.astype(np.float32),
            "det_score": float(face.det_score),
            "num_faces": len(faces),
        }

    def embed_bytes(self, raw: bytes):
        """Convenience: raw bytes -> embed()."""
        return self.embed(self.decode_image(raw))

    # ---- matching: embedding vs enrolled set ----------------------------
    @staticmethod
    def best_match(probe: np.ndarray, gallery: dict[str, np.ndarray]):
        """
        Compare a probe embedding against {person_id: embedding}.
        Embeddings are L2-normalized, so cosine similarity = dot product.

        Returns (best_person_id, best_score). gallery empty -> (None, -1.0).
        """
        if not gallery:
            return None, -1.0
        best_id, best_score = None, -1.0
        for pid, emb in gallery.items():
            score = float(np.dot(probe, emb))
            if score > best_score:
                best_id, best_score = pid, score
        return best_id, best_score
