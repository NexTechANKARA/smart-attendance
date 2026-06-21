# End-to-end HTTP test of the running service (port 8088).
# Prepares face image files from InsightFace's bundled sample, then
# exercises /enroll and /recognize for: match, unknown, no_face.
import io
import numpy as np
import cv2
import requests
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

BASE = "http://127.0.0.1:8088"

# --- prepare test face crops from sample image -----------------------------
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image("t1")
faces = sorted(app.get(img), key=lambda f: f.bbox[0])  # left-to-right order

def crop(face, pad=0.4):
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * w)); y1 = max(0, int(y1 - pad * h))
    x2 = min(img.shape[1], int(x2 + pad * w)); y2 = min(img.shape[0], int(y2 + pad * h))
    return img[y1:y2, x1:x2]

def to_jpg(arr):
    ok, buf = cv2.imencode(".jpg", arr)
    return buf.tobytes()

alice = to_jpg(crop(faces[0]))
bob = to_jpg(crop(faces[1]))
blank = to_jpg(np.full((300, 300, 3), 240, np.uint8))  # no face

def post(path, files, data=None):
    r = requests.post(f"{BASE}{path}", files=files, data=data)
    return r.status_code, r.json()

print("health:", requests.get(f"{BASE}/health").json())

print("\n--- enroll alice ---")
print(post("/enroll", {"image": ("alice.jpg", alice, "image/jpeg")}, {"person_id": "alice"}))

print("\n--- recognize alice (expect MATCH alice) ---")
print(post("/recognize", {"image": ("probe.jpg", alice, "image/jpeg")}))

print("\n--- recognize bob (expect UNKNOWN: only alice enrolled) ---")
print(post("/recognize", {"image": ("probe.jpg", bob, "image/jpeg")}))

print("\n--- enroll bob, then recognize bob (expect MATCH bob) ---")
print(post("/enroll", {"image": ("bob.jpg", bob, "image/jpeg")}, {"person_id": "bob"}))
print(post("/recognize", {"image": ("probe.jpg", bob, "image/jpeg")}))

print("\n--- recognize blank image (expect NO_FACE) ---")
print(post("/recognize", {"image": ("blank.jpg", blank, "image/jpeg")}))

print("\nfinal people:", requests.get(f"{BASE}/people").json())
