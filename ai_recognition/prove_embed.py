# Proof-of-life: load the pretrained buffalo_s model and embed faces from a sample image.
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

print("Loading buffalo_s model (downloads ~tens of MB on first run)...")
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model ready.\n")

# Built-in sample image that contains faces.
img = ins_get_image("t1")
faces = app.get(img)
print(f"Faces detected: {len(faces)}")

for i, f in enumerate(faces):
    emb = f.normed_embedding          # 512-d normalized embedding
    print(f"  face {i}: bbox={f.bbox.astype(int).tolist()}  "
          f"embedding_dim={emb.shape[0]}  det_score={f.det_score:.3f}")

# Quick cosine-similarity sanity check: same face vs different face.
if len(faces) >= 2:
    a = faces[0].normed_embedding
    b = faces[1].normed_embedding
    sim_self = float(np.dot(a, a))    # identical -> ~1.0
    sim_diff = float(np.dot(a, b))    # different people -> lower
    print(f"\nCosine(self, self) = {sim_self:.3f}  (expect ~1.0)")
    print(f"Cosine(face0, face1) = {sim_diff:.3f}  (different people -> lower)")

print("\nEMBEDDING PIPELINE WORKS.")
