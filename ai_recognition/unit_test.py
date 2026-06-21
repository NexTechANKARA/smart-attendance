# Unit tests for store.py persistence and FaceEngine.best_match (no server needed).
import os
import numpy as np
from store import EmbeddingStore
from face_engine import FaceEngine

p = "_unit.json"
if os.path.exists(p):
    os.remove(p)

s = EmbeddingStore(p)
a = np.ones(512, dtype=np.float32);   a /= np.linalg.norm(a)
b = np.arange(512, dtype=np.float32); b /= np.linalg.norm(b)
s.add("alice", a)
s.add("bob", b)
assert len(s) == 2, "add failed"

# persistence: reload from disk
s2 = EmbeddingStore(p)
assert set(s2.ids()) == {"alice", "bob"}, "persistence failed"

# matching: probe == alice -> alice, score ~1.0
bid, score = FaceEngine.best_match(a, s2.all())
assert bid == "alice" and score > 0.99, f"match wrong: {bid}, {score}"

# matching against bob's vector -> bob
bid2, _ = FaceEngine.best_match(b, s2.all())
assert bid2 == "bob", "match wrong for bob"

# empty gallery -> (None, -1)
none_id, none_score = FaceEngine.best_match(a, {})
assert none_id is None and none_score == -1.0, "empty gallery handling failed"

# remove
assert s2.remove("alice") and not s2.remove("nope")
assert len(s2) == 1

os.remove(p)
print("STORE + MATCH UNIT TESTS PASSED")
