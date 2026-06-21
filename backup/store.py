"""
store.py
--------
In-RAM store of enrolled face embeddings, with JSON persistence.

Why in-RAM:  matching at check-in is then pure math over a small array
             (microseconds) -> the "no delay" requirement.
Why persist: so the demo survives a restart. In the real system the
             Spring Boot DB is the source of truth; this file is just a
             local cache / standalone fallback.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).with_name("embeddings.json")


class EmbeddingStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, np.ndarray] = {}
        self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}

    def _save(self) -> None:
        serializable = {k: v.tolist() for k, v in self._data.items()}
        self.path.write_text(json.dumps(serializable), encoding="utf-8")

    # ---- operations -----------------------------------------------------
    def add(self, person_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._data[person_id] = np.asarray(embedding, dtype=np.float32)
            self._save()

    def remove(self, person_id: str) -> bool:
        with self._lock:
            existed = self._data.pop(person_id, None) is not None
            if existed:
                self._save()
            return existed

    def all(self) -> dict[str, np.ndarray]:
        # Shallow copy of the mapping; vectors themselves are read-only in use.
        with self._lock:
            return dict(self._data)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)
