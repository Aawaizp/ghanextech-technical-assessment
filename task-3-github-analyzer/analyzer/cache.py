"""Local CLI checkpoints containing public evidence and analysis, never keys."""

import hashlib
import json
from pathlib import Path
import time


class LocalCache:
    def __init__(self, directory, ttl=3600):
        self.directory = Path(directory)
        self.ttl = ttl

    def path(self, repository):
        digest = hashlib.sha256(repository.lower().encode()).hexdigest()
        return self.directory / (digest + ".json")

    def load(self, repository):
        path = self.path(repository)
        try:
            if path.stat().st_size > 5_000_000:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            if not 0 <= time.time() - data["saved_at"] < self.ttl:
                return None
            snapshot = data["snapshot"]
            if snapshot["repository"].lower() != repository.lower():
                return None
            if not isinstance(snapshot["evidence"], list) or not isinstance(data["progress"], dict):
                return None
            if not all(isinstance(item, dict) and {"id", "path", "text"} <= item.keys() for item in snapshot["evidence"]):
                return None
            return snapshot, data["progress"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def save(self, repository, snapshot, progress):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(repository)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "saved_at": time.time(), "snapshot": snapshot, "progress": progress,
        }, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
