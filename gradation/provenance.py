"""Portable streaming hashes and immutable run metadata."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(inputs, settings):
    return {"python": platform.python_version(), "settings": settings,
            "inputs": {str(p): sha256(p) for p in inputs},
            "versions": {n: importlib.metadata.version(n) for n in
                         ["numpy", "scipy", "pandas", "scikit-learn", "Pillow"]}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
