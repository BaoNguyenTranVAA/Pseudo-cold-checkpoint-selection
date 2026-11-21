"""
Shared helpers: console/log tee, hashing, guarded writes, gates.

Hash conventions
----------------
array_sha256(a)   journal convention ("jv1"): SHA-256 over a header
                  "<dtype.str>|<shape>" and the C-contiguous bytes.
                  String arrays are hashed as UTF-8 text joined by "\\n".
file_sha256(p)    SHA-256 of the file bytes.
adc_fp16(a)       ADC-style 16-hex fingerprint, reproduced only to compare
                  with ADC records: sha256(str(numpy dtype)+str(shape)+bytes)[:16].
adc_sha_torch(a)  ADC replay-manifest style: sha256(str(torch dtype)+str(shape)+bytes).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

import constants as C


# ------------------------------------------------------------------
# Gates
# ------------------------------------------------------------------
class GateError(RuntimeError):
    """A validity gate failed. Stop, fix, log a deviation."""


def gate(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(f"GATE FAIL: {message}")


# ------------------------------------------------------------------
# Console + log file
# ------------------------------------------------------------------
class _Tee:
    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        self._handle.flush()

    def flush(self):
        self._stream.flush()
        self._handle.flush()


def start_log(script_name: str) -> Path:
    """Mirror stdout/stderr into logs/<script>_<UTC>.log (UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = C.LOG_DIR / f"{script_name}_{stamp}.log"
    handle = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    print(f"[log] {path}")
    print(f"[env] {json.dumps(env_info())}")
    return path


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_info() -> Dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    return info


# ------------------------------------------------------------------
# Hashing
# ------------------------------------------------------------------
def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.asarray(a)
    h = hashlib.sha256()
    if a.dtype.kind in ("U", "S", "O"):
        h.update(f"str|{a.shape}".encode("utf-8"))
        h.update("\n".join(str(x) for x in a.ravel().tolist()).encode("utf-8"))
    else:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}|{a.shape}".encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def adc_fp16(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()[:16]


_TORCH_DTYPE_NAMES = {
    np.dtype(np.int64): "torch.int64",
    np.dtype(np.float32): "torch.float32",
    np.dtype(np.bool_): "torch.bool",
}


def adc_sha_torch(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(_TORCH_DTYPE_NAMES[a.dtype].encode())
    h.update(str(tuple(int(s) for s in a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def item_order_fingerprint(item_ids, image_paths) -> str:
    """Feature-cache alignment fingerprint over ordered (item_id, image_path)."""
    h = hashlib.sha256()
    for item_id, image_path in zip(item_ids, image_paths):
        h.update(str(item_id).encode("utf-8"))
        h.update(b"\0")
        h.update(str(image_path).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def combined_sha256(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def code_hashes(names: Iterable[str]) -> Dict[str, str]:
    """SHA-256 of the given code files (relative to this folder)."""
    here = Path(__file__).resolve().parent
    return {n: file_sha256(here / n) for n in names}


# ------------------------------------------------------------------
# Guarded I/O (never overwrite a written artifact)
# ------------------------------------------------------------------
def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise GateError(
            f"GATE FAIL: {path} already exists; artifacts are never "
            "overwritten. Delete it by hand only if a deviation is logged."
        )


def write_json_new(path: Path, obj) -> None:
    path = Path(path)
    _refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz_new(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    _refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_npy_new(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    _refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def domain_name(tag: str) -> str:
    tag = tag.upper()
    if tag not in C.DOMAINS:
        raise SystemExit(f"unknown domain '{tag}'; use one of {list(C.DOMAINS)}")
    return C.DOMAINS[tag]


def core_dir(tag: str) -> Path:
    return C.DATA_DIR / domain_name(tag) / "core"


def split_dir(tag: str, data_seed: int) -> Path:
    return C.DATA_DIR / domain_name(tag) / f"split_{data_seed}"


def pair_keys(pairs: np.ndarray, num_items: int) -> np.ndarray:
    """(user, item) -> user * num_items + item, int64."""
    pairs = np.asarray(pairs, dtype=np.int64)
    return pairs[:, 0] * np.int64(num_items) + pairs[:, 1]
