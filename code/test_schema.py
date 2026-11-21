"""
Frozen interfaces of the TEST stage (Protocol V1.3 section 7; TEST_ANALYSIS_PLAN.md).
Step B freezes this file together with analysis_test.py; steps C (test_dryrun.py) and D (open_test.py)
must write exactly these files / columns, and analysis_test.py reads only these.

Logical states per trajectory (1,200 trajectories):
    R1, R2                     every arm            -> 2,400 rows
    e001 ... e020, e030 ... e120 (30 grid states)  A0, A3, A5, A7 only (320 trajectories) -> 9,600 rows
    total 12,000 logical rows; physical files may coincide in epoch but are scored per logical row.

Files:
    analysis\\test_dryrun\\dryrun_scores.csv     step C, DEV-only re-scoring (DRYRUN_COLUMNS)
    analysis\\test_dryrun\\TEST_DRYRUN_REPORT.json
    journal_v1\\TEST_PLAN_FREEZE.json            step B freeze manifest (FREEZE_*), created and pinned BEFORE step D
    analysis\\test_open\\receipts\\01_PREPARED.json        step D, immutable (O_EXCL), chain start
    analysis\\test_open\\COLD_TEST_OPENED.json              step D, the O_EXCL opening marker = OPENED record
    analysis\\test_open\\receipts\\02r_RESUMED_<k>.json    step D, only after a reviewer RESUME_APPROVAL
    analysis\\test_open\\receipts\\03_SCORES_COMPLETE.json step D, immutable
    analysis\\test_open\\receipts\\04_ANALYSIS_COMPLETE.json, 05_LOCKED.json   after analysis_test.py --final
    analysis\\test_open\\test_scores.csv          step D, TEST scores (TEST_COLUMNS)
    analysis\\test_final\\                        analysis_test.py --final output (OUT_FILES)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import constants as C

GRID = tuple(C.GRID_EPOCHS)                       # 1..20, 30..120 (30 epochs)
GRID_ARMS = tuple(C.GRID_ARMS)                    # A0, A3, A5, A7
SELECT_RULES = ("R1", "R2")
HORIZON = 120


def grid_state(e: int) -> str:
    return f"e{int(e):03d}"


def logical_states(arm: str) -> List[str]:
    return list(SELECT_RULES) + ([grid_state(e) for e in GRID] if arm in GRID_ARMS else [])


def expected_keys() -> List[Tuple[str, int, str, int, str]]:
    """(domain, data_seed, arm, r, state) for all 12,000 logical rows, in canonical order."""
    tags = sorted(C.DOMAIN_INDEX, key=C.DOMAIN_INDEX.get)
    return [(t, s, a, r, st) for t in tags for s in C.DATA_SEEDS for a in C.ARMS
            for r in C.TRAINING_SEEDS for st in logical_states(a)]


N_LOGICAL = 1200 * 2 + 320 * 30                   # 12,000

KEY_COLUMNS = ("domain", "data_seed", "arm", "r", "state", "epoch", "state_sha256", "weights_sha256")
# state_sha256   = SHA-256 of the checkpoint FILE bytes (equals the 'sha256' recorded for that file in the trajectory
#                  manifest; step C checks this). R1.pt / R2.pt / eNNN.pt of the same epoch differ in bytes (the saved
#                  object carries its own 'kind'), so this is NOT a checkpoint identity.
# weights_sha256 = weights_fingerprint(state_dict) of the loaded checkpoint: the canonical identity of the weights.
#                  Logical rows of the same trajectory and the same epoch MUST have equal weights_sha256 (F5).
DRYRUN_METRICS = ("pv_ndcg10", "pv_recall10", "dev_ndcg10", "dev_recall10", "warmval_ndcg10", "pv_noid_ndcg10")
def weights_fingerprint(state_dict) -> str:
    """Canonical fingerprint of model weights, independent of file serialization: entries sorted by name; for each,
    'name|dtype|shape' then the raw little-endian C-order bytes. Accepts torch tensors or numpy arrays."""
    import hashlib
    import numpy as np
    h = hashlib.sha256()
    for k in sorted(state_dict):
        v = state_dict[k]
        if hasattr(v, "detach"):
            v = v.detach().cpu().contiguous().numpy()
        a = np.ascontiguousarray(np.asarray(v))
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        h.update(f"{k}|{a.dtype.newbyteorder('<').str}|{tuple(a.shape)}\n".encode("utf-8"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


DRYRUN_COLUMNS = KEY_COLUMNS + DRYRUN_METRICS + ("max_abs_diff_vs_log",)
TEST_METRICS = ("test_ndcg10", "test_recall10")
TEST_COLUMNS = KEY_COLUMNS + TEST_METRICS + ("test_n_users",)

DRYRUN_DIR = C.ANALYSIS_DIR / "test_dryrun"
DRYRUN_SCORES = DRYRUN_DIR / "dryrun_scores.csv"
DRYRUN_REPORT = DRYRUN_DIR / "TEST_DRYRUN_REPORT.json"
OPEN_DIR = C.ANALYSIS_DIR / "test_open"
RECEIPT = OPEN_DIR / "COLD_TEST_OPENED.json"
TEST_SCORES = OPEN_DIR / "test_scores.csv"
OUT_DIR = C.ANALYSIS_DIR / "test_final"
OUT_FILES = ("test_per_trajectory.csv", "test_units.csv", "test_results.json")

# ---------------------------------------------------------------- freeze manifest (step B)
FREEZE = C.PROJECT_ROOT / "TEST_PLAN_FREEZE.json"
FREEZE_TYPE = "TEST_PLAN_FREEZE"
FREEZE_FILES = ("analysis_test.py", "test_schema.py", "TEST_ANALYSIS_PLAN.md")
# {"manifest_type": FREEZE_TYPE, "protocol_version": "V1.3", "created_utc": str,
#  "files": {name: sha256 for FREEZE_FILES}, "dev_results_lock_sha256": str, "post_lock_addendum_sha256": str}
# Its SHA-256 is (a) recorded in POST_LOCK_AUDIT_LOG.md with the reviewer approval before step D,
# (b) passed by hand as --freeze-sha, (c) carried by 01_PREPARED.json. None of the three may be derived
# from the others at run time.

# ---------------------------------------------------------------- receipt chain (step D)
RECEIPT_DIR = OPEN_DIR / "receipts"
RECEIPT = OPEN_DIR / "COLD_TEST_OPENED.json"           # O_EXCL marker, written once before the first sealed read
R_PREPARED = RECEIPT_DIR / "01_PREPARED.json"
R_SCORES = RECEIPT_DIR / "03_SCORES_COMPLETE.json"
R_ANALYSIS = RECEIPT_DIR / "04_ANALYSIS_COMPLETE.json"
R_LOCKED = RECEIPT_DIR / "05_LOCKED.json"
RESUME_GLOB = "02r_RESUMED_*.json"
RECEIPT_STATES = ("PREPARED", "OPENED", "SCORES_COMPLETE", "ANALYSIS_COMPLETE", "LOCKED")
# every record: {"opening_id", "state", "utc", "prev_sha256"}; each file created with O_EXCL and never rewritten;
# prev_sha256 = SHA-256 of the previous record FILE (None for PREPARED). Order: PREPARED -> OPENED (marker)
# -> RESUMED_1 -> ... -> SCORES_COMPLETE -> ANALYSIS_COMPLETE -> LOCKED.
PREPARED_FIELDS = ("freeze_manifest_sha256", "dev_results_lock_sha256", "dryrun_scores_sha256",
                   "dryrun_report_sha256", "expected_sealed_sha256", "scorer_code_sha256")
RESUMED_FIELDS = ("resume_approval_file", "resume_approval_sha256", "reason")
SCORES_FIELDS = ("test_scores_sha256", "test_users_by_split", "n_rows")
ANALYSIS_FIELDS = ("output_sha256", "analysis_code_sha256", "freeze_manifest_sha256", "recovered_with_approval")
LOCKED_FIELDS = ("output_sha256", "analysis_complete_sha256")
# 04_ANALYSIS_COMPLETE is written by analysis_test.py --final right after the three outputs (or, after a crash
# between the two, by --finalize-receipt with a reviewer approval, which re-computes the outputs and requires
# byte-identical files). 05_LOCKED is written by analysis_test.py --lock after re-verifying the outputs.


def file_sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(p: Path) -> dict:
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def split_tag(tag: str, seed: int) -> str:
    return f"{tag}-{int(seed)}"


def verify_freeze(expected_sha: str, code_dir: Path) -> dict:
    """The freeze manifest is the approved one (hand-supplied SHA + append-only audit log) and still matches."""
    if not FREEZE.is_file():
        _err(f"{FREEZE} missing")
    got = file_sha(FREEZE)
    if got != expected_sha:
        _err(f"TEST_PLAN_FREEZE.json SHA-256 {got[:16]}... != --freeze-sha {expected_sha[:16]}...")
    audit = C.PROJECT_ROOT / "POST_LOCK_AUDIT_LOG.md"
    if not audit.is_file() or expected_sha not in audit.read_text(encoding="utf-8"):
        _err("the freeze manifest SHA-256 is not recorded in POST_LOCK_AUDIT_LOG.md")
    m = _load(FREEZE)
    if m.get("manifest_type") != FREEZE_TYPE or set(m.get("files", {})) != set(FREEZE_FILES):
        _err("TEST_PLAN_FREEZE.json: wrong type or file list")
    actual = {"analysis_test.py": file_sha(code_dir / "analysis_test.py"),
              "test_schema.py": file_sha(code_dir / "test_schema.py"),
              "TEST_ANALYSIS_PLAN.md": file_sha(C.PROJECT_ROOT / "TEST_ANALYSIS_PLAN.md")}
    for n, v in actual.items():
        if m["files"][n] != v:
            _err(f"{n} SHA-256 differs from the frozen manifest")
    add = C.PROJECT_ROOT / "POST_LOCK_ADDENDUM.md"
    if not add.is_file() or file_sha(add) != m.get("post_lock_addendum_sha256"):
        _err("POST_LOCK_ADDENDUM.md SHA-256 differs from the frozen manifest")
    return m


def create_record(path: Path, obj: dict) -> str:
    """Creates an immutable record exactly once: write + fsync a private temp file (O_EXCL), then hard-link it
    to the final name (fails if the name exists, never overwrites; the record appears complete or not at all),
    then remove the temp name. Returns the SHA-256 of the record."""
    import json, os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _err(f"{path.name} already exists (records are written once)")
    tmp = path.with_name(path.name + ".tmp")
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
    except FileExistsError:
        _err(f"{path.name} already exists (records are written once)")
    finally:
        os.remove(tmp)
    return file_sha(path)


_CHAIN_ORDER = ("SCORES_COMPLETE", "ANALYSIS_COMPLETE", "LOCKED")


def verify_chain(freeze_sha: str, through: str = "SCORES_COMPLETE") -> dict:
    """Validates the immutable receipt chain PREPARED -> OPENED -> RESUMED* -> SCORES_COMPLETE [-> ANALYSIS_COMPLETE
    [-> LOCKED]] up to `through`; records after `through` must be absent."""
    if through not in _CHAIN_ORDER:
        _err(f"unknown chain end {through}")
    if not R_PREPARED.is_file() or not RECEIPT.is_file():
        _err("receipt chain incomplete: 01_PREPARED.json and COLD_TEST_OPENED.json are required (step D)")
    resumes = sorted(RECEIPT_DIR.glob(RESUME_GLOB), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    files = [R_PREPARED, RECEIPT] + resumes
    tail = {"SCORES_COMPLETE": R_SCORES, "ANALYSIS_COMPLETE": R_ANALYSIS, "LOCKED": R_LOCKED}
    k = _CHAIN_ORDER.index(through)
    for st in _CHAIN_ORDER[:k + 1]:
        if not tail[st].is_file():
            _err(f"{tail[st].name} missing: chain does not reach {st}")
        files.append(tail[st])
    later = [tail[st].name for st in _CHAIN_ORDER[k + 1:] if tail[st].exists()]
    if later:
        _err(f"{later} already recorded: this step runs once (chain must end at {through})")
    if list(RECEIPT_DIR.glob("*.tmp")):
        _err("a partial record (*.tmp) exists in receipts: stop and ask the reviewer")
    recs = [_load(p) for p in files]
    want = ["PREPARED", "OPENED"] + ["RESUMED"] * len(resumes) + list(_CHAIN_ORDER[:k + 1])
    if [r.get("state") for r in recs] != want:
        _err(f"receipt states {[r.get('state') for r in recs]} != {want}")
    oid = recs[0].get("opening_id")
    prev = None
    for p, r in zip(files, recs):
        if not oid or r.get("opening_id") != oid:
            _err(f"{p.name}: opening_id differs")
        if r.get("prev_sha256") != prev:
            _err(f"{p.name}: prev_sha256 breaks the chain")
        prev = file_sha(p)
    utcs = [r.get("utc", "") for r in recs]
    if utcs != sorted(utcs) or not all(utcs):
        _err("receipt timestamps missing or not in order")
    pr = recs[0]
    if not all(f in pr for f in PREPARED_FIELDS):
        _err("01_PREPARED.json lacks required fields")
    if pr["freeze_manifest_sha256"] != freeze_sha:
        _err("01_PREPARED.json does not carry the approved freeze manifest SHA-256")
    fz = _load(FREEZE)
    if not fz.get("created_utc") or fz["created_utc"] >= pr["utc"]:
        _err("freeze manifest is not older than the PREPARED record")
    for p, r in zip(resumes, recs[2:2 + len(resumes)]):
        if not all(f in r for f in RESUMED_FIELDS):
            _err(f"{p.name}: lacks resume fields")
        ap = OPEN_DIR / r["resume_approval_file"]
        if not ap.is_file() or file_sha(ap) != r["resume_approval_sha256"]:
            _err(f"{p.name}: reviewer RESUME_APPROVAL file missing or changed")
        if _load(ap).get("opening_id") != oid:
            _err(f"{p.name}: RESUME_APPROVAL is for another opening")
    sc = recs[len(files) - (k + 1)]
    if not all(f in sc for f in SCORES_FIELDS):
        _err("03_SCORES_COMPLETE.json lacks required fields")
    for st, need in (("ANALYSIS_COMPLETE", ANALYSIS_FIELDS), ("LOCKED", LOCKED_FIELDS)):
        if st in want:
            r = recs[want.index(st)]
            if not all(f in r for f in need):
                _err(f"{tail[st].name} lacks required fields")
    return {"opening_id": oid, "records": dict(zip([p.name for p in files], recs)),
            "record_sha256": {p.name: file_sha(p) for p in files}, "n_resumes": len(resumes)}


TOL_SAME_STATE = 1e-9          # R1/R2 state vs grid state of the same epoch (same weights, same scorer)
TOL_DEV_VS_LOG = 1e-6          # dry-run DEV score vs the locked per_trajectory / CSV log


class SchemaError(Exception):
    pass


def _err(msg):
    raise SchemaError(f"GATE FAIL: {msg}")


def read_scores(path: Path, kind: str) -> Dict[Tuple, dict]:
    """Strict reader for dryrun_scores.csv (kind='dryrun') or test_scores.csv (kind='test')."""
    cols = DRYRUN_COLUMNS if kind == "dryrun" else TEST_COLUMNS
    metrics = DRYRUN_METRICS if kind == "dryrun" else TEST_METRICS
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if tuple(header or ()) != cols:
            _err(f"{path.name}: header {header} != {list(cols)}")
        out = {}
        for n, row in enumerate(rd, start=2):
            if len(row) != len(cols):
                _err(f"{path.name} line {n}: {len(row)} fields")
            d = dict(zip(cols, row))
            try:
                key = (d["domain"], int(d["data_seed"]), d["arm"], int(d["r"]), d["state"])
                rec = {"epoch": int(d["epoch"]), "state_sha256": d["state_sha256"], "weights_sha256": d["weights_sha256"]}
                for m in metrics:
                    rec[m] = float(d[m])
                if kind == "dryrun":
                    rec["max_abs_diff_vs_log"] = float(d["max_abs_diff_vs_log"])
                else:
                    rec["test_n_users"] = int(d["test_n_users"])
            except ValueError as e:
                _err(f"{path.name} line {n}: {e}")
            for m in metrics:
                if not (math.isfinite(rec[m]) and 0.0 <= rec[m] <= 1.0):
                    _err(f"{path.name} line {n}: {m} = {rec[m]} not finite in [0, 1]")
            if kind == "dryrun":
                x = rec["max_abs_diff_vs_log"]
                if not (math.isfinite(x) and 0.0 <= x <= TOL_DEV_VS_LOG):
                    _err(f"{path.name} line {n}: max_abs_diff_vs_log = {x} not in [0, {TOL_DEV_VS_LOG}]")
            elif rec["test_n_users"] <= 0:
                _err(f"{path.name} line {n}: test_n_users = {rec['test_n_users']} <= 0")
            for hk in ("state_sha256", "weights_sha256"):
                if len(rec[hk]) != 64 or any(c not in "0123456789abcdef" for c in rec[hk]):
                    _err(f"{path.name} line {n}: bad {hk}")
            st = key[4]
            if st.startswith("e") and rec["epoch"] != int(st[1:]):
                _err(f"{path.name} line {n}: grid state {st} has epoch {rec['epoch']}")
            if not 1 <= rec["epoch"] <= HORIZON:
                _err(f"{path.name} line {n}: epoch {rec['epoch']} outside 1..{HORIZON}")
            if key in out:
                _err(f"{path.name}: duplicate row {key}")
            out[key] = rec
    exp = expected_keys()
    if set(out) != set(exp):
        missing = sorted(set(exp) - set(out))[:5]
        extra = sorted(set(out) - set(exp))[:5]
        _err(f"{path.name}: {len(out)} rows != {len(exp)} expected; missing e.g. {missing}; extra e.g. {extra}")
    if kind == "test":
        users = {}
        for k, rec in out.items():
            users.setdefault(split_tag(k[0], k[1]), set()).add(rec["test_n_users"])
        varying = {sp: sorted(v) for sp, v in users.items() if len(v) != 1}
        if varying:
            _err(f"{path.name}: test_n_users varies within a split: {dict(list(varying.items())[:3])}")
    return out
