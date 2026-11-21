"""
Locked trajectory file schema (Protocol V1.3 section 5; G7).

Layout (one trajectory = one CSV + one JSON manifest):
    runs/<Category>/split_<data_seed>/<ARM>/seed_<r>.csv
    runs/<Category>/split_<data_seed>/<ARM>/seed_<r>.json
A trajectory is complete only when the CSV has epochs 1..120 and the
manifest says "complete": true (the trainer writes *.partial files and
renames them at the end).

CSV columns (exact order; floats written with 8 decimals, '%.8f'):
    epoch               int   1..120
    pv_ndcg10           float pseudo-validation NDCG@10 (selection signal)
    pv_recall10         float
    dev_ndcg10          float cold-DEV NDCG@10 (endpoint C_e)
    dev_recall10        float
    warmval_ndcg10      float warm-val NDCG@10 (RQ5)
    pv_noid_ndcg10      float PV NDCG@10 with the ID term switched off
    pv_items_id_active  int   PV items with active ID term at the end of the epoch
    bpr_loss_mean       float mean BPR loss over the epoch
    ma5_pv              float trailing MA5 of pv_ndcg10 up to this epoch
    sel_r1_epoch        int   epoch selected by R1 using epochs 1..e
    sel_r2_epoch        int   epoch selected by R2 using epochs 1..e

Manifest keys required by the analysis:
    domain, data_seed, arm, r, effective_seed, complete (true), n_epochs (120)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

import constants as C
import selection as SEL

COLUMNS = (
    "epoch", "pv_ndcg10", "pv_recall10", "dev_ndcg10", "dev_recall10",
    "warmval_ndcg10", "pv_noid_ndcg10", "pv_items_id_active", "bpr_loss_mean",
    "ma5_pv", "sel_r1_epoch", "sel_r2_epoch",
)
INT_COLUMNS = ("epoch", "pv_items_id_active", "sel_r1_epoch", "sel_r2_epoch")
METRIC_COLUMNS = ("pv_ndcg10", "pv_recall10", "dev_ndcg10", "dev_recall10",
                  "warmval_ndcg10", "pv_noid_ndcg10")
MANIFEST_REQUIRED = ("domain", "data_seed", "arm", "r", "effective_seed", "complete", "n_epochs")


class SchemaError(ValueError):
    pass


def traj_paths(root: Path, tag: str, data_seed: int, arm: str, r: int):
    d = Path(root) / C.DOMAINS[tag] / f"split_{data_seed}" / arm
    return d / f"seed_{r}.csv", d / f"seed_{r}.json"


def fmt(v: float) -> str:
    return f"{float(v):.{C.CSV_DECIMALS}f}"


def round8(v: float) -> float:
    """The value exactly as it will be read back from the CSV."""
    return float(fmt(v))


def format_row(row: Dict[str, float]) -> list:
    return [str(int(row[c])) if c in INT_COLUMNS else fmt(row[c]) for c in COLUMNS]


def write_csv_header(handle) -> None:
    csv.writer(handle).writerow(COLUMNS)


def read_trajectory(csv_path: Path, json_path: Path, expect=None) -> Dict[str, object]:
    """Reads and validates one trajectory; raises SchemaError on any violation."""
    csv_path, json_path = Path(csv_path), Path(json_path)
    if not csv_path.is_file() or not json_path.is_file():
        raise SchemaError(f"missing trajectory files for {csv_path}")
    with open(json_path, encoding="utf-8") as f:
        man = json.load(f)
    for k in MANIFEST_REQUIRED:
        if k not in man:
            raise SchemaError(f"{json_path}: manifest lacks '{k}'")
    if man["complete"] is not True or int(man["n_epochs"]) != C.EPOCHS:
        raise SchemaError(f"{json_path}: trajectory not complete")
    if expect:
        for k, v in expect.items():
            if man[k] != v:
                raise SchemaError(f"{json_path}: manifest {k}={man[k]!r}, expected {v!r}")
        exp_eff = C.effective_seed(man["domain"], man["data_seed"], man["r"])
        if int(man["effective_seed"]) != exp_eff:
            raise SchemaError(f"{json_path}: effective_seed {man['effective_seed']} != {exp_eff}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise SchemaError(f"{csv_path}: header differs from the locked schema")
    body = rows[1:]
    if len(body) != C.EPOCHS:
        raise SchemaError(f"{csv_path}: {len(body)} epochs, expected {C.EPOCHS}")
    data = {}
    for j, col in enumerate(COLUMNS):
        try:
            vals = [r[j] for r in body]
        except IndexError:
            raise SchemaError(f"{csv_path}: short row")
        if col in INT_COLUMNS:
            data[col] = np.array([int(v) for v in vals], dtype=np.int64)
        else:
            for v in vals:
                if len(v.split(".")[-1]) != C.CSV_DECIMALS:
                    raise SchemaError(f"{csv_path}: {col} value {v!r} not written with 8 decimals")
            data[col] = np.array([float(v) for v in vals], dtype=np.float64)
    if not np.array_equal(data["epoch"], np.arange(1, C.EPOCHS + 1)):
        raise SchemaError(f"{csv_path}: epochs are not 1..{C.EPOCHS}")
    for col in METRIC_COLUMNS:
        x = data[col]
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise SchemaError(f"{csv_path}: {col} outside [0, 1]")
    run = SEL.running_selection(data["pv_ndcg10"])
    if not np.array_equal(run["R1"], data["sel_r1_epoch"]):
        raise SchemaError(f"{csv_path}: logged sel_r1_epoch disagrees with R1 recomputed from PV")
    if not np.array_equal(run["R2"], data["sel_r2_epoch"]):
        raise SchemaError(f"{csv_path}: logged sel_r2_epoch disagrees with R2 recomputed from PV")
    if np.abs(np.array([round8(v) for v in run["MA5"]]) - data["ma5_pv"]).max() > 0:
        raise SchemaError(f"{csv_path}: logged ma5_pv disagrees with trailing MA5 of PV")
    return {"manifest": man, **data}
