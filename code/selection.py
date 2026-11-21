"""
Checkpoint-selection rules and regret (Protocol V1.3 section 5).

Single implementation shared by the trainer (running best states) and the
analysis scripts, so a saved state and an analysed e* cannot disagree.

Conventions
  * Inputs are the values logged in the trajectory CSV (rounded to 8
    decimals) as float64 arrays indexed by epoch 1..H (array position e-1).
  * Ties are broken by the EARLIEST epoch (np.argmax returns the first max).
  * Moving averages use math.fsum over the window values (exact,
    order-independent rounding), divided by the window length.
  * Every function returns 1-based epochs.

Rules
  R1            argmax_e PV_e                                     (primary)
  R2            trailing MA5: s_e = mean PV[max(1,e-4)..e]         (secondary, locked)
  MA5_full      trailing MA5 over full windows only (e >= 5)
  R1_from5      argmax over e >= 5
  MA3, MA9      trailing MA3 / MA9 (shorter windows at the start)
  MA5_centered  mean PV[max(1,e-2)..min(H,e+2)]
  ES10          early stopping, patience 10: stop at the first epoch e with
                e - best_epoch >= 10 (best = earliest strict maximum so far);
                e* = best_epoch at that time; if never triggered, e* = R1.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("selection input must be a non-empty finite 1-D array")
    return a


def trailing_ma(pv, window: int, full_only: bool = False) -> np.ndarray:
    """s_e for e = 1..H; with full_only, epochs e < window get -inf."""
    v = _as_array(pv).tolist()
    out = np.empty(len(v), dtype=np.float64)
    for e in range(len(v)):
        lo = max(0, e - window + 1)
        if full_only and e - lo + 1 < window:
            out[e] = -np.inf
        else:
            out[e] = math.fsum(v[lo:e + 1]) / (e - lo + 1)
    return out


def centered_ma(pv, half: int = 2) -> np.ndarray:
    v = _as_array(pv).tolist()
    n = len(v)
    out = np.empty(n, dtype=np.float64)
    for e in range(n):
        lo, hi = max(0, e - half), min(n, e + half + 1)
        out[e] = math.fsum(v[lo:hi]) / (hi - lo)
    return out


def argmax_epoch(score) -> int:
    s = np.asarray(score, dtype=np.float64)
    return int(np.argmax(s)) + 1


def early_stop_epoch(pv, patience: int = 10) -> int:
    v = _as_array(pv)
    best_val, best_e = -np.inf, 0
    for e in range(v.size):
        if v[e] > best_val:
            best_val, best_e = v[e], e
        elif e - best_e >= patience:
            return best_e + 1
    return argmax_epoch(v)


def select(pv, rule: str) -> int:
    pv = _as_array(pv)
    if rule == "R1":
        return argmax_epoch(pv)
    if rule == "R2":
        return argmax_epoch(trailing_ma(pv, 5))
    if rule == "MA5_full":
        return argmax_epoch(trailing_ma(pv, 5, full_only=True))
    if rule == "R1_from5":
        s = pv.copy()
        s[:4] = -np.inf
        return argmax_epoch(s)
    if rule == "MA3":
        return argmax_epoch(trailing_ma(pv, 3))
    if rule == "MA9":
        return argmax_epoch(trailing_ma(pv, 9))
    if rule == "MA5_centered":
        return argmax_epoch(centered_ma(pv, 2))
    if rule == "ES10":
        return early_stop_epoch(pv, 10)
    raise ValueError(f"unknown selection rule {rule}")


def regret(pv, cold, rule: str = "R1", horizon: int = None) -> Dict[str, float]:
    """Within-horizon regret R = max_{e<=H} C_e - C_{e*}, e* chosen on PV_{1..H}."""
    pv, cold = _as_array(pv), _as_array(cold)
    if pv.size != cold.size:
        raise ValueError("PV and cold series differ in length")
    H = pv.size if horizon is None else int(horizon)
    if not 1 <= H <= pv.size:
        raise ValueError(f"horizon {H} outside 1..{pv.size}")
    p, c = pv[:H], cold[:H]
    e_star = select(p, rule)
    c_max = float(c.max())
    c_sel = float(c[e_star - 1])
    return {
        "e_star": e_star,
        "regret": c_max - c_sel,
        "normalized_regret": (c_max - c_sel) / c_max if c_max > 0 else float("nan"),
        "selected_cold": c_sel,
        "max_cold": c_max,
        "oracle_epoch": int(np.argmax(c)) + 1,
        "max_pv": float(p.max()),
    }


def running_selection(pv) -> Dict[str, np.ndarray]:
    """Epoch selected by R1 and R2 using epochs 1..e only, for every e (logged by the trainer)."""
    pv = _as_array(pv)
    r1 = np.array([select(pv[:e], "R1") for e in range(1, pv.size + 1)])
    r2 = np.array([select(pv[:e], "R2") for e in range(1, pv.size + 1)])
    return {"R1": r1, "R2": r2, "MA5": trailing_ma(pv, 5)}


def early_spike(pv, early_max: int = 10, late_after: int = 20) -> bool:
    """argmax PV at epoch <= 10 and PV there > max PV over epochs > 20 (RQ3)."""
    pv = _as_array(pv)
    e = argmax_epoch(pv)
    return bool(e <= early_max and pv.size > late_after and pv[e - 1] > pv[late_after:].max())


def spearman(a, b) -> float:
    a, b = _as_array(a), _as_array(b)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b)[0])
