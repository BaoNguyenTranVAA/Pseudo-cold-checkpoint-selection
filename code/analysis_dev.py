"""
DEV analysis (Protocol V1.3 sections 5-6; DEVIATIONS C1-C9). Locked by G7.

    python analysis_dev.py --final         # the single locked DEV analysis (step 7)

The same functions are exercised by g7_selftest.py on synthetic trajectories
with known answers. Nothing here reads cold-TEST.

Outputs (in --out, never overwritten):
    per_trajectory.csv   every trajectory x selection rule x horizon
    units.csv            16 split effects d_{j,s} for every predeclared contrast
    results.json         confirmatory P1 -> P2, secondary estimates, A_ID,
                         descriptives per arm, RQ5 correlations
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import common as U
import constants as C
import selection as SEL
import stats_confirm as ST
import trajectory_schema as TS

CODE_FILES = ("constants.py", "common.py", "selection.py", "stats_confirm.py",
              "trajectory_schema.py", "analysis_dev.py")
Key = Tuple[str, int, str, int]          # (tag, data_seed, arm, r)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_runs(root: Path, arms=None) -> Tuple[Dict[Key, dict], List[str]]:
    arms = list(arms or C.ARMS)
    runs, problems = {}, []
    for tag, seed in ST.unit_order():
        for arm in arms:
            for r in C.TRAINING_SEEDS:
                cp, jp = TS.traj_paths(root, tag, seed, arm, r)
                if not cp.exists() and not jp.exists():
                    continue
                try:
                    runs[(tag, seed, arm, r)] = TS.read_trajectory(
                        cp, jp, expect={"domain": tag, "data_seed": seed, "arm": arm, "r": r})
                except TS.SchemaError as e:
                    problems.append(str(e))
    return runs, problems


# ------------------------------------------------------------------
# Per-trajectory quantities
# ------------------------------------------------------------------
def trajectory_rows(runs: Dict[Key, dict]) -> Dict[Tuple, dict]:
    out = {}
    for (tag, seed, arm, r), t in runs.items():
        pv, dev = t["pv_ndcg10"], t["dev_ndcg10"]
        e_warm = SEL.select(t["warmval_ndcg10"], "R1")
        for rule in C.SELECTION_RULES:
            for H in C.HORIZONS:
                g = SEL.regret(pv, dev, rule, H)
                row = {"domain": tag, "data_seed": seed, "arm": arm, "r": r,
                       "rule": rule, "horizon": H, **g}
                if rule == "R1" and H == C.PRIMARY_HORIZON:
                    e = g["e_star"]
                    row.update({
                        "spearman_pv_dev": SEL.spearman(pv, dev),
                        "early_spike": SEL.early_spike(pv),
                        "memorization_gap_at_e_star": float(pv[e - 1] - t["pv_noid_ndcg10"][e - 1]),
                        "warmval_epoch": e_warm,
                        "selection_lag_pv_vs_warmval": abs(e - e_warm),
                    })
                out[(tag, seed, arm, r, rule, H)] = row
    return out


def unit_effects(rows, arm_a: str, arm_b: str, rule: str = "R1",
                 horizon: int = C.PRIMARY_HORIZON) -> Optional[np.ndarray]:
    """d_{j,s} = mean_r (R_a - R_b) over the 5 seeds; None if anything is missing."""
    d = []
    for tag, seed in ST.unit_order():
        diffs = []
        for r in C.TRAINING_SEEDS:
            ka, kb = (tag, seed, arm_a, r, rule, horizon), (tag, seed, arm_b, r, rule, horizon)
            if ka not in rows or kb not in rows:
                return None
            diffs.append(rows[ka]["regret"] - rows[kb]["regret"])
        d.append(float(np.mean(diffs)))
    return np.array(d)


def seed_direction_count(rows, arm_a, arm_b, rule="R1", horizon=C.PRIMARY_HORIZON):
    n = pos = 0
    for tag, seed in ST.unit_order():
        for r in C.TRAINING_SEEDS:
            ka, kb = (tag, seed, arm_a, r, rule, horizon), (tag, seed, arm_b, r, rule, horizon)
            if ka in rows and kb in rows:
                n += 1
                pos += rows[ka]["regret"] > rows[kb]["regret"]
    return {"pairs": n, "pairs_positive": int(pos)}


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
def confirmatory(rows, test: str) -> dict:
    out = {"test": test, "rule": "R1", "horizon": C.PRIMARY_HORIZON, "family": "fixed sequence P1 -> P2"}
    p1_confirmed = False
    for name, a, b in C.CONFIRMATORY:
        d = unit_effects(rows, a, b)
        if d is None:
            out[name] = {"status": "incomplete: not all 80 (split, seed) pairs of both arms present"}
            continue
        res = ST.summarize_contrast(d, test)
        res["contrast"] = f"{a} - {b}"
        res.update(seed_direction_count(rows, a, b))
        if name == "P1":
            res["confirmatory"] = test != "estimate_only"
            res["confirmed"] = bool(res["confirmed_rule"]) if test != "estimate_only" else None
            p1_confirmed = bool(res["confirmed"])
        else:
            res["confirmatory"] = (test != "estimate_only") and p1_confirmed
            res["confirmed"] = bool(res["confirmed_rule"]) if res["confirmatory"] else None
            if not res["confirmatory"]:
                res["note"] = "descriptive only: P1 not confirmed (fixed sequence) or estimate-only lock"
        out[name] = res
    return out


def secondary(rows, test: str) -> dict:
    """Secondary estimates; p and CI use the method of the locked test (none if estimate_only)."""
    out = {}
    for name, a, b, rule, H in C.SECONDARY:
        d = unit_effects(rows, a, b, rule, H)
        if d is None:
            out[name] = {"status": "not available (arm missing or incomplete)"}
            continue
        res = ST.summarize_contrast(d, test)
        res.update({"contrast": f"{a} - {b}", "rule": rule, "horizon": H,
                    "label": "secondary estimate, not confirmatory"})
        res.update(seed_direction_count(rows, a, b, rule, H))
        out[name] = res
    for name, a, b in C.MA5_INTERACTIONS:
        d1, d2 = unit_effects(rows, a, b, "R1"), unit_effects(rows, a, b, "R2")
        if d1 is None or d2 is None:
            out[name] = {"status": "not available"}
            continue
        res = ST.summarize_contrast(d1 - d2, test)
        res.update({"contrast": f"({a}-{b} | R1) - ({a}-{b} | R2)",
                    "label": "secondary estimate, not confirmatory"})
        out[name] = res
    return out


def a_id(rows) -> dict:
    out = {}
    tags = [t for t, _ in ST.unit_order()[::ST.SPLITS_PER_DOMAIN]]
    for name, leaky, idoff, clean in C.A_ID_DEFS:
        dl, di = unit_effects(rows, leaky, clean), unit_effects(rows, idoff, clean)
        if dl is None or di is None:
            out[name] = {"status": "not available"}
            continue
        per = {}
        for i, t in enumerate(tags):
            sl = slice(i * 4, i * 4 + 4)
            ci = ST.t_ci(dl[sl])
            if ci["lo"] > 0:
                val = 1.0 - di[sl].mean() / dl[sl].mean()
                per[t] = {"A_ID": float(val), "delta_leaky": ci, "delta_idoff_mean": float(di[sl].mean()),
                          "label": ("substantial attenuation under ID disabling" if val > 0.5 else "")}
            else:
                per[t] = {"A_ID": "undetermined (t-CI of Delta_LEAKY not entirely above 0)",
                          "delta_leaky": ci}
        out[name] = {"definition": f"1 - ({idoff}-{clean}) / ({leaky}-{clean}) per domain", "domains": per}
    return out


def descriptives(rows) -> dict:
    out = {}
    for arm in C.ARMS:
        sel = [v for k, v in rows.items() if k[2] == arm and k[4] == "R1" and k[5] == C.PRIMARY_HORIZON]
        if not sel:
            continue
        g = lambda key: np.array([float(v[key]) for v in sel], dtype=float)
        out[arm] = {
            "n_trajectories": len(sel),
            "mean_regret": float(g("regret").mean()),
            "mean_normalized_regret": float(np.nanmean(g("normalized_regret"))),
            "mean_selected_dev": float(g("selected_cold").mean()),
            "mean_max_dev": float(g("max_cold").mean()),
            "mean_max_pv": float(g("max_pv").mean()),
            "mean_spearman_pv_dev": float(np.nanmean(g("spearman_pv_dev"))),
            "early_spike_rate": float(np.mean([bool(v["early_spike"]) for v in sel])),
            "median_e_star": float(np.median(g("e_star"))),
            "by_rule_mean_regret": {
                rule: float(np.mean([v["regret"] for k, v in rows.items()
                                     if k[2] == arm and k[4] == rule and k[5] == C.PRIMARY_HORIZON]))
                for rule in C.SELECTION_RULES},
        }
    return out


def rq5(rows) -> dict:
    from scipy.stats import spearmanr
    leaky = [a for a, s in C.ARMS.items() if s["k"] > 0 and s["fill"] != "cr"]
    reg, infl, gap, lag = [], [], [], []
    for k, v in rows.items():
        tag, seed, arm, r, rule, H = k
        if arm not in leaky or rule != "R1" or H != C.PRIMARY_HORIZON:
            continue
        clean = "B0" if C.ARMS[arm]["backbone"] == "B" else "A0"
        kc = (tag, seed, clean, r, rule, H)
        if kc not in rows:
            continue
        reg.append(v["regret"])
        infl.append(v["max_pv"] - rows[kc]["max_pv"])
        gap.append(v["memorization_gap_at_e_star"])
        lag.append(v["selection_lag_pv_vs_warmval"])
    if len(reg) < 3:
        return {"status": "not available"}
    f = lambda x: float(spearmanr(reg, x)[0]) if np.ptp(x) > 0 else float("nan")
    return {"n_trajectories": len(reg), "label": "descriptive (RQ5)",
            "spearman_regret_vs_pv_inflation": f(infl),
            "spearman_regret_vs_memorization_gap": f(gap),
            "spearman_regret_vs_selection_lag": f(lag)}


def run_analysis(runs_root: Path, out_dir: Path, test: str) -> dict:
    runs, problems = load_runs(runs_root)
    if problems:
        raise U.GateError("GATE FAIL: trajectory schema problems:\n  " + "\n  ".join(problems[:20]))
    rows = trajectory_rows(runs)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise U.GateError(f"GATE FAIL: {out_dir} is not empty; analysis outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["domain", "data_seed", "arm", "r", "rule", "horizon", "e_star", "regret",
            "normalized_regret", "selected_cold", "max_cold", "oracle_epoch", "max_pv",
            "spearman_pv_dev", "early_spike", "memorization_gap_at_e_star", "warmval_epoch",
            "selection_lag_pv_vs_warmval"]
    with open(out_dir / "per_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    unit_rows = []
    contrasts = [(n, a, b, "R1", C.PRIMARY_HORIZON) for n, a, b in C.CONFIRMATORY] + list(C.SECONDARY)
    for name, a, b, rule, H in contrasts:
        d = unit_effects(rows, a, b, rule, H)
        if d is not None:
            for (tag, seed), v in zip(ST.unit_order(), d):
                unit_rows.append({"contrast": name, "domain": tag, "data_seed": seed, "d": repr(float(v))})
    with open(out_dir / "units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "domain", "data_seed", "d"])
        w.writeheader()
        w.writerows(unit_rows)
    res = {
        "report_type": "dev_analysis",
        "protocol_version": C.PROTOCOL_VERSION,
        "n_trajectories": len(runs),
        "confirmatory": confirmatory(rows, test),
        "secondary": secondary(rows, test),
        "A_ID": a_id(rows),
        "descriptives_by_arm": descriptives(rows),
        "rq5": rq5(rows),
        "code_sha256": U.code_hashes(CODE_FILES),
        "created_utc": U.utc_now(),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=float)
    return res


def locked_test() -> str:
    lock = C.G7_DIR / "G7_LOCK.json"
    U.gate(lock.exists(), f"{lock} missing: G7 must pass before the DEV analysis")
    man = U.read_json(lock)
    for name, sha in man["code_sha256"].items():
        U.gate(U.file_sha256(Path(__file__).resolve().parent / name) == sha,
               f"{name} changed after the G7 lock")
    return man["locked_test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="run the single locked DEV analysis")
    a = ap.parse_args()
    if not a.final:
        ap.error("use --final (the G7 self-test calls run_analysis directly)")
    U.start_log("analysis_dev_final")
    test = locked_test()
    res = run_analysis(C.RUNS_DIR, C.ANALYSIS_DIR / "dev_final", test)
    for name in ("P1", "P2"):
        print(name, json.dumps({k: res["confirmatory"][name].get(k) for k in
                                ("T", "p", "n_domains_positive", "confirmatory", "confirmed")}))


if __name__ == "__main__":
    try:
        main()
    except U.GateError as e:
        print(f"\n{e}")
        raise SystemExit(2)
