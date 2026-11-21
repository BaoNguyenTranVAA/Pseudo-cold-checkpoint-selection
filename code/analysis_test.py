"""
TEST analysis (Protocol V1.3 section 7; TEST_ANALYSIS_PLAN.md). Frozen in step B, before any sealed read.

    python analysis_test.py --final --freeze-sha <SHA>
        once, only when the immutable step-D receipt chain ends in SCORES_COMPLETE; writes the three outputs, then
        receipts\\04_ANALYSIS_COMPLETE.json (output SHA-256, code SHA-256, prev link)
    python analysis_test.py --lock --freeze-sha <SHA>
        once, after ANALYSIS_COMPLETE: re-verifies the chain and the outputs, writes receipts\\05_LOCKED.json
    python analysis_test.py --finalize-receipt --approval <RESUME_APPROVAL file in analysis\\test_open> --freeze-sha <SHA>
        ONLY after a crash between the outputs and 04 and with a reviewer approval: re-computes the analysis into
        analysis\\test_final_recheck, requires byte-identical outputs, then writes 04 marked as recovered
    <SHA> = SHA-256 of TEST_PLAN_FREEZE.json, as approved and recorded in POST_LOCK_AUDIT_LOG.md

Inputs (never anything under sealed\\):
    analysis\\dev_final\\per_trajectory.csv     locked DEV output (e* of R1 / R2 at H = 120); hash-checked
    analysis\\test_dryrun\\dryrun_scores.csv    step C (PV / DEV re-scored at every logical state)
    analysis\\test_open\\test_scores.csv        step D (TEST NDCG@10 / Recall@10 at every logical state)
Outputs: analysis\\test_final\\{test_per_trajectory.csv, test_units.csv, test_results.json}, never overwritten.

Primary endpoint (Protocol section 7, verbatim definition): grid-restricted TEST regret
    GR = max_{e in G u {e*}} TEST(e) - TEST(e*),   G = {1..20} u {30, 40, ..., 120},  e* = R1 at H = 120,
computed for A0, A3, A5, A7. GR >= 0 by construction; it is a LOWER BOUND of the full TEST regret.
P1 (A3 - A0) then P2 (A3 - A5) on 16 split units with the locked section-6 procedure (stats_confirm, the
G7-locked test), fixed sequence; label "precommitted sealed-TEST replication informed by DEV".
Everything else is secondary (R2 version, selected TEST NDCG, grid rho) or the precommitted exploratory
list E1-E4 (21 TEST contrasts, one Holm family), plus the DEV-only grid calibration E5 (2 contrasts, reported
apart, never mixed into TEST p-values). Note: each GR is a lower bound of that arm's full TEST regret, but a
DIFFERENCE of two GRs (P1, P2) is not a lower bound of the difference of full regrets.
Nothing here changes the confirmatory DEV conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import common as U
import constants as C
import selection as SEL
import stats_confirm as ST
import test_schema as S

CODE_FILES = ("analysis_test.py", "test_schema.py", "stats_confirm.py", "selection.py",
              "constants.py", "common.py", "analysis_dev.py")
EXPECTED_DEV_LOCK_SHA = "0775d9fb819ce0f0bf85333a3fc801d1cd0265c344ae47457384066cd980ab3b"
PLAN_NAME = "TEST_ANALYSIS_PLAN.md"
TKey = Tuple[str, int, str, int]


class GateFail(Exception):
    pass


def gate(ok, msg):
    if not ok:
        raise GateFail(f"GATE FAIL: {msg}")


# ------------------------------------------------------------------ loading
def load_selection(per_traj: Path) -> Dict[TKey, dict]:
    """e*, selected DEV NDCG and full DEV regret for R1 / R2 at H = 120, from the locked DEV output."""
    out: Dict[TKey, dict] = {}
    with open(per_traj, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["rule"] not in S.SELECT_RULES or int(row["horizon"]) != S.HORIZON:
                continue
            k = (row["domain"], int(row["data_seed"]), row["arm"], int(row["r"]))
            rec = out.setdefault(k, {})
            gate(row["rule"] not in rec, f"per_trajectory: duplicate {k} {row['rule']}")
            rec[row["rule"]] = {"e_star": int(row["e_star"]), "selected_cold": float(row["selected_cold"]),
                                "regret": float(row["regret"])}
    exp = {k[:4] for k in S.expected_keys()}
    complete = sum(set(v) == set(S.SELECT_RULES) for v in out.values())
    gate(set(out) == exp and complete == len(exp),
         f"per_trajectory: {len(out)} trajectories at H120, {complete} with both R1 and R2; expected {len(exp)}")
    return out


def _same_state(dry, test, k1, k2, what):
    """Two logical rows of the same trajectory and epoch must be the same checkpoint: equal canonical weight
    fingerprint (F5; file bytes legitimately differ between R1.pt / R2.pt / eNNN.pt), and equal scores on all
    2 TEST + 6 dry-run metrics."""
    gate(dry[k1]["weights_sha256"] == dry[k2]["weights_sha256"] and test[k1]["weights_sha256"] == test[k2]["weights_sha256"],
         f"{what}: weights_sha256 differs between {k1[4]} and {k2[4]} (not the same checkpoint)")
    for m in S.TEST_METRICS:
        gate(abs(test[k1][m] - test[k2][m]) <= S.TOL_SAME_STATE, f"{what}: TEST {m} differs between {k1[4]} and {k2[4]}")
    for m in S.DRYRUN_METRICS:
        gate(abs(dry[k1][m] - dry[k2][m]) <= S.TOL_SAME_STATE, f"{what}: dry-run {m} differs between {k1[4]} and {k2[4]}")


def check_identity(sel, dry, test):
    """Cross-checks between the locked DEV selection, the dry-run and the TEST scores."""
    for key, d in dry.items():
        t = test[key]
        gate(t["state_sha256"] == d["state_sha256"] and t["weights_sha256"] == d["weights_sha256"]
             and t["epoch"] == d["epoch"],
             f"{key}: state hash / weights fingerprint / epoch differ between dry-run and TEST scores")
    for tk, rules in sel.items():
        arm = tk[2]
        for rule in S.SELECT_RULES:
            e = rules[rule]["e_star"]
            d = dry[tk + (rule,)]
            gate(d["epoch"] == e, f"{tk} {rule}: state epoch {d['epoch']} != locked e* {e}")
            gate(abs(d["dev_ndcg10"] - rules[rule]["selected_cold"]) <= S.TOL_DEV_VS_LOG,
                 f"{tk} {rule}: dry-run DEV NDCG differs from the locked selected_cold")
            if rule == "R2" and rules["R1"]["e_star"] == e:
                _same_state(dry, test, tk + ("R1",), tk + ("R2",), f"{tk}: R1 and R2 select epoch {e}")
            if arm in S.GRID_ARMS and e in S.GRID:
                _same_state(dry, test, tk + (rule,), tk + (S.grid_state(e),),
                            f"{tk}: {rule} state and grid state of epoch {e}")


# ------------------------------------------------------------------ per trajectory
def grid_regret(scores_at_grid: Dict[int, float], e_star: int, at_star: float) -> Tuple[float, int]:
    """max over G u {e*} minus value at e*; returns (regret >= 0, argmax epoch, earliest on ties)."""
    cand = dict(scores_at_grid)
    cand[e_star] = at_star
    best_e = min(cand, key=lambda e: (-cand[e], e))
    return cand[best_e] - at_star, best_e


def trajectory_table(sel, dry, test) -> Dict[TKey, dict]:
    rows = {}
    for tk, rules in sel.items():
        arm = tk[2]
        r = {"domain": tk[0], "data_seed": tk[1], "arm": arm, "r": tk[3]}
        for rule in S.SELECT_RULES:
            e = rules[rule]["e_star"]
            t_sel = test[tk + (rule,)]["test_ndcg10"]
            r[f"e_star_{rule}"] = e
            r[f"sel_test_{rule}"] = t_sel
            r[f"sel_test_recall_{rule}"] = test[tk + (rule,)]["test_recall10"]
            r[f"sel_dev_{rule}"] = dry[tk + (rule,)]["dev_ndcg10"]
            r[f"dev_full_regret_{rule}"] = rules[rule]["regret"]
            if arm in S.GRID_ARMS:
                tg = {e_: test[tk + (S.grid_state(e_),)]["test_ndcg10"] for e_ in S.GRID}
                dg = {e_: dry[tk + (S.grid_state(e_),)]["dev_ndcg10"] for e_ in S.GRID}
                gr, ge = grid_regret(tg, e, t_sel)
                gd, _ = grid_regret(dg, e, dry[tk + (rule,)]["dev_ndcg10"])
                gate(gr >= 0 and gd >= 0, f"{tk}: negative grid regret")
                r[f"test_grid_regret_{rule}"] = gr
                r[f"test_grid_max_{rule}"] = t_sel + gr
                r[f"test_grid_oracle_epoch_{rule}"] = ge
                r[f"dev_grid_regret_{rule}"] = gd
        if arm in S.GRID_ARMS:
            pv = [dry[tk + (S.grid_state(e_),)]["pv_ndcg10"] for e_ in S.GRID]
            tv = [test[tk + (S.grid_state(e_),)]["test_ndcg10"] for e_ in S.GRID]
            r["rho_grid_pv_test"] = SEL.spearman(pv, tv)
        rows[tk] = r
    return rows


# ------------------------------------------------------------------ contrasts
def units(rows, a, b, col) -> np.ndarray:
    """16 split effects d_{j,s} = mean over the 5 training seeds of (col[a] - col[b]), locked unit order."""
    d = []
    for tag, seed in ST.unit_order():
        diffs = [rows[(tag, seed, a, r)][col] - rows[(tag, seed, b, r)][col] for r in C.TRAINING_SEEDS]
        gate(all(math.isfinite(x) for x in diffs), f"non-finite {col} difference {a}-{b} {tag} {seed}")
        d.append(float(np.mean(diffs)))
    return np.asarray(d)


def pairs_positive(rows, a, b, col):
    return int(sum(rows[(t, s, a, r)][col] - rows[(t, s, b, r)][col] > 0
                   for t, s in ST.unit_order() for r in C.TRAINING_SEEDS))


def summary(d, test, label, extra=None):
    res = ST.summarize_contrast(d, test)
    res["label"] = label
    res.update(extra or {})
    return res


# Exploratory family (TEST_ANALYSIS_PLAN.md section 5): name -> (spec, expected sign)
# expected sign = sign of the DEV point estimate of the analogous contrast (DEV full regret for grid-regret
# contrasts, DEV selected NDCG for selected-TEST contrasts); a sign, not a hypothesis test.
#   spec ("diff", a, b, col)            d = col[a] - col[b]
#   spec ("inter", a, b, col1, col2)    d = (col1[a]-col1[b]) - (col2[a]-col2[b])
EXPLORATORY = {
    "E1a_selected_loss_A0_minus_A3_R1": (("diff", "A0", "A3", "sel_test_R1"), +1),
    "E1b_grid_ceiling_loss_A0_minus_A3_R1": (("diff", "A0", "A3", "test_grid_max_R1"), +1),
    "E1c_selected_loss_A5_minus_A3_R1": (("diff", "A5", "A3", "sel_test_R1"), +1),
    "E1d_grid_ceiling_loss_A5_minus_A3_R1": (("diff", "A5", "A3", "test_grid_max_R1"), +1),
    "E2a_P1_R1_minus_R2": (("inter", "A3", "A0", "test_grid_regret_R1", "test_grid_regret_R2"), +1),
    "E2b_P2_R1_minus_R2": (("inter", "A3", "A5", "test_grid_regret_R1", "test_grid_regret_R2"), +1),
    "E3a_A3_minus_A7_R1": (("diff", "A3", "A7", "test_grid_regret_R1"), +1),
    "E3b_A3_minus_A7_R2": (("diff", "A3", "A7", "test_grid_regret_R2"), +1),
    "E3c_A7_minus_A0_R1": (("diff", "A7", "A0", "test_grid_regret_R1"), +1),
    "E3d_A7_minus_A0_R2": (("diff", "A7", "A0", "test_grid_regret_R2"), +1),
    "E4a_selected_loss_A0_minus_A8_R1": (("diff", "A0", "A8", "sel_test_R1"), +1),
    "E4b_selected_loss_A0_minus_A9_R1": (("diff", "A0", "A9", "sel_test_R1"), +1),
    "E4c_selected_loss_A0_minus_A10_R1": (("diff", "A0", "A10", "sel_test_R1"), +1),
    "E4d_selected_loss_A0_minus_A2_R1": (("diff", "A0", "A2", "sel_test_R1"), +1),
    "E4e_selected_loss_A0_minus_A8_R2": (("diff", "A0", "A8", "sel_test_R2"), +1),
    "E4f_selected_loss_A0_minus_A9_R2": (("diff", "A0", "A9", "sel_test_R2"), +1),
    "E4g_selected_loss_A0_minus_A10_R2": (("diff", "A0", "A10", "sel_test_R2"), +1),
    "E4h_selected_loss_A0_minus_A2_R2": (("diff", "A0", "A2", "sel_test_R2"), +1),
    "E4i_A2_minus_A8_selected_R1": (("diff", "A2", "A8", "sel_test_R1"), +1),
    "E4j_A2_minus_A8_selected_R2": (("diff", "A2", "A8", "sel_test_R2"), -1),
    "E4k_coverage_rule_interaction": (("inter", "A2", "A8", "sel_test_R1", "sel_test_R2"), +1),
}
# DEV-only grid calibration (post-DEV-lock, exploratory). Not TEST, not in the Holm family above.
DEV_CALIBRATION = {
    "E5a_DEV_grid_P1_R1": (("diff", "A3", "A0", "dev_grid_regret_R1"), +1),
    "E5b_DEV_grid_P2_R1": (("diff", "A3", "A5", "dev_grid_regret_R1"), +1),
}
DEV_FULL_REFERENCE = {"E5a_DEV_grid_P1_R1": ("A3", "A0"), "E5b_DEV_grid_P2_R1": ("A3", "A5")}


def exploratory_d(rows, spec):
    if spec[0] == "diff":
        return units(rows, spec[1], spec[2], spec[3])
    _, a, b, c1, c2 = spec
    return units(rows, a, b, c1) - units(rows, a, b, c2)


def holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------ main analysis
def run_analysis(per_traj: Path, dryrun_csv: Path, test_csv: Path, out_dir: Path, test: str) -> dict:
    gate(test in ("signflip", "ttest", "estimate_only"), f"unknown locked test {test}")
    out_dir = Path(out_dir)
    gate(not out_dir.exists() or not any(out_dir.iterdir()), f"{out_dir} is not empty (single run)")
    sel = load_selection(per_traj)
    try:
        dry = S.read_scores(Path(dryrun_csv), "dryrun")
        tst = S.read_scores(Path(test_csv), "test")
    except S.SchemaError as e:
        raise GateFail(str(e))
    check_identity(sel, dry, tst)
    rows = trajectory_table(sel, dry, tst)

    lab_c = "precommitted sealed-TEST replication informed by DEV (grid-restricted regret, lower bound)"
    res = {"report_type": "test_analysis", "protocol_version": C.PROTOCOL_VERSION, "test": test,
           "endpoint": "grid-restricted TEST regret = max_{G u {e*}} TEST - TEST(e*), e* = R1 at H120",
           "grid": list(S.GRID), "n_trajectories": len(rows), "n_logical_states": len(dry)}

    # primary: fixed sequence P1 -> P2 (R1)
    prim = {}
    for name, (a, b) in (("P1", ("A3", "A0")), ("P2", ("A3", "A5"))):
        d = units(rows, a, b, "test_grid_regret_R1")
        prim[name] = summary(d, test, lab_c, {"contrast": f"{a} - {b}",
                                              "pairs_positive": pairs_positive(rows, a, b, "test_grid_regret_R1"),
                                              "pairs": 80, "units": d.tolist()})
    if test == "estimate_only":
        prim["P1"]["replicated"] = prim["P2"]["replicated"] = None
        prim["P2"]["confirmatory"] = False
    else:
        prim["P1"]["replicated"] = bool(prim["P1"]["confirmed_rule"])
        prim["P2"]["confirmatory"] = prim["P1"]["replicated"]
        prim["P2"]["replicated"] = bool(prim["P2"]["confirmed_rule"]) if prim["P1"]["replicated"] else None
    res["primary"] = {**prim, "family": "fixed sequence P1 -> P2", "rule": "R1", "horizon": S.HORIZON}

    # secondary
    sec = {}
    for name, (a, b) in (("P1_under_R2", ("A3", "A0")), ("P2_under_R2", ("A3", "A5"))):
        d = units(rows, a, b, "test_grid_regret_R2")
        sec[name] = summary(d, test, "secondary estimate, not confirmatory", {"contrast": f"{a} - {b}"})
    arms_desc = {}
    for arm in C.ARMS:
        rs = [v for k, v in rows.items() if k[2] == arm]
        dsc = {f"mean_sel_test_{x}": float(np.mean([v[f"sel_test_{x}"] for v in rs])) for x in S.SELECT_RULES}
        dsc.update({f"mean_sel_test_recall_{x}": float(np.mean([v[f"sel_test_recall_{x}"] for v in rs]))
                    for x in S.SELECT_RULES})
        dsc["by_domain_sel_test_R1"] = {t: float(np.mean([v["sel_test_R1"] for v in rs if v["domain"] == t]))
                                        for t in sorted(C.DOMAIN_INDEX, key=C.DOMAIN_INDEX.get)}
        if arm in S.GRID_ARMS:
            for x in S.SELECT_RULES:
                dsc[f"mean_test_grid_regret_{x}"] = float(np.mean([v[f"test_grid_regret_{x}"] for v in rs]))
                dsc[f"mean_dev_grid_regret_{x}"] = float(np.mean([v[f"dev_grid_regret_{x}"] for v in rs]))
                dsc[f"mean_dev_full_regret_{x}"] = float(np.mean([v[f"dev_full_regret_{x}"] for v in rs]))
            rho = [v["rho_grid_pv_test"] for v in rs if math.isfinite(v["rho_grid_pv_test"])]
            dsc["mean_rho_grid_pv_test"] = float(np.mean(rho)) if rho else None
            dsc["n_rho_finite"] = len(rho)
        arms_desc[arm] = dsc
    sec["descriptives_by_arm"] = arms_desc
    sec["label"] = "secondary / descriptive"
    res["secondary"] = sec

    # exploratory, precommitted
    expl, ps = {}, []
    for name, (spec, sign) in EXPLORATORY.items():
        d = exploratory_d(rows, spec)
        r = summary(d, test, "exploratory (precommitted before TEST)",
                    {"spec": list(spec), "dev_informed_expected_sign": sign})
        if r.get("p") is not None:
            r["direction_as_expected"] = (None if sign == 0 else bool(np.sign(r["T"]) == sign))
            ps.append((name, r["p"]))
        expl[name] = r
    if ps:
        for (name, _), a in zip(ps, holm([p for _, p in ps])):
            expl[name]["p_holm_within_exploratory_family"] = float(a)
    res["exploratory"] = {"label": "exploratory on sealed TEST, precommitted; Holm within these 21 contrasts; "
                                   "expected signs are from DEV point estimates, tests stay two-sided",
                          "n_contrasts": len(expl), "contrasts": expl}

    cal = {}
    for name, (spec, sign) in DEV_CALIBRATION.items():
        d = exploratory_d(rows, spec)
        full = units(rows, *DEV_FULL_REFERENCE[name], "dev_full_regret_R1")
        cal[name] = summary(d, test, "DEV-only grid calibration, exploratory after the DEV lock; not TEST",
                            {"spec": list(spec), "dev_full_regret_T": float(full.mean()),
                             "grid_over_full_ratio": (float(d.mean() / full.mean()) if full.mean() != 0 else None)})
    res["dev_grid_calibration"] = cal
    res["notes"] = ["GR of one arm is a lower bound of its full TEST regret; P1/P2 differences of GR are not "
                    "lower bounds of differences of full regrets."]

    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for v in rows.values() for c in v}, key=lambda c: (c not in ("domain", "data_seed", "arm", "r"), c))
    with open(out_dir / "test_per_trajectory.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows, key=lambda k: (C.DOMAIN_INDEX[k[0]], k[1], list(C.ARMS).index(k[2]), k[3])):
            w.writerow(rows[k])
    with open(out_dir / "test_units.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "domain", "data_seed", "d"])
        allc = {**{k: v for k, v in prim.items() if isinstance(v, dict)},
                **{k: v for k, v in sec.items() if isinstance(v, dict) and "domains" in v}, **expl, **cal}
        for name, r in allc.items():
            for t, dom in r["domains"].items():
                for s, v in zip(C.DATA_SEEDS, dom["splits"]):
                    w.writerow([name, t, s, repr(float(v))])
    with open(out_dir / "test_results.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    return res


# ------------------------------------------------------------------ --final
def check_final_inputs(here: Path, freeze_sha: str):
    """B1: approved freeze manifest; B5: immutable receipt chain; B2: every DEV output vs its lock."""
    try:
        fz = S.verify_freeze(freeze_sha, here)
        chain = S.verify_chain(freeze_sha, through="SCORES_COMPLETE")
    except S.SchemaError as e:
        raise GateFail(str(e))
    lock_path = C.ANALYSIS_DIR / "DEV_RESULTS_LOCK.json"
    gate(U.file_sha256(lock_path) == EXPECTED_DEV_LOCK_SHA, "DEV_RESULTS_LOCK.json changed")
    gate(fz.get("dev_results_lock_sha256") == EXPECTED_DEV_LOCK_SHA, "freeze manifest pins another DEV lock")
    lock = U.read_json(lock_path)
    for name, h in lock["result_files_sha256"].items():
        gate(U.file_sha256(C.ANALYSIS_DIR / "dev_final" / name) == h, f"locked DEV output {name} changed")
    gate(set(lock["result_files_sha256"]) == {"per_trajectory.csv", "units.csv", "results.json"},
         "DEV lock does not list the three DEV outputs")
    prep = chain["records"]["01_PREPARED.json"]
    sc = chain["records"]["03_SCORES_COMPLETE.json"]
    gate(prep["dev_results_lock_sha256"] == EXPECTED_DEV_LOCK_SHA, "PREPARED record pins another DEV lock")
    gate(prep["dryrun_scores_sha256"] == U.file_sha256(S.DRYRUN_SCORES), "dryrun_scores.csv differs from PREPARED")
    gate(prep["dryrun_report_sha256"] == U.file_sha256(S.DRYRUN_REPORT), "TEST_DRYRUN_REPORT.json differs from PREPARED")
    gate(sc["test_scores_sha256"] == U.file_sha256(S.TEST_SCORES), "test_scores.csv differs from SCORES_COMPLETE")
    gate(int(sc["n_rows"]) == S.N_LOGICAL, "SCORES_COMPLETE n_rows != 12,000")
    return fz, chain


def check_users(test_csv: Path, recorded: dict):
    """B3: test users per split == the count recorded at SCORES_COMPLETE (taken by step D from the sealed manifest)."""
    try:
        t = S.read_scores(Path(test_csv), "test")
    except S.SchemaError as e:
        raise GateFail(str(e))
    seen = {}
    for k, rec in t.items():
        seen[S.split_tag(k[0], k[1])] = rec["test_n_users"]
    gate(set(seen) == set(recorded) and all(int(recorded[k]) == v for k, v in seen.items()),
         "test_n_users per split differ from SCORES_COMPLETE.test_users_by_split")


def output_sha(out_dir: Path) -> Dict[str, str]:
    names = sorted(p.name for p in Path(out_dir).iterdir())
    gate(names == sorted(S.OUT_FILES), f"{out_dir}: files {names} != {sorted(S.OUT_FILES)}")
    return {n: U.file_sha256(Path(out_dir) / n) for n in S.OUT_FILES}


def code_sha(here: Path) -> Dict[str, str]:
    return {f: U.file_sha256(here / f) for f in CODE_FILES}


def prepare(freeze_sha: str):
    """Common checks of --final and --finalize-receipt (chain must end at SCORES_COMPLETE)."""
    import analysis_dev                                   # locked: G7 code hashes + locked test
    test = analysis_dev.locked_test()
    here = Path(__file__).resolve().parent
    fz, chain = check_final_inputs(here, freeze_sha)
    check_users(S.TEST_SCORES, chain["records"]["03_SCORES_COMPLETE.json"]["test_users_by_split"])
    dev_res = U.read_json(C.ANALYSIS_DIR / "dev_final" / "results.json")
    gate(dev_res["confirmatory"]["test"] == test, "locked test differs from the one used for DEV")
    return test, here, chain


def write_analysis_record(here, chain, freeze_sha, outputs, recovered=None):
    rec = {"opening_id": chain["opening_id"], "state": "ANALYSIS_COMPLETE", "utc": U.utc_now(),
           "prev_sha256": chain["record_sha256"]["03_SCORES_COMPLETE.json"], "output_sha256": outputs,
           "analysis_code_sha256": code_sha(here), "freeze_manifest_sha256": freeze_sha,
           "recovered_with_approval": recovered}
    try:
        return S.create_record(S.R_ANALYSIS, rec)
    except S.SchemaError as e:
        raise GateFail(str(e))


def cmd_final(freeze_sha):
    test, here, chain = prepare(freeze_sha)
    res = run_analysis(C.ANALYSIS_DIR / "dev_final" / "per_trajectory.csv", S.DRYRUN_SCORES, S.TEST_SCORES,
                       S.OUT_DIR, test)
    outputs = output_sha(S.OUT_DIR)
    h = write_analysis_record(here, chain, freeze_sha, outputs)
    print(f"opening_id {chain['opening_id']}  freeze {freeze_sha[:16]}...  resumes {chain['n_resumes']}")
    for name in ("P1", "P2"):
        p = res["primary"][name]
        print(name, json.dumps({k: p.get(k) for k in ("T", "p", "n_domains_positive", "replicated")}))
    print(f"[write] {S.R_ANALYSIS.name} {h}")


def cmd_finalize_receipt(freeze_sha, approval_name):
    test, here, chain = prepare(freeze_sha)
    ap = S.OPEN_DIR / approval_name
    gate(Path(approval_name).name == approval_name and ap.is_file(), f"approval {approval_name} not found in {S.OPEN_DIR}")
    a = U.read_json(ap)
    gate(a.get("opening_id") == chain["opening_id"] and a.get("stage") == "ANALYSIS_RECEIPT",
         "approval is not an ANALYSIS_RECEIPT approval for this opening")
    gate(S.OUT_DIR.is_dir(), "no outputs to finalize")
    existing = output_sha(S.OUT_DIR)
    recheck = C.ANALYSIS_DIR / "test_final_recheck"
    run_analysis(C.ANALYSIS_DIR / "dev_final" / "per_trajectory.csv", S.DRYRUN_SCORES, S.TEST_SCORES, recheck, test)
    gate(output_sha(recheck) == existing, "re-computed outputs differ from the existing ones: stop, ask the reviewer")
    h = write_analysis_record(here, chain, freeze_sha, existing,
                              recovered={"file": approval_name, "sha256": U.file_sha256(ap)})
    print(f"[write] {S.R_ANALYSIS.name} {h} (recovered, outputs byte-identical)")


def cmd_lock(freeze_sha):
    here = Path(__file__).resolve().parent
    try:
        S.verify_freeze(freeze_sha, here)
        chain = S.verify_chain(freeze_sha, through="ANALYSIS_COMPLETE")
    except S.SchemaError as e:
        raise GateFail(str(e))
    gate(U.file_sha256(C.ANALYSIS_DIR / "DEV_RESULTS_LOCK.json") == EXPECTED_DEV_LOCK_SHA, "DEV_RESULTS_LOCK.json changed")
    ac = chain["records"]["04_ANALYSIS_COMPLETE.json"]
    gate(ac["freeze_manifest_sha256"] == freeze_sha, "ANALYSIS_COMPLETE was written under another freeze")
    gate(ac["analysis_code_sha256"] == code_sha(here), "analysis code changed since ANALYSIS_COMPLETE")
    gate(output_sha(S.OUT_DIR) == ac["output_sha256"], "TEST outputs changed since ANALYSIS_COMPLETE")
    rec = {"opening_id": chain["opening_id"], "state": "LOCKED", "utc": U.utc_now(),
           "prev_sha256": chain["record_sha256"]["04_ANALYSIS_COMPLETE.json"], "output_sha256": ac["output_sha256"],
           "analysis_complete_sha256": chain["record_sha256"]["04_ANALYSIS_COMPLETE.json"]}
    try:
        h = S.create_record(S.R_LOCKED, rec)
    except S.SchemaError as e:
        raise GateFail(str(e))
    print("TEST RESULTS LOCKED")
    print(f"[write] {S.R_LOCKED.name} {h}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final", action="store_true")
    g.add_argument("--lock", action="store_true")
    g.add_argument("--finalize-receipt", action="store_true")
    ap.add_argument("--freeze-sha", help="SHA-256 of TEST_PLAN_FREEZE.json as approved and logged before step D")
    ap.add_argument("--approval", help="--finalize-receipt only: reviewer approval file name in analysis\\test_open")
    a = ap.parse_args()
    gate(bool(a.freeze_sha) and len(a.freeze_sha) == 64, "--freeze-sha <64 hex> is required")
    if a.final:
        U.start_log("analysis_test_final")
        cmd_final(a.freeze_sha)
    elif a.lock:
        U.start_log("analysis_test_lock")
        cmd_lock(a.freeze_sha)
    else:
        gate(bool(a.approval), "--finalize-receipt needs --approval <file>")
        U.start_log("analysis_test_finalize_receipt")
        cmd_finalize_receipt(a.freeze_sha, a.approval)


if __name__ == "__main__":
    main()
