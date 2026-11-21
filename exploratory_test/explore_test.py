"""POST-LOCK EXPLORATORY analysis of the locked cold-TEST results (Protocol V1.3; L12).
Nothing here is confirmatory: the only confirmatory TEST results are P1/P2 in test_results.json (05_LOCKED).
Reads only the four locked TEST files and the three locked DEV files (SHA-256 checked first). No sealed\\ file,
no checkpoint, no training log is read; nothing locked is written.

    python explore_test.py <TEST dir: test_scores.csv test_per_trajectory.csv test_units.csv test_results.json>
                           <DEV dir: per_trajectory.csv units.csv results.json> <out_dir>
"""
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

TDIR, DDIR, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
LOCKED = {  # TEST: 05_LOCKED / 03_SCORES_COMPLETE (L12); DEV: DEV_RESULTS_LOCK (0775d9fb...)
    (TDIR, "test_scores.csv"): "93e1a08efb0dfca4d1147c3d69a6e8f146039c8cf1fe7e64be156e4357d50be3",
    (TDIR, "test_per_trajectory.csv"): "daafdf896da3d45ffcc649fc2fa69877568dba1144f9c66ec6f076fa4e62b49a",
    (TDIR, "test_units.csv"): "456ac67f94b509b5b1c950abd5c98ca62ddca1e7a588733e01991523fcb0f8b0",
    (TDIR, "test_results.json"): "a95d8d1920176999573f323a93864cfa8696abb4935f4f184119627cc074098f",
    (DDIR, "per_trajectory.csv"): "5a63539f952f7a2eebf746a20e9f4060d5c0368aaabe1205f9f485caad130b2f",
    (DDIR, "units.csv"): "d2f5168b03f4ff844ac25803df9fc3f446a35df86306741c5e4a9800c8b2a59d",
    (DDIR, "results.json"): "f717cec3bec5654086df9a7cbbdaf15aab2ec1dcc57f5128c645b22bcd61b64e",
}
for (d, f), h in LOCKED.items():
    with open(os.path.join(d, f), "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    assert got == h, f"{f}: SHA-256 {got[:12]} differs from the locked {h[:12]}"
print("inputs == locked SHA-256 (4 TEST + 3 DEV)")

SC = pd.read_csv(os.path.join(TDIR, "test_scores.csv"))
TP = pd.read_csv(os.path.join(TDIR, "test_per_trajectory.csv"))
TU = pd.read_csv(os.path.join(TDIR, "test_units.csv"))
TR = json.load(open(os.path.join(TDIR, "test_results.json"), encoding="utf-8"))
DP = pd.read_csv(os.path.join(DDIR, "per_trajectory.csv"))
DU = pd.read_csv(os.path.join(DDIR, "units.csv"))
DR = json.load(open(os.path.join(DDIR, "results.json"), encoding="utf-8"))

DOM = ["VG", "CSJ", "SO", "TG"]
SPL = [237, 337, 437, 537]
SEEDS = [42, 52, 62, 72, 82]
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "B0", "B1", "B2", "B3"]
GRID_ARMS = ["A0", "A3", "A5", "A7"]
GRID = list(range(1, 21)) + list(range(30, 121, 10))
UIDX = pd.MultiIndex.from_product([DOM, SPL], names=["domain", "data_seed"])
S16 = np.array(list(itertools.product([1, -1], repeat=16)), dtype=float)
S12 = np.array(list(itertools.product([1, -1], repeat=12)), dtype=float)
REP = {}


def signflip(u):
    u = np.asarray(u, float)
    S = S16 if len(u) == 16 else S12 if len(u) == 12 else np.array(list(itertools.product([1, -1], repeat=len(u))), float)
    T = u.mean()
    return float(T), float(np.mean(np.abs(S @ u / len(u)) >= abs(T) - 1e-15))


def hl(u):
    u = np.asarray(u, float)
    return float(np.median([(u[i] + u[j]) / 2 for i in range(len(u)) for j in range(i, len(u))]))


def t_domains(u):
    """one-sample t (df = 3) on the 4 domain means of 16 units in locked order."""
    from math import sqrt
    m = np.asarray(u, float).reshape(4, 4).mean(1)
    t = m.mean() / (m.std(ddof=1) / sqrt(4))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), 3))
    except Exception:
        p = None
    return float(t), p


def summ(u):
    u = np.asarray(u, float)
    T, p = signflip(u)
    return {"T": T, "p": p, "median": float(np.median(u)), "HL": hl(u), "split_pos": int((u > 0).sum()),
            "dom_pos": int((u.reshape(4, 4).mean(1) > 0).sum()), "dom_means": u.reshape(4, 4).mean(1).round(6).tolist()}


T1 = TP.set_index(["domain", "data_seed", "arm", "r"]).sort_index()


def tunits(a, b, col, X=T1):
    d = (X.xs(a, level="arm")[col] - X.xs(b, level="arm")[col]).groupby(level=["domain", "data_seed"]).mean()
    return d.reindex(UIDX).values


def dsel(rule="R1", H=120):
    return DP[(DP.rule == rule) & (DP.horizon == H)].set_index(["domain", "data_seed", "arm", "r"]).sort_index()


def dunits(a, b, col, rule="R1"):
    X = dsel(rule)
    d = (X.xs(a, level="arm")[col] - X.xs(b, level="arm")[col]).groupby(level=["domain", "data_seed"]).mean()
    return d.reindex(UIDX).values


# ============================================================ 0. independent reproduction from test_scores.csv
K = ["domain", "data_seed", "arm", "r"]
st = SC.copy()
st["is_grid"] = st.state.str.startswith("e")
G = st[st.is_grid].pivot_table(index=K, columns="epoch", values="test_ndcg10")
assert list(G.columns) == GRID and len(G) == 320
sel = st[~st.is_grid].pivot_table(index=K, columns="state", values=["test_ndcg10", "epoch"])
repro = {}
for rule in ("R1", "R2"):
    gr = {}
    for k in G.index:
        e, v = int(sel.loc[k, ("epoch", rule)]), sel.loc[k, ("test_ndcg10", rule)]
        gr[k] = max(G.loc[k].max(), v) - v
    gr = pd.Series(gr)
    gr.index = pd.MultiIndex.from_tuples(gr.index, names=K)
    for name, (a, b) in {"P1": ("A3", "A0"), "P2": ("A3", "A5")}.items():
        u = (gr.xs(a, level="arm") - gr.xs(b, level="arm")).groupby(level=["domain", "data_seed"]).mean().reindex(UIDX).values
        T, p = signflip(u)
        key = name if rule == "R1" else f"{name}_under_R2"
        locked = TR["primary"][name] if rule == "R1" else TR["secondary"][key]
        lu = TU[TU.contrast == key].set_index(["domain", "data_seed"])["d"].reindex(UIDX).values
        repro[key] = {"T": T, "p": p, "T_locked": locked["T"], "p_locked": locked["p"],
                      "max_abs_unit_diff": float(np.abs(lu - u).max())}
        assert abs(T - locked["T"]) < 1e-12 and abs(p - locked["p"]) < 1e-12
REP["reproduction_from_test_scores"] = repro
print("reproduced from test_scores.csv:", {k: (round(v["T"], 7), v["p"]) for k, v in repro.items()})

# ============================================================ 1. DEV vs TEST primary side by side + robustness
rob = {}
for name, (a, b) in {"P1": ("A3", "A0"), "P2": ("A3", "A5")}.items():
    for rule in ("R1", "R2"):
        ut = tunits(a, b, f"test_grid_regret_{rule}")
        ud = dunits(a, b, "regret", rule)
        ug = tunits(a, b, f"dev_grid_regret_{rule}")
        r = {"TEST": summ(ut), "DEV_full": summ(ud), "DEV_grid": summ(ug)}
        r["TEST"]["t_domains"] = t_domains(ut)
        lodo = {}
        for i, dm in enumerate(DOM):
            keep = np.r_[0:i * 4, i * 4 + 4:16]
            lodo[f"drop_{dm}"] = dict(zip(("T", "p"), signflip(ut[keep])))
        r["TEST"]["LODO"] = lodo
        top2 = np.argsort(-ut)[:2]
        u14 = np.delete(ut, top2)                            # exact sign-flip on the remaining 14 units
        r["TEST"]["drop_top2"] = {"units": [f"{UIDX[i][0]}-{UIDX[i][1]}" for i in top2], "T": float(u14.mean())}
        S14 = np.array(list(itertools.product([1, -1], repeat=14)), float)
        r["TEST"]["drop_top2"]["p"] = float(np.mean(np.abs(S14 @ u14 / 14) >= abs(u14.mean()) - 1e-15))
        r["TEST"]["loo_split_T_range"] = [float(min(np.delete(ut, i).mean() for i in range(16))),
                                          float(max(np.delete(ut, i).mean() for i in range(16)))]
        # seed noise: within-split SD of the 5 paired differences / sqrt(5), pooled (RMS) over splits
        X = T1[f"test_grid_regret_{rule}"]
        pd_ = (X.xs(a, level="arm") - X.xs(b, level="arm"))
        within = pd_.groupby(level=["domain", "data_seed"]).std(ddof=1) / np.sqrt(5)
        r["TEST"]["unit_sd"] = float(np.std(ut, ddof=1))
        r["TEST"]["seed_only_sd"] = float(np.sqrt((within ** 2).mean()))
        pairs = pd_.values
        r["TEST"]["pairs_pos"] = f"{int((pairs > 0).sum())}/{len(pairs)}"
        # transfer DEV -> TEST across the 16 units
        from scipy import stats
        r["transfer_units"] = {"pearson_DEVfull_TEST": float(np.corrcoef(ud, ut)[0, 1]),
                               "spearman_DEVfull_TEST": float(stats.spearmanr(ud, ut)[0]),
                               "sign_agree": int((np.sign(ud) == np.sign(ut)).sum()),
                               "TEST_over_DEVfull_T": float(ut.mean() / ud.mean())}
        r["units"] = pd.DataFrame({"DEV_full": ud, "DEV_grid": ug, "TEST": ut}, index=UIDX).round(8).reset_index().to_dict("records")
        rob[f"{name}_{rule}"] = r
REP["primary_dev_vs_test"] = rob
pd.DataFrame([{"contrast": k, **{f"{s}_{m}": v[s][m] for s in ("DEV_full", "DEV_grid", "TEST") for m in ("T", "p", "split_pos", "dom_pos")},
               "TEST_HL": v["TEST"]["HL"], "TEST_median": v["TEST"]["median"], "TEST_t_dom_p": v["TEST"]["t_domains"][1],
               "TEST_unit_sd": v["TEST"]["unit_sd"], "TEST_seed_sd": v["TEST"]["seed_only_sd"],
               "TEST_drop_top2_T": v["TEST"]["drop_top2"]["T"], "TEST_drop_top2_p": v["TEST"]["drop_top2"]["p"],
               "pearson_units": v["transfer_units"]["pearson_DEVfull_TEST"], "spearman_units": v["transfer_units"]["spearman_DEVfull_TEST"]}
              for k, v in rob.items()]).to_csv(os.path.join(OUT, "primary_dev_vs_test.csv"), index=False)
pd.DataFrame([{"contrast": k, "unit": f"{x['domain']}-{x['data_seed']}", "DEV_full": x["DEV_full"], "DEV_grid": x["DEV_grid"], "TEST": x["TEST"]}
              for k, v in rob.items() for x in v["units"]]).to_csv(os.path.join(OUT, "units_dev_vs_test.csv"), index=False)

# ============================================================ 2. arm table on TEST (R1/R2) + DEV analogue
g = T1.groupby(level="arm")
arm = pd.DataFrame({
    "sel_test_R1": g.sel_test_R1.mean(), "sel_test_R2": g.sel_test_R2.mean(),
    "sel_recall_R1": g.sel_test_recall_R1.mean(), "sel_recall_R2": g.sel_test_recall_R2.mean(),
    "e_star_R1_med": g.e_star_R1.median(), "e_star_R2_med": g.e_star_R2.median(),
}).reindex(ARMS)
for rule in ("R1", "R2"):
    arm[f"test_GR_{rule}"] = g[f"test_grid_regret_{rule}"].mean()
    arm[f"test_gridmax_{rule}"] = g[f"test_grid_max_{rule}"].mean()
    arm[f"test_oracle_{rule}_med"] = g[f"test_grid_oracle_epoch_{rule}"].median()
arm["rho_pv_test"] = g.rho_grid_pv_test.mean()
D1 = dsel("R1").groupby(level="arm")
arm["dev_sel_R1"] = D1.selected_cold.mean()
arm["dev_regret_R1"] = D1.regret.mean()
arm["dev_oracle_R1_med"] = D1.oracle_epoch.median()
arm["dev_rho_pv_dev"] = D1.spearman_pv_dev.mean()
arm["dev_early_spike"] = D1.early_spike.mean()
arm["sel_test_R1_rel_A0_pct"] = 100 * (arm.sel_test_R1 / arm.loc["A0", "sel_test_R1"] - 1)
arm.to_csv(os.path.join(OUT, "arm_table_test.csv"))
REP["arm_table"] = arm.round(6).reset_index().to_dict("records")
REP["random_recall_at10_300"] = 10 / 300
print(arm[["sel_test_R1", "sel_test_R2", "test_GR_R1", "test_GR_R2", "e_star_R1_med", "test_oracle_R1_med", "rho_pv_test"]].round(5).to_string())

# ============================================================ 3. decomposition: selected loss = ceiling loss + GR difference
dec = {}
for (lo, hi) in (("A0", "A3"), ("A5", "A3"), ("A0", "A7"), ("A7", "A3")):
    for rule in ("R1", "R2"):
        sl = tunits(lo, hi, f"sel_test_{rule}")              # selected NDCG of lo minus hi (loss of hi)
        cl = tunits(lo, hi, f"test_grid_max_{rule}")         # ceiling loss
        rg = tunits(hi, lo, f"test_grid_regret_{rule}")      # regret increase
        assert np.allclose(sl, cl + rg, atol=1e-12)
        tsl, psl = signflip(sl); tcl, pcl = signflip(cl); trg, prg = signflip(rg)
        dec[f"{hi}_vs_{lo}_{rule}"] = {"selected_loss": tsl, "p_sel": psl, "ceiling_loss": tcl, "p_ceiling": pcl,
                                      "regret_increase": trg, "p_regret": prg,
                                      "selection_share": trg / tsl if tsl else None}
# DEV analogue (full regret; DEV grid-restricted not needed: E5 ratio ~ 1)
for (lo, hi) in (("A0", "A3"), ("A5", "A3"), ("A0", "A7"), ("A7", "A3")):
    for rule in ("R1", "R2"):
        sl = dunits(lo, hi, "selected_cold", rule); cl = dunits(lo, hi, "max_cold", rule); rg = dunits(hi, lo, "regret", rule)
        tsl, psl = signflip(sl); tcl, pcl = signflip(cl); trg, prg = signflip(rg)
        dec[f"DEV_{hi}_vs_{lo}_{rule}"] = {"selected_loss": tsl, "p_sel": psl, "ceiling_loss": tcl, "p_ceiling": pcl,
                                          "regret_increase": trg, "p_regret": prg, "selection_share": trg / tsl if tsl else None}
REP["decomposition"] = dec
pd.DataFrame(dec).T.to_csv(os.path.join(OUT, "decomposition.csv"))

# ============================================================ 4. trajectory-level DEV -> TEST transfer
from scipy import stats
traj = TP.merge(dsel("R1").reset_index()[K + ["regret", "selected_cold", "max_cold", "oracle_epoch", "early_spike",
                                              "max_pv", "spearman_pv_dev"]], on=K, how="left")
assert traj.regret.notna().all() and len(traj) == 1200
tr = {}
for a in GRID_ARMS:
    s = traj[traj.arm == a]
    tr[a] = {"spearman_devregret_testGR_R1": float(stats.spearmanr(s.regret, s.test_grid_regret_R1)[0]),
             "spearman_devoracle_testoracle": float(stats.spearmanr(s.oracle_epoch, s.test_grid_oracle_epoch_R1)[0])}
tr["all_arms_spearman_dev_sel_vs_test_sel_R1"] = float(stats.spearmanr(traj.selected_cold, traj.sel_test_R1)[0])
tr["all_arms_pearson_dev_sel_vs_test_sel_R1"] = float(np.corrcoef(traj.selected_cold, traj.sel_test_R1)[0, 1])
s = traj[traj.arm.isin(GRID_ARMS)]
tr["grid_arms_spearman_devregret_testGR_R1"] = float(stats.spearmanr(s.regret, s.test_grid_regret_R1)[0])
# early spike (DEV flag) and TEST GR
a3 = traj[traj.arm == "A3"]
sp = a3[a3.early_spike == 1]
nsp = a3[a3.early_spike == 0]
a0 = traj[traj.arm == "A0"]
tr["early_spike_A3"] = {
    "n_spike": int(len(sp)), "splits": {f"{d}-{s}": int(n) for (d, s), n in sp.groupby(["domain", "data_seed"]).size().items()},
    "test_GR_R1_spike": float(sp.test_grid_regret_R1.mean()), "test_GR_R1_no_spike": float(nsp.test_grid_regret_R1.mean()),
    "test_GR_R1_A0": float(a0.test_grid_regret_R1.mean()),
    "e_star_R1_spike_med": float(sp.e_star_R1.median()), "test_oracle_R1_spike_med": float(sp.test_grid_oracle_epoch_R1.median()),
    "share_of_P1_sum_from_spike": float((sp.test_grid_regret_R1.values -
                                         a0.set_index(K[:2] + ["r"]).loc[[tuple(x) for x in sp[["domain", "data_seed", "r"]].values]].test_grid_regret_R1.values).sum()
                                        / (a3.test_grid_regret_R1.sum() - a0.test_grid_regret_R1.sum())),
    "test_GR_R2_spike": float(sp.test_grid_regret_R2.mean()), "test_GR_R2_no_spike": float(nsp.test_grid_regret_R2.mean()),
}
tr["early_spike_by_arm_dev"] = traj.groupby("arm").early_spike.mean().reindex(ARMS).round(4).to_dict()
REP["transfer_trajectory"] = tr
traj.to_csv(os.path.join(OUT, "trajectory_dev_test.csv"), index=False)

# ============================================================ 5. timing on TEST
tim = []
for a in ARMS:
    s = traj[traj.arm == a]
    row = {"arm": a, "e_star_R1_med": s.e_star_R1.median(), "dev_oracle_med": s.oracle_epoch.median(),
           "e_star_R2_med": s.e_star_R2.median()}
    if a in GRID_ARMS:
        o = s.test_grid_oracle_epoch_R1
        row.update({"test_oracle_R1_med": o.median(),
                    "frac_e_late_gt10": float((s.e_star_R1 > o + 10).mean()),
                    "frac_e_early_gt10": float((s.e_star_R1 < o - 10).mean()),
                    "frac_test_oracle_le20": float((o <= 20).mean())})
    tim.append(row)
tim = pd.DataFrame(tim).set_index("arm")
tim.to_csv(os.path.join(OUT, "timing_test.csv"))
REP["timing"] = tim.round(4).reset_index().to_dict("records")
# mean TEST curve on the grid, per grid arm (absolute and relative to each trajectory's grid max)
curve = []
for a in GRID_ARMS:
    M = G.xs(a, level="arm")
    rel = M.div(M.max(axis=1), axis=0)
    for e in GRID:
        curve.append({"arm": a, "epoch": e, "mean_test_ndcg": float(M[e].mean()), "mean_rel_to_gridmax": float(rel[e].mean())})
curve = pd.DataFrame(curve)
curve.to_csv(os.path.join(OUT, "test_curves_grid.csv"), index=False)
REP["test_curve_peak_epoch"] = {a: int(curve[curve.arm == a].sort_values("mean_test_ndcg").epoch.iloc[-1]) for a in GRID_ARMS}

# ============================================================ 6. PV -> TEST ranking fidelity (rho over the grid)
rho = {}
for a in GRID_ARMS:
    rho[a] = {"mean": float(T1.xs(a, level="arm").rho_grid_pv_test.mean()),
              "median": float(T1.xs(a, level="arm").rho_grid_pv_test.median())}
for a, b in (("A3", "A0"), ("A3", "A5"), ("A7", "A0"), ("A3", "A7")):
    u = tunits(a, b, "rho_grid_pv_test")
    rho[f"{a}-{b}"] = summ(u)
REP["rho_pv_test"] = rho

# ============================================================ 7. other contrasts on selected TEST NDCG (post-lock, not in the plan)
oth = {}
spec = {"dose_A0-A1": ("A0", "A1"), "dose_A0-A2": ("A0", "A2"), "dose_A0-A3": ("A0", "A3"),
        "CR_A0-A4": ("A0", "A4"), "CR_A0-A5": ("A0", "A5"), "ID_A6-A2": ("A6", "A2"), "ID_A0-A6": ("A0", "A6"),
        "ID_A7-A3": ("A7", "A3"), "ID_A0-A7": ("A0", "A7"),
        "B_A0-B0": ("A0", "B0"), "B_B0-B2": ("B0", "B2"), "B_B0-B3": ("B0", "B3"), "B_B0-B1": ("B0", "B1")}
for name, (a, b) in spec.items():
    for rule in ("R1", "R2"):
        oth[f"{name}_{rule}"] = summ(tunits(a, b, f"sel_test_{rule}"))
        oth[f"{name}_{rule}"]["DEV_T"] = float(dunits(a, b, "selected_cold", rule).mean())
REP["selected_loss_other"] = oth
pd.DataFrame({k: {m: v[m] for m in ("T", "p", "split_pos", "dom_pos", "DEV_T")} for k, v in oth.items()}).T.to_csv(
    os.path.join(OUT, "selected_loss_other.csv"))

# ============================================================ 8. exploratory family summary (locked) + DEV analogue
ex = TR["exploratory"]["contrasts"]
fam = pd.DataFrame([{"name": k, "T": v["T"], "p": v["p"], "p_holm": v["p_holm_within_exploratory_family"],
                     "split_pos": v["n_splits_positive"], "dom_pos": v["n_domains_positive"],
                     "dev_sign": v["dev_informed_expected_sign"], "as_expected": v["direction_as_expected"],
                     "ci_lo": v["ci95"]["lo"], "ci_hi": v["ci95"]["hi"]} for k, v in ex.items()])
fam.to_csv(os.path.join(OUT, "exploratory_family.csv"), index=False)
REP["exploratory_family"] = {"n": len(fam), "holm_lt_0.05": fam[fam.p_holm < .05].name.tolist(),
                             "raw_lt_0.05": fam[fam.p < .05].name.tolist(),
                             "direction_as_expected": int(fam.as_expected.sum())}
# sign test of the 21 directions (not independent; descriptive)
REP["exploratory_family"]["binom_p_directions"] = float(stats.binomtest(int(fam.as_expected.sum()), len(fam), 0.5).pvalue) \
    if hasattr(stats, "binomtest") else float(stats.binom_test(int(fam.as_expected.sum()), len(fam), 0.5))

with open(os.path.join(OUT, "explore_test_report.json"), "w", encoding="utf-8") as f:
    json.dump(REP, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("written", OUT)
