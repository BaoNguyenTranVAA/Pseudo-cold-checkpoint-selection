"""Check the quantitative claims of the manuscript against the files of this repository.

    python scripts/verify_manuscript_numbers.py            (from the repository root; or pass the root as argument)

Scope and method
  * The manuscript values are TRANSCRIBED into this script by hand (the strings in check(...) calls below, taken
    from manuscript version V23 of 20 Nov 2026). Nothing is extracted from the PDF; a value that is not
    transcribed here is not checked.
  * Each computed value is rounded to the number of decimals printed in the manuscript and compared as text.
  * Column `basis` of the output says where the computed value comes from:
      recomputed      computed here from per-trajectory, per-state or per-unit rows of the released tables;
      locked-field    read from a field of a locked result file (results.json, test_results.json,
                      exploratory_family.csv) and rounded; these are NOT recomputed by the claim itself, but
                      part 2 below recomputes T, p, 95% CI and Holm of those fields from the unit rows;
      ratio of locked fields / arithmetic   simple arithmetic on locked fields or on constants.
  * Part 2 ("consistency") recomputes every locked contrast field that has unit rows (T = mean of the 16
    split effects, exact two-sided sign-flip p over 2^16 patterns, CI by test inversion with the locked
    code/stats_confirm.py, Holm over the 21 exploratory contrasts) and compares with the locked value.
  * Part 3 checks the plotted data of Figure 3 in figure_data/: hashes against FIG3_MANIFEST.json, the cold-TEST
    curve (panel b) and the vertical lines against the released TEST tables. The PV curve (panel a) comes from
    the DEV-only dry-run scores, which are not released; its provenance is the input hash in the manifest.
  * Exit code 0 only if every check of parts 1-3 matches; 1 otherwise.
Writes manuscript_check.csv and consistency_check.csv into the current folder (or the folder given by --out).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default=None, help="folder for the two CSV reports (default: the current folder)")
ARGS = ap.parse_args()
R = ARGS.root
OUTDIR = ARGS.out or os.getcwd()
os.makedirs(OUTDIR, exist_ok=True)
J = lambda p: json.load(open(os.path.join(R, p), encoding="utf-8"))
C = lambda p: pd.read_csv(os.path.join(R, p))
DP, TP, TS = C("dev_results_locked/per_trajectory.csv"), C("test_results_locked/test_per_trajectory.csv"), C("test_results_locked/test_scores.csv")
DU, TU = C("dev_results_locked/units.csv"), C("test_results_locked/test_units.csv")
DR, TR = J("dev_results_locked/results.json"), J("test_results_locked/test_results.json")
FAM = C("exploratory_test/out/exploratory_family.csv").set_index("name")
DOM, SPL = ["VG", "CSJ", "SO", "TG"], [237, 337, 437, 537]
IDX = pd.MultiIndex.from_product([DOM, SPL], names=["domain", "data_seed"])
S16 = np.array(list(itertools.product([1., -1.], repeat=16)))
rows = []


def dec(s):
    s = s.replace("%", "").replace("−", "-").replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def check(where, claim, paper, value, scale=1.0, pct=False, b=None):
    """paper: string as printed; value: computed number (scale applied); pct: value is a fraction shown as %."""
    v = value * (100 if pct else scale)
    d = dec(paper)
    got = f"{v:.{d}f}"
    want = paper.replace("−", "-").replace("%", "").replace(",", "")
    ok = abs(float(got) - float(want)) < 0.5 * 10 ** (-d) + 1e-12 and got.lstrip("-") == want.lstrip("-") or got == want
    rows.append({"where": where, "claim": claim, "paper": paper, "computed": got, "basis": b, "ok": ok})


def checkint(where, claim, paper, value, b=None):
    rows.append({"where": where, "claim": claim, "paper": str(paper), "computed": str(value), "basis": b,
                 "ok": str(paper) == str(value)})


def sf(u):
    u = np.asarray(u, float); T = u.mean()
    return T, float(np.mean(np.abs(S16 @ u / 16) >= abs(T) * (1 - 1e-12) - 1e-15)) if len(u) == 16 else None


def units_dev(a, b, col="regret", rule="R1", H=120):
    X = DP[(DP.rule == rule) & (DP.horizon == H)].set_index(["domain", "data_seed", "arm", "r"])[col]
    return (X.xs(a, level="arm") - X.xs(b, level="arm")).groupby(level=["domain", "data_seed"]).mean().reindex(IDX).values


def units_test(a, b, col):
    X = TP.set_index(["domain", "data_seed", "arm", "r"])[col]
    return (X.xs(a, level="arm") - X.xs(b, level="arm")).groupby(level=["domain", "data_seed"]).mean().reindex(IDX).values


def signflip_n(u):
    u = np.asarray(u, float); S = np.array(list(itertools.product([1., -1.], repeat=len(u))))
    return float(np.mean(np.abs(S @ u / len(u)) >= abs(u.mean()) * (1 - 1e-12) - 1e-15))


# ---------------- Abstract / Intro
p1t, p2t = TR["primary"]["P1"], TR["primary"]["P2"]
for w in ("Abstract", "Intro", "Conclusion"):
    check(w, "TEST P1 T", "0.00848", p1t["T"], b="locked-field: results.json / test_results.json"); check(w, "TEST P1 p", "0.00058", p1t["p"], b="locked-field: results.json / test_results.json")
    check(w, "TEST P2 T", "0.00819", p2t["T"], b="locked-field: results.json / test_results.json"); check(w, "TEST P2 p", "0.00269", p2t["p"], b="locked-field: results.json / test_results.json")
tm = TP.groupby("arm").mean(numeric_only=True)
drop = 1 - tm.loc["A3", "sel_test_R1"] / tm.loc["A0", "sel_test_R1"]
for w in ("Abstract", "Intro", "IV-C", "V-B", "Conclusion"):
    check(w, "A3 vs A0 selected TEST NDCG drop", "15.29%", drop, pct=True, b="recomputed")
checkint("Abstract/Intro", "trajectories", 1200, len(TP), b="recomputed")
for k, q in ((205, "10.01%"), (410, "20.02%"), (685, "33.45%")):
    check("Intro/III-D", f"dose {k}/2048", q, k / 2048, pct=True, b="arithmetic")
dom_pos = lambda u: int((np.asarray(u).reshape(4, 4).mean(1) > 0).sum())
checkint("Intro", "P1/P2 TEST positive domain means", "4/4,4/4",
         f"{dom_pos(units_test('A3','A0','test_grid_regret_R1'))}/4,{dom_pos(units_test('A3','A5','test_grid_regret_R1'))}/4", b="recomputed")

# ---------------- Table 3
T3 = {"DEV R1/P1": ("3.766", "1.317", "6.335", "0.00140", "14/16"), "DEV R1/P2": ("3.364", "0.721", "6.126", "0.00989", "10/16"),
      "TEST R1/P1": ("8.477", "2.966", "14.198", "0.00058", "14/16"), "TEST R1/P2": ("8.194", "2.564", "14.044", "0.00269", "12/16"),
      "DEV R2/P1": ("1.578", "0.571", "2.600", "0.00284", None), "DEV R2/P2": ("1.305", "0.281", "2.346", "0.01477", None),
      "TEST R2/P1": ("4.454", "1.182", "7.758", "0.00046", None), "TEST R2/P2": ("4.110", "0.758", "7.553", "0.00668", None)}
src = {"DEV R1/P1": (DR["confirmatory"]["P1"], DU, "P1"), "DEV R1/P2": (DR["confirmatory"]["P2"], DU, "P2"),
       "TEST R1/P1": (TR["primary"]["P1"], TU, "P1"), "TEST R1/P2": (TR["primary"]["P2"], TU, "P2"),
       "DEV R2/P1": (DR["secondary"]["P1_under_R2"], DU, "P1_under_R2"), "DEV R2/P2": (DR["secondary"]["P2_under_R2"], DU, "P2_under_R2"),
       "TEST R2/P1": (TR["secondary"]["P1_under_R2"], TU, "P1_under_R2"), "TEST R2/P2": (TR["secondary"]["P2_under_R2"], TU, "P2_under_R2")}
for lab, (t, lo, hi, p, pos) in T3.items():
    r, U, key = src[lab]
    check("Table 3", f"{lab} T x1e3", t, r["T"], 1e3, b="locked-field: results.json / test_results.json"); check("Table 3", f"{lab} CI lo", lo, r["ci95"]["lo"], 1e3, b="locked-field: results.json / test_results.json")
    check("Table 3", f"{lab} CI hi", hi, r["ci95"]["hi"], 1e3, b="locked-field: results.json / test_results.json"); check("Table 3", f"{lab} p", p, r["p"], b="locked-field: results.json / test_results.json")
    u = U[U.contrast == key].set_index(["domain", "data_seed"]).d.reindex(IDX).values
    if pos:
        checkint("Table 3 / IV-A", f"{lab} positive splits", pos, f"{int((u > 0).sum())}/16", b="recomputed")
    if lab.startswith("TEST R1"):
        Tt, pp = sf(u)
        check("Table 3 (recomputed)", f"{lab} p from units", p, pp, b="recomputed")
# DEV calibration
check("IV-A / App A", "DEV grid calibration P1 %", "99.973%", TR["dev_grid_calibration"]["E5a_DEV_grid_P1_R1"]["grid_over_full_ratio"], pct=True, b="locked-field: results.json / test_results.json")
check("IV-A / App A", "DEV grid calibration P2 %", "98.816%", TR["dev_grid_calibration"]["E5b_DEV_grid_P2_R1"]["grid_over_full_ratio"], pct=True, b="locked-field: results.json / test_results.json")

# ---------------- IV-B heterogeneity
for name, (a, b), paper in (("P1", ("A3", "A0"), ("0.01075", "0.01495", "0.00369", "0.00452")),
                            ("P2", ("A3", "A5"), ("0.01096", "0.01427", "0.00352", "0.00403"))):
    u = units_test(a, b, "test_grid_regret_R1"); m = u.reshape(4, 4).mean(1)
    for dname, pv, mv in zip(DOM, paper, m):
        check("IV-B", f"TEST {name} domain mean {dname}", pv, mv, b="recomputed")
    top2 = np.argsort(-u)[:2]; u14 = np.delete(u, top2)
    check("IV-B", f"{name} drop top-2 T", {"P1": "0.005438", "P2": "0.005182"}[name], u14.mean(), b="recomputed")
    check("IV-B", f"{name} drop top-2 p", {"P1": "0.002319", "P2": "0.010742"}[name], signflip_n(u14), b="recomputed")
    lodo = [signflip_n(np.delete(u.reshape(4, 4), i, 0).ravel()) for i in range(4)]
    lo_, hi_ = {"P1": ("0.000488", "0.009277"), "P2": ("0.004883", "0.031250")}[name]
    check("IV-B", f"{name} LODO min p", lo_, min(lodo), b="recomputed"); check("IV-B", f"{name} LODO max p", hi_, max(lodo), b="recomputed")
    ud = units_dev(a, b)
    check("IV-B", f"{name} Pearson DEV vs TEST units", {"P1": "0.91", "P2": "0.88"}[name], np.corrcoef(ud, u)[0, 1], b="recomputed")

# ---------------- Table 4 / IV-C
T4 = {"A0": ("0.05740", "0.00194", "0.05934", "0.911"), "A5": ("0.05693", "0.00222", "0.05915", "0.922"),
      "A7": ("0.05318", "0.00519", "0.05837", "0.811"), "A3": ("0.04862", "0.01042", "0.05904", "0.654")}
for a, (s, g, mx, rho) in T4.items():
    check("Table 4", f"{a} selected", s, tm.loc[a, "sel_test_R1"], b="recomputed"); check("Table 4", f"{a} grid regret", g, tm.loc[a, "test_grid_regret_R1"], b="recomputed")
    check("Table 4", f"{a} grid max", mx, tm.loc[a, "test_grid_max_R1"], b="recomputed"); check("Table 4", f"{a} rho grid", rho, tm.loc[a, "rho_grid_pv_test"], b="recomputed")
check("IV-C", "A3 grid regret / A0", "5.37", tm.loc["A3", "test_grid_regret_R1"] / tm.loc["A0", "test_grid_regret_R1"], b="recomputed")
for a, v in (("A0", "0.09911"), ("A5", "0.09842"), ("A7", "0.09494"), ("A3", "0.08752")):
    check("IV-C", f"{a} selected Recall@10", v, tm.loc[a, "sel_test_recall_R1"], b="recomputed")
check("IV-C", "random Recall@10 300", "0.03333", 10 / 300, b="arithmetic")
check("IV-C", "A0 recall / random", "2.97", tm.loc["A0", "sel_test_recall_R1"] / (10 / 300), b="recomputed")
for nm, v in (("E1a_selected_loss_A0_minus_A3_R1", "0.008776"), ("E1c_selected_loss_A5_minus_A3_R1", "0.008305")):
    check("IV-C", f"{nm} T", v, FAM.loc[nm, "T"], b="locked-field: exploratory_family.csv")
check("IV-C", "regret diff P1", "0.008477", p1t["T"], b="locked-field: results.json / test_results.json"); check("IV-C", "regret diff P2", "0.008194", p2t["T"], b="locked-field: results.json / test_results.json")
check("IV-C / V-B / App A", "regret share A0-A3", "96.59%", p1t["T"] / FAM.loc["E1a_selected_loss_A0_minus_A3_R1", "T"], pct=True, b="ratio of locked fields")
check("IV-C / V-B / App A", "regret share A5-A3", "98.66%", p2t["T"] / FAM.loc["E1c_selected_loss_A5_minus_A3_R1", "T"], pct=True, b="ratio of locked fields")
check("IV-C", "E1a Holm", "0.13184", FAM.loc["E1a_selected_loss_A0_minus_A3_R1", "p_holm"], b="locked-field: exploratory_family.csv")
check("IV-C", "E1c Holm", "0.13437", FAM.loc["E1c_selected_loss_A5_minus_A3_R1", "p_holm"], b="locked-field: exploratory_family.csv")
# DEV decomposition A0 - A3
sl, cl, rg = units_dev("A0", "A3", "selected_cold"), units_dev("A0", "A3", "max_cold"), units_dev("A3", "A0", "regret")
check("IV-C", "DEV selected gap A0-A3", "0.005774", sl.mean(), b="recomputed"); check("IV-C", "DEV ceiling gap", "0.002008", cl.mean(), b="recomputed")
check("IV-C", "DEV ceiling gap raw p", "0.21097", sf(cl)[1], b="recomputed"); check("IV-C", "DEV regret diff", "0.003766", rg.mean(), b="recomputed")
check("IV-C / V-B", "DEV regret share", "65.22%", rg.mean() / sl.mean(), pct=True, b="recomputed")
check("IV-C", "TEST ceiling gap A0-A3", "0.000300", FAM.loc["E1b_grid_ceiling_loss_A0_minus_A3_R1", "T"], b="locked-field: exploratory_family.csv")
check("IV-C", "TEST ceiling gap A0-A3 p", "0.88388", FAM.loc["E1b_grid_ceiling_loss_A0_minus_A3_R1", "p"], b="locked-field: exploratory_family.csv")
check("IV-C", "TEST ceiling gap A5-A3", "0.000111", FAM.loc["E1d_grid_ceiling_loss_A5_minus_A3_R1", "T"], b="locked-field: exploratory_family.csv")
check("IV-C", "TEST ceiling gap A5-A3 p", "0.95691", FAM.loc["E1d_grid_ceiling_loss_A5_minus_A3_R1", "p"], b="locked-field: exploratory_family.csv")
check("App A", "selected gap A0-A3", "0.0087763", FAM.loc["E1a_selected_loss_A0_minus_A3_R1", "T"], b="locked-field: exploratory_family.csv")
check("App A", "ceiling gap A0-A3", "0.0002997", FAM.loc["E1b_grid_ceiling_loss_A0_minus_A3_R1", "T"], b="locked-field: exploratory_family.csv")
check("App A", "regret gap A0-A3", "0.0084766", p1t["T"], b="locked-field: results.json / test_results.json"); check("App A", "selected gap A5-A3", "0.0083052", FAM.loc["E1c_selected_loss_A5_minus_A3_R1", "T"], b="locked-field: exploratory_family.csv")
check("App A", "regret diff A5-A3", "0.0081944", p2t["T"], b="locked-field: results.json / test_results.json")

# ---------------- IV-D secondary
check("IV-D", "TEST R2 P1", "0.004454", TR["secondary"]["P1_under_R2"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "TEST R2 P1 p", "0.000458", TR["secondary"]["P1_under_R2"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "TEST R2 P2", "0.004110", TR["secondary"]["P2_under_R2"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "TEST R2 P2 p", "0.006683", TR["secondary"]["P2_under_R2"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "E2a Holm", "0.38452", FAM.loc["E2a_P1_R1_minus_R2", "p_holm"], b="locked-field: exploratory_family.csv"); check("IV-D", "E2b Holm", "0.18173", FAM.loc["E2b_P2_R1_minus_R2", "p_holm"], b="locked-field: exploratory_family.csv")
sec = DR["secondary"]
check("IV-D", "DEV A1-A0 T", "0.000221", sec["dose_q010"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV A1-A0 p", "0.35196", sec["dose_q010"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV A2-A0 T", "0.001165", sec["dose_q020"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV A2-A0 p", "0.02863", sec["dose_q020"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV A3-A0 T", "0.003766", DR["confirmatory"]["P1"]["T"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV R2 P1", "0.001578", sec["P1_under_R2"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV R2 P1 p", "0.002838", sec["P1_under_R2"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV R2 P2", "0.001305", sec["P2_under_R2"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV R2 P2 p", "0.01477", sec["P2_under_R2"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV R1-R2 interaction", "0.002188", sec["ma5x_A3_A0"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV R1-R2 interaction p", "0.03915", sec["ma5x_A3_A0"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV H40 P1", "0.005341", sec["horizon40_P1"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV H40 P1 p", "0.001587", sec["horizon40_P1"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV A4-A0 T", "0.000147", sec["cr_A4_minus_A0"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV A4-A0 p", "0.33322", sec["cr_A4_minus_A0"]["p"], b="locked-field: results.json / test_results.json")
# Table 5
T5 = {"R1": ("3.766", "0.00140", "3.364", "0.00989"), "R2": ("1.578", "0.00284", "1.305", "0.01477"),
      "ES10": ("8.806", "0.00293", "8.115", "0.00562"), "MA3": ("2.402", "0.00159", "2.205", "0.00510"),
      "MA9": ("1.371", "0.00113", "1.259", "0.00546"), "MA5_centered": ("1.691", "0.00342", "1.438", "0.01370"),
      "R1_from5": ("2.234", "0.00198", "1.832", "0.01648")}
for rule, (t1, q1, t2, q2) in T5.items():
    u1, u2 = units_dev("A3", "A0", rule=rule), units_dev("A3", "A5", rule=rule)
    check("Table 5", f"{rule} P1 T x1e3", t1, u1.mean(), 1e3, b="recomputed"); check("Table 5", f"{rule} P1 p", q1, sf(u1)[1], b="recomputed")
    check("Table 5", f"{rule} P2 T x1e3", t2, u2.mean(), 1e3, b="recomputed"); check("Table 5", f"{rule} P2 p", q2, sf(u2)[1], b="recomputed")
d120 = DP[DP.horizon == 120]
g = lambda rule, a, col: d120[(d120.rule == rule) & (d120.arm == a)][col]
check("IV-D", "A0 DEV regret ES10", "0.004712", g("ES10", "A0", "regret").mean(), b="recomputed"); check("IV-D", "A0 DEV regret R1", "0.001721", g("R1", "A0", "regret").mean(), b="recomputed")
check("IV-D", "A3 DEV regret ES10", "0.013518", g("ES10", "A3", "regret").mean(), b="recomputed"); check("IV-D", "A3 DEV regret R1", "0.005487", g("R1", "A3", "regret").mean(), b="recomputed")
check("IV-D", "ES10 median e* A3", "4", g("ES10", "A3", "e_star").median(), b="recomputed"); check("IV-D", "ES10 median e* A0", "30.5", g("ES10", "A0", "e_star").median(), b="recomputed")
checkint("IV-D / V-C", "ES10 e*<=10 A3", "56/80", f"{int((g('ES10','A3','e_star') <= 10).sum())}/80", b="recomputed")
checkint("IV-D", "ES10 e*<=10 A0", "1/80", f"{int((g('ES10','A0','e_star') <= 10).sum())}/80", b="recomputed")
check("IV-D", "A7 TEST grid regret", "0.00519", tm.loc["A7", "test_grid_regret_R1"], b="recomputed"); check("IV-D", "A7 TEST grid regret (text)", "0.005186", tm.loc["A7", "test_grid_regret_R1"], b="recomputed")
check("IV-D", "A7 - A5 grid regret", "0.002964", tm.loc["A7", "test_grid_regret_R1"] - tm.loc["A5", "test_grid_regret_R1"], b="recomputed")
check("IV-D", "A5 grid regret", "0.002222", tm.loc["A5", "test_grid_regret_R1"], b="recomputed")
checkint("IV-D", "E3c/E3d survive Holm", "True,True", f"{FAM.loc['E3c_A7_minus_A0_R1','p_holm']<.05},{FAM.loc['E3d_A7_minus_A0_R2','p_holm']<.05}", b="locked-field: exploratory_family.csv")
check("IV-D", "DEV B2-B0 T", "0.000381", sec["B_B2_minus_B0"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV B2-B0 p", "0.32211", sec["B_B2_minus_B0"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "DEV B3-B0 T", "0.000264", sec["B_B3_minus_B0"]["T"], b="locked-field: results.json / test_results.json"); check("IV-D", "DEV B3-B0 p", "0.47003", sec["B_B3_minus_B0"]["p"], b="locked-field: results.json / test_results.json")
check("IV-D", "B0 selected TEST", "0.05387", tm.loc["B0", "sel_test_R1"], b="recomputed")
checkint("IV-D", "A0 > B0 in splits", "16/16", f"{int((units_test('A0','B0','sel_test_R1') > 0).sum())}/16", b="recomputed")

# ---------------- IV-E timing / spikes
r1 = d120[d120.rule == "R1"]
check("IV-E", "A3 median DEV oracle", "40", r1[r1.arm == "A3"].oracle_epoch.median(), b="recomputed")
check("IV-E", "A3 median TEST grid oracle", "40", TP[TP.arm == "A3"].test_grid_oracle_epoch_R1.median(), b="recomputed")
check("IV-E / Fig 3", "A3 median e*", "89.5", TP[TP.arm == "A3"].e_star_R1.median(), b="recomputed")
check("IV-E", "A0 median DEV oracle", "100.5", r1[r1.arm == "A0"].oracle_epoch.median(), b="recomputed")
check("IV-E", "A0 median TEST grid oracle", "100", TP[TP.arm == "A0"].test_grid_oracle_epoch_R1.median(), b="recomputed")
check("IV-E", "A0 median e*", "109", TP[TP.arm == "A0"].e_star_R1.median(), b="recomputed")
G = TS[TS.state.str.startswith("e") & TS.arm.isin(["A0", "A3"])].groupby(["arm", "epoch"]).test_ndcg10.mean().unstack(0)
checkint("IV-E", "epochs A3 > A0 (TEST mean)", "8,9,11-20,30",
         ",".join(str(e) for e in G.index if G.loc[e, "A3"] > G.loc[e, "A0"]).replace("11,12,13,14,15,16,17,18,19,20", "11-20"), b="recomputed")
sp = r1.groupby("arm").early_spike.apply(lambda s: int((s == True).sum()))
checkint("IV-E", "early spike counts A3,A8,A9,A10,A2,B0", "14,14,7,4,2,1", ",".join(str(sp[a]) for a in ("A3", "A8", "A9", "A10", "A2", "B0")), b="recomputed")
checkint("IV-E", "early spike in remaining 9 arms", "0", int(sp.drop(["A3", "A8", "A9", "A10", "A2", "B0"]).sum()), b="recomputed")
a3 = r1[r1.arm == "A3"].set_index(["domain", "data_seed", "r"])
spk = a3.early_spike == True
check("IV-E", "spike A3 mean DEV regret", "0.01664", a3[spk].regret.mean(), b="recomputed"); check("IV-E", "non-spike A3 mean DEV regret", "0.00312", a3[~spk].regret.mean(), b="recomputed")
checkint("IV-E", "spike / non-spike counts", "14/66", f"{int(spk.sum())}/{int((~spk).sum())}", b="recomputed")
t3 = TP[TP.arm == "A3"].set_index(["domain", "data_seed", "r"]).test_grid_regret_R1
t0 = TP[TP.arm == "A0"].set_index(["domain", "data_seed", "r"]).test_grid_regret_R1
diff = (t3 - t0)
check("IV-E", "spike share of summed A3-A0 TEST GR", "59.4%", diff[spk[spk].index].sum() / diff.sum(), pct=True, b="recomputed")

# ---------------- IV-F / Table 6 / Table C-I
check("IV-F", "rho A0", "0.911", tm.loc["A0", "rho_grid_pv_test"], b="recomputed"); check("IV-F", "rho A5", "0.922", tm.loc["A5", "rho_grid_pv_test"], b="recomputed")
check("IV-F", "rho A3", "0.654", tm.loc["A3", "rho_grid_pv_test"], b="recomputed")
checkint("IV-F", "A0-A3 rho positive splits", "16/16", f"{int((units_test('A0','A3','rho_grid_pv_test') > 0).sum())}/16", b="recomputed")
checkint("IV-F / App C", "raw p<.05 / Holm<.05", "10/3", f"{int((FAM.p < .05).sum())}/{int((FAM.p_holm < .05).sum())}", b="locked-field: exploratory_family.csv")
T6 = {"E3c_A7_minus_A0_R1": ("0.00325", "0.00034", "0.00671"), "E3d_A7_minus_A0_R2": ("0.00301", "0.00021", "0.00449"),
      "E4k_coverage_rule_interaction": ("0.00381", "0.00082", "0.01566"), "E1a_selected_loss_A0_minus_A3_R1": ("0.00878", "0.00732", "0.13184"),
      "E2b_P2_R1_minus_R2": ("0.00408", "0.01212", "0.18173")}
for nm, (t, p, h) in T6.items():
    check("Table 6", f"{nm} T", t, FAM.loc[nm, "T"], b="locked-field: exploratory_family.csv"); check("Table 6", f"{nm} p", p, FAM.loc[nm, "p"], b="locked-field: exploratory_family.csv"); check("Table 6", f"{nm} Holm", h, FAM.loc[nm, "p_holm"], b="locked-field: exploratory_family.csv")
CI = [("E1a_selected_loss_A0_minus_A3_R1", "+0.008776", "0.007324", "0.131836"), ("E1b_grid_ceiling_loss_A0_minus_A3_R1", "+0.000300", "0.883881", "1.000000"),
      ("E1c_selected_loss_A5_minus_A3_R1", "+0.008305", "0.007904", "0.134369"), ("E1d_grid_ceiling_loss_A5_minus_A3_R1", "+0.000111", "0.956909", "1.000000"),
      ("E2a_P1_R1_minus_R2", "+0.004022", "0.032043", "0.384521"), ("E2b_P2_R1_minus_R2", "+0.004085", "0.012115", "0.181732"),
      ("E3a_A3_minus_A7_R1", "+0.005231", "0.024292", "0.315796"), ("E3b_A3_minus_A7_R2", "+0.001447", "0.128906", "1.000000"),
      ("E3c_A7_minus_A0_R1", "+0.003246", "0.000336", "0.006714"), ("E3d_A7_minus_A0_R2", "+0.003007", "0.000214", "0.004486"),
      ("E4a_selected_loss_A0_minus_A8_R1", "+0.005639", "0.010193", "0.163086"), ("E4b_selected_loss_A0_minus_A9_R1", "+0.004497", "0.017578", "0.246094"),
      ("E4c_selected_loss_A0_minus_A10_R1", "+0.003629", "0.065216", "0.717377"), ("E4d_selected_loss_A0_minus_A2_R1", "+0.001032", "0.613098", "1.000000"),
      ("E4e_selected_loss_A0_minus_A8_R2", "+0.001212", "0.137909", "1.000000"), ("E4f_selected_loss_A0_minus_A9_R2", "+0.001941", "0.109344", "1.000000"),
      ("E4g_selected_loss_A0_minus_A10_R2", "+0.002063", "0.201019", "1.000000"), ("E4h_selected_loss_A0_minus_A2_R2", "+0.000415", "0.822571", "1.000000"),
      ("E4i_A2_minus_A8_selected_R1", "+0.004607", "0.108032", "1.000000"), ("E4j_A2_minus_A8_selected_R2", "+0.000797", "0.670685", "1.000000"),
      ("E4k_coverage_rule_interaction", "+0.003810", "0.000824", "0.015656")]
checkint("Table C-I", "row count / order", 21, len(FAM), b="locked-field: exploratory_family.csv (row order)")
for i, (nm, t, p, h) in enumerate(CI):
    checkint("Table C-I", f"row {i+1} name", nm, FAM.index[i], b="locked-field: exploratory_family.csv (row order)")
    check("Table C-I", f"{nm} T", t.lstrip("+"), FAM.loc[nm, "T"], b="locked-field: exploratory_family.csv"); check("Table C-I", f"{nm} p", p, FAM.loc[nm, "p"], b="locked-field: exploratory_family.csv"); check("Table C-I", f"{nm} Holm", h, FAM.loc[nm, "p_holm"], b="locked-field: exploratory_family.csv")
checkint("Table C-I", "E4j direction not as DEV-expected (as_expected = False in the records; no dagger printed)", "False", str(bool(FAM.loc["E4j_A2_minus_A8_selected_R2", "as_expected"])), b="locked-field: exploratory_family.csv")

out = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
bad = out[~out.ok]
print(f"Part 1 - manuscript claims: {len(out)} checked, {int(out.ok.sum())} match, {len(bad)} differ")
print(out.groupby("basis").size().to_string())
if len(bad):
    print(bad.to_string(index=False))
out.to_csv(os.path.join(OUTDIR, "manuscript_check.csv"), index=False)

# ---------------- Part 2: locked contrast fields vs recomputation from unit rows
sys.dont_write_bytecode = True  # leave no __pycache__ in the repository
sys.path.insert(0, os.path.join(R, "code"))
os.environ.setdefault("JV_ROOT", R)
import stats_confirm as SC  # locked implementation (sign-flip p and CI by test inversion)

cons = []


def cmp(src, name, field, locked, recomputed, tol=1e-12):
    cons.append({"source": src, "contrast": name, "field": field, "locked": locked, "recomputed": recomputed,
                 "abs_diff": abs(locked - recomputed), "ok": abs(locked - recomputed) <= tol})


def from_units(U, key):
    d = U[U.contrast == key].set_index(["domain", "data_seed"]).d.reindex(IDX).values
    assert len(d) == 16 and np.isfinite(d).all(), key
    return d


def contrast_fields(src, name, rec, d):
    cmp(src, name, "T", rec["T"], float(d.mean()))
    cmp(src, name, "p (exact sign-flip, enumeration)", rec["p"], sf(d)[1])
    cmp(src, name, "p (locked stats_confirm)", rec["p"], float(SC.signflip_pvalues(d[None, :])[0]))
    if rec.get("ci95"):
        ci = SC.signflip_ci(d)
        cmp(src, name, "CI95 lo", rec["ci95"]["lo"], ci["lo"]); cmp(src, name, "CI95 hi", rec["ci95"]["hi"], ci["hi"])
    cmp(src, name, "splits positive", rec["n_splits_positive"], int((d > 0).sum()), 0)
    cmp(src, name, "domains positive", rec["n_domains_positive"], int((d.reshape(4, 4).mean(1) > 0).sum()), 0)


for name, rec in DR["confirmatory"].items():
    if isinstance(rec, dict) and "T" in rec:
        contrast_fields("dev results.json confirmatory", name, rec, from_units(DU, name))
for name, rec in DR["secondary"].items():
    if (DU.contrast == name).any():
        contrast_fields("dev results.json secondary", name, rec, from_units(DU, name))
for name, (a, b) in (("ma5x_A3_A0", ("A3", "A0")), ("ma5x_A2_A0", ("A2", "A0"))):  # rule interaction, no unit rows:
    d = units_dev(a, b, rule="R1") - units_dev(a, b, rule="R2")              # rebuilt from per-trajectory rows
    contrast_fields("dev results.json secondary (from per_trajectory.csv)", name, DR["secondary"][name], d)
for name, conf in (("E5a_DEV_grid_P1_R1", "P1"), ("E5b_DEV_grid_P2_R1", "P2")):
    rec = TR["dev_grid_calibration"][name]
    d = from_units(TU, name)
    cmp("test_results.json dev_grid_calibration", name, "T", rec["T"], float(d.mean()))
    cmp("test_results.json dev_grid_calibration", name, "dev_full_regret_T", rec["dev_full_regret_T"], DR["confirmatory"][conf]["T"])
    cmp("test_results.json dev_grid_calibration", name, "grid_over_full_ratio", rec["grid_over_full_ratio"],
        float(d.mean()) / DR["confirmatory"][conf]["T"])
for part in ("primary", "secondary"):
    for name, rec in TR[part].items():
        if isinstance(rec, dict) and "T" in rec and (TU.contrast == name).any():
            contrast_fields(f"test_results.json {part}", name, rec, from_units(TU, name))
ex = TR["exploratory"]["contrasts"]
names = list(ex)
ps = np.array([sf(from_units(TU, n))[1] for n in names])
order = np.argsort(ps, kind="stable"); m = len(ps)
holm = np.empty(m); run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * ps[i])); holm[i] = run
for n, ph in zip(names, holm):
    d = from_units(TU, n)
    contrast_fields("test_results.json exploratory", n, ex[n], d)
    cmp("test_results.json exploratory", n, "Holm (21 contrasts)", ex[n]["p_holm_within_exploratory_family"], float(ph))
    cmp("exploratory_family.csv", n, "T", FAM.loc[n, "T"], float(d.mean()))
    cmp("exploratory_family.csv", n, "p", FAM.loc[n, "p"], sf(d)[1])
    cmp("exploratory_family.csv", n, "Holm", FAM.loc[n, "p_holm"], float(ph))
cmp("exploratory_family.csv", "(all)", "row order = test_results.json order", 0, 0 if FAM.index.tolist() == names else 1, 0)
CO = pd.DataFrame(cons)
CO.to_csv(os.path.join(OUTDIR, "consistency_check.csv"), index=False)
cbad = CO[~CO.ok]
print(f"Part 2 - locked fields vs unit rows: {len(CO)} checked, {int(CO.ok.sum())} match, {len(cbad)} differ")
if len(cbad):
    print(cbad.to_string(index=False))

# ---------------- Part 3: plotted data of Figure 3 (figure_data/)
import hashlib
fd = os.path.join(R, "figure_data")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = json.load(open(os.path.join(fd, "FIG3_MANIFEST.json"), encoding="utf-8"))
f3 = []


def fcheck(what, ok):
    f3.append({"check": what, "ok": bool(ok)})


fcheck("fig3_curves.py SHA-256 = manifest script_sha256", sha(os.path.join(fd, "fig3_curves.py")) == man["script_sha256"])
for k in ("FIG3_curves.csv", "FIG3_trajectory_medians.json"):
    fcheck(f"{k} SHA-256 = manifest", sha(os.path.join(fd, k)) == man["outputs_sha256"][k])
for k, rel in (("test_scores.csv", "test_results_locked/test_scores.csv"),
               ("test_per_trajectory.csv", "test_results_locked/test_per_trajectory.csv")):
    fcheck(f"manifest input {k} = released file", man["inputs_sha256"][k] == sha(os.path.join(R, rel)))
cur = pd.read_csv(os.path.join(fd, "FIG3_curves.csv"))
tg = TS[TS.state.str.startswith("e") & TS.arm.isin(["A0", "A3", "A5", "A7"])]
w = tg.groupby(["arm", "epoch"]).test_ndcg10.agg(["mean", "size"])
mm = cur.set_index(["arm", "epoch"]).join(w, how="outer")
fcheck("120 rows (4 grid arms x 30 grid epochs), n = 80 everywhere", len(cur) == 120 and (mm.n == 80).all() and (mm["size"] == 80).all())
fcheck("mean_test_ndcg10 = mean of test_scores.csv per (arm, epoch), |diff| <= 1e-9",
       float((mm.mean_test_ndcg10 - mm["mean"]).abs().max()) <= 1e-9)
fcheck("mean_pv_ndcg10 finite (PV scores of grid states are not released; provenance = manifest input dryrun_scores.csv)",
       np.isfinite(cur.mean_pv_ndcg10).all())
med = json.load(open(os.path.join(fd, "FIG3_trajectory_medians.json"), encoding="utf-8"))
ok = True
for a, v in med["trajectory_medians"].items():
    P = TP[TP.arm == a]
    ok &= v["n"] == len(P) and v["median_test_grid_oracle_epoch_R1"] == P.test_grid_oracle_epoch_R1.median() \
        and v["median_e_star_R1"] == P.e_star_R1.median()
fcheck("trajectory medians (vertical lines) = test_per_trajectory.csv", ok)
ww = mm["mean"].unstack(0)
fcheck("A3 above A0 at epochs = test_scores.csv",
       med["A3_mean_test_above_A0_at_epochs"] == [int(e) for e in ww.index if ww.loc[e, "A3"] > ww.loc[e, "A0"]])
F3 = pd.DataFrame(f3)
fbad = F3[~F3.ok]
print(f"Part 3 - Figure 3 data: {len(F3)} checked, {int(F3.ok.sum())} match, {len(fbad)} differ")
if len(fbad):
    print(fbad.to_string(index=False))

allok = not len(bad) and not len(cbad) and not len(fbad)
print("RESULT:", "ALL MATCH" if allok else "MISMATCH")
sys.exit(0 if allok else 1)
