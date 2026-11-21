"""POST-LOCK descriptive Figure 3 (Journal V1.3 manuscript): mean PV and cold-TEST NDCG@10 at the 30 saved grid
epochs for A0, A3, A5. No checkpoint is loaded, nothing is re-scored, no sealed file is opened.

Inputs (read only, SHA-256 must equal the locked values):
  dryrun_scores.csv        analysis\\test_dryrun\\   4130f20e... (step C, L9)   PV NDCG@10 of every grid state
  test_scores.csv          analysis\\test_open\\     93e1a08e... (L12)          TEST NDCG@10 of every grid state
  test_per_trajectory.csv  analysis\\test_final\\    daafdf89... (L12)          A3 medians for the vertical lines
Checks: the 9,600 grid states (4 grid arms x 80 trajectories x 30 epochs) have the same keys and the same
state_sha256 / weights_sha256 in both score files (same checkpoints), every value finite.
Outputs, in a NEW folder (refused if it exists): FIG3_curves.csv (the plotted means: one row per arm and grid epoch,
n = 80 trajectories), FIG3_trajectory_medians.json (A3 medians drawn as the vertical lines of panel (b)),
FIG3_MANIFEST.json (SHA-256 of script, inputs, outputs). The script writes data only; it does not draw the figure.
Columns used in the printed Fig. 3: arms A0, A3, A5; mean_pv_ndcg10 -> panel (a); mean_test_ndcg10 -> panel (b);
vertical lines: A3 median_test_grid_oracle_epoch_R1 (dashed) and median_e_star_R1 (dotted). A7 is included for
completeness and is not drawn.

    python fig3_curves.py --dryrun <dryrun_scores.csv> --test-scores <test_scores.csv>
                          --test-traj <test_per_trajectory.csv> --out <new folder>
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LOCKED = {"dryrun": "4130f20e095fe752f8d7c323654f70704adfeb1f3a9a8c822de34f5f058b851e",
          "test_scores": "93e1a08efb0dfca4d1147c3d69a6e8f146039c8cf1fe7e64be156e4357d50be3",
          "test_traj": "daafdf896da3d45ffcc649fc2fa69877568dba1144f9c66ec6f076fa4e62b49a"}
GRID = list(range(1, 21)) + list(range(30, 121, 10))
GRID_ARMS = ("A0", "A3", "A5", "A7")
PLOT_ARMS = ("A0", "A3", "A5")
KEY = ["domain", "data_seed", "arm", "r", "state", "epoch"]


class GateError(Exception):
    pass


def gate(ok, msg):
    if not ok:
        raise GateError(f"GATE FAIL: {msg}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(dryrun, test_scores, test_traj, out, expected=LOCKED):
    gate(not os.path.exists(out), f"{out} exists already (outputs are written once)")
    for name, p in (("dryrun", dryrun), ("test_scores", test_scores), ("test_traj", test_traj)):
        gate(os.path.isfile(p) and sha(p) == expected[name], f"{p} is not the locked {name} file ({expected[name][:8]}...)")
    D = pd.read_csv(dryrun)
    T = pd.read_csv(test_scores)
    P = pd.read_csv(test_traj)
    dg = D[D.state.str.startswith("e") & D.arm.isin(GRID_ARMS)].copy()
    tg = T[T.state.str.startswith("e") & T.arm.isin(GRID_ARMS)].copy()
    gate(len(dg) == len(tg) == 9600, f"grid states: dry run {len(dg)}, TEST {len(tg)}, expected 9600")
    m = dg.merge(tg, on=KEY, suffixes=("_dry", "_test"), validate="one_to_one")
    gate(len(m) == 9600, "grid keys differ between the dry run and TEST")
    gate((m.state_sha256_dry == m.state_sha256_test).all() and (m.weights_sha256_dry == m.weights_sha256_test).all(),
         "a grid state has another checkpoint identity in the dry run and in TEST")
    gate(sorted(m.epoch.unique().tolist()) == GRID, "grid epochs differ from G")
    gate(np.isfinite(m.pv_ndcg10).all() and np.isfinite(m.test_ndcg10).all(), "non-finite score")
    gate((m.groupby(["arm", "epoch"]).size() == 80).all(), "not 80 trajectories at every (arm, epoch)")

    cur = m.groupby(["arm", "epoch"]).agg(mean_pv_ndcg10=("pv_ndcg10", "mean"), mean_test_ndcg10=("test_ndcg10", "mean"),
                                          n=("pv_ndcg10", "size")).reset_index()
    a3 = P[P.arm == "A3"]
    gate(len(a3) == 80, "test_per_trajectory: not 80 A3 trajectories")
    med = {a: {"median_test_grid_oracle_epoch_R1": float(P[P.arm == a].test_grid_oracle_epoch_R1.median()),
               "median_e_star_R1": float(P[P.arm == a].e_star_R1.median()), "n": int((P.arm == a).sum())}
           for a in GRID_ARMS}
    w = cur.pivot(index="epoch", columns="arm", values="mean_test_ndcg10")
    facts = {"A3_mean_test_above_A0_at_epochs": [int(e) for e in w.index if w.loc[e, "A3"] > w.loc[e, "A0"]],
             "trajectory_medians": med}

    os.makedirs(out)
    files = {}
    p = os.path.join(out, "FIG3_curves.csv")
    cur.to_csv(p, index=False, float_format="%.10f")
    files["FIG3_curves.csv"] = p
    p = os.path.join(out, "FIG3_trajectory_medians.json")
    json.dump(facts, open(p, "w", encoding="utf-8"), indent=1)
    files["FIG3_trajectory_medians.json"] = p
    man = {"report_type": "post_lock_fig3", "label": "POST-LOCK descriptive figure; no checkpoint loaded, no re-scoring",
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script_sha256": sha(os.path.abspath(__file__)),
           "inputs_sha256": {"dryrun_scores.csv": sha(dryrun), "test_scores.csv": sha(test_scores),
                             "test_per_trajectory.csv": sha(test_traj)},
           "checks": {"grid_states": 9600, "identity_matched": True, "epochs": GRID},
           "outputs_sha256": {k: sha(v) for k, v in files.items()},
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                           "platform": platform.platform()},
           "facts": facts}
    json.dump(man, open(os.path.join(out, "FIG3_MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: man[k] for k in ("script_sha256", "inputs_sha256", "outputs_sha256", "facts")}, indent=1))
    print(f"FIG3 MANIFEST {sha(os.path.join(out, 'FIG3_MANIFEST.json'))}")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--test-traj", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        run(a.dryrun, a.test_scores, a.test_traj, a.out)
    except GateError as e:
        print(e)
        sys.exit(2)
