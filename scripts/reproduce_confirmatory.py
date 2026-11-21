"""Recompute the confirmatory contrasts P1 (A3 - A0) and P2 (A3 - A5) from the released per-trajectory tables.

Needs only numpy and pandas. From the repository root:
    python scripts/reproduce_confirmatory.py

Steps
  1. Check the SHA-256 of the released inputs against the values pinned in this script.
  2. DEV: per-trajectory regret at rule R1 (and R2), horizon 120, from dev_results_locked/per_trajectory.csv.
     TEST: grid-restricted regret from test_results_locked/test_per_trajectory.csv.
  3. Pair the two arms on (domain, data split, training seed), average the 5 seeds -> 16 split effects.
  4. Exact two-sided sign-flip test over all 2^16 sign assignments; number of domains with a positive mean.
  5. Compare with the locked unit files and the locked result files (tolerance 1e-12).
Optionally (if code/ imports), the locked implementation code/stats_confirm.py is run on the same units too.
Exit code 0 if everything matches.
"""
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = {
    "dev_results_locked/per_trajectory.csv": "5a63539f952f7a2eebf746a20e9f4060d5c0368aaabe1205f9f485caad130b2f",
    "dev_results_locked/units.csv": "d2f5168b03f4ff844ac25803df9fc3f446a35df86306741c5e4a9800c8b2a59d",
    "dev_results_locked/results.json": "f717cec3bec5654086df9a7cbbdaf15aab2ec1dcc57f5128c645b22bcd61b64e",
    "test_results_locked/test_per_trajectory.csv": "daafdf896da3d45ffcc649fc2fa69877568dba1144f9c66ec6f076fa4e62b49a",
    "test_results_locked/test_units.csv": "456ac67f94b509b5b1c950abd5c98ca62ddca1e7a588733e01991523fcb0f8b0",
    "test_results_locked/test_results.json": "a95d8d1920176999573f323a93864cfa8696abb4935f4f184119627cc074098f",
}
DOMAINS = ["VG", "CSJ", "SO", "TG"]          # Video_Games, Clothing_Shoes_and_Jewelry, Sports_and_Outdoors, Toys_and_Games
SPLITS = [237, 337, 437, 537]
IDX = pd.MultiIndex.from_product([DOMAINS, SPLITS], names=["domain", "data_seed"])
CONTRASTS = {"P1": ("A3", "A0"), "P2": ("A3", "A5")}
SIGNS = np.array(list(itertools.product([1.0, -1.0], repeat=16)))


def path(rel):
    return os.path.join(ROOT, *rel.split("/"))


def signflip_p(u):
    T = u.mean()
    return float(np.mean(np.abs(SIGNS @ u / 16) >= abs(T) * (1 - 1e-12) - 1e-15))


def split_effects(df, col, a, b):
    X = df.set_index(["domain", "data_seed", "arm", "r"])[col]
    d = (X.xs(a, level="arm") - X.xs(b, level="arm")).groupby(level=["domain", "data_seed"])
    assert (d.size() == 5).all(), "expected 5 training seeds per split"
    return d.mean().reindex(IDX).values


def main():
    ok = True
    for rel, h in PINNED.items():
        got = hashlib.sha256(open(path(rel), "rb").read()).hexdigest()
        print(f"{'OK  ' if got == h else 'FAIL'} sha256 {rel}")
        ok &= got == h
    dev = pd.read_csv(path("dev_results_locked/per_trajectory.csv"))
    tst = pd.read_csv(path("test_results_locked/test_per_trajectory.csv"))
    du = pd.read_csv(path("dev_results_locked/units.csv"))
    tu = pd.read_csv(path("test_results_locked/test_units.csv"))
    dr = json.load(open(path("dev_results_locked/results.json"), encoding="utf-8"))
    tr = json.load(open(path("test_results_locked/test_results.json"), encoding="utf-8"))
    rows, units = [], {}
    for set_, rule in (("DEV", "R1"), ("DEV", "R2"), ("TEST", "R1"), ("TEST", "R2")):
        for name, (a, b) in CONTRASTS.items():
            key = name if rule == "R1" else f"{name}_under_R2"
            if set_ == "DEV":
                sub = dev[(dev.rule == rule) & (dev.horizon == 120)]
                u = split_effects(sub, "regret", a, b)
                locked_u = du[du.contrast == key].set_index(["domain", "data_seed"]).d.reindex(IDX).values
                lr = dr["confirmatory"][name] if rule == "R1" else dr["secondary"][key]
            else:
                u = split_effects(tst, f"test_grid_regret_{rule}", a, b)
                locked_u = tu[tu.contrast == key].set_index(["domain", "data_seed"]).d.reindex(IDX).values
                lr = tr["primary"][name] if rule == "R1" else tr["secondary"][key]
            T, p = float(u.mean()), signflip_p(u)
            dom = int((u.reshape(4, 4).mean(1) > 0).sum())
            match = (np.abs(u - locked_u).max() < 1e-12 and abs(T - lr["T"]) < 1e-12 and abs(p - lr["p"]) < 1e-12
                     and dom == lr["n_domains_positive"])
            ok &= bool(match)
            units[(set_, key)] = u
            rows.append({"set": set_, "contrast": key, "T": T, "p_signflip": p, "domains_positive": f"{dom}/4",
                         "splits_positive": f"{int((u > 0).sum())}/16", "matches_locked": bool(match)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    try:                                                   # optional: the locked implementation itself
        sys.path.insert(0, path("code"))
        import stats_confirm as ST
        for (set_, key), u in units.items():
            p_locked = float(ST.signflip_pvalues(u[None, :])[0])
            same = abs(p_locked - signflip_p(u)) < 1e-12
            ok &= same
            print(f"code/stats_confirm.signflip_pvalues {set_} {key}: {p_locked:.12g} {'(same)' if same else '(DIFFERENT)'}")
    except ImportError as e:
        print(f"(skipped the cross-check with code/stats_confirm.py: {e})")
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
