"""
Confirmatory statistics (Protocol V1.3 section 6; DEVIATIONS C2, C4).

Units: 16 split effects d_{j,s}, ordered domain-major (VG, CSJ, SO, TG by
domain_index) and data_seed within domain (237, 337, 437, 537).

Sign-flip test (two-sided, exact): T = mean(d); p = #{patterns with |T*| >= |T|} / 2^16
over all 65,536 independent sign patterns. Interpreted under the assumption
that the joint distribution of the 16 split effects is invariant to independent
sign flips (sufficient: independent and symmetric about 0) -- DEVIATIONS C2.
Implementation: meet-in-the-middle (two halves of 8 units, 256 x 256 sums),
counted exactly per row; ties (|T*| == |T|) count, with a relative tolerance
of 1e-7 after scaling each row by max|d|. Verified against brute-force
enumeration in g7_selftest.py.

Confirmation (step 4): p < 0.05 AND T > 0 AND at least 3 of 4 domains have
mean d_j > 0. Fixed sequence: P2 is tested confirmatorily only if P1 is
confirmed. Fallback test (section 6.1): one-sample t-test, df = 15.

CI 95% for T follows the locked test: sign-flip inversion under a common
shift (d - delta), reported with that assumption (C2), or the t-interval
(df = 15) when the t-test is locked; none when the lock is estimate_only.
Domain CI: t (df = 3), labelled descriptive.
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np
from scipy import stats

import constants as C

N_DOMAINS = 4
SPLITS_PER_DOMAIN = 4
N_UNITS = N_DOMAINS * SPLITS_PER_DOMAIN
_SIGNS8 = np.array(list(itertools.product((1.0, -1.0), repeat=8)), dtype=np.float64)
_REL_TOL = 1e-7
_CHUNK = 512


def unit_order():
    """[(tag, data_seed), ...] in the locked unit order."""
    tags = sorted(C.DOMAIN_INDEX, key=C.DOMAIN_INDEX.get)
    return [(t, s) for t in tags for s in C.DATA_SEEDS]


def signflip_pvalues(D) -> np.ndarray:
    """Exact two-sided sign-flip p-values for each row of D (n x 16)."""
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if D.shape[1] != N_UNITS:
        raise ValueError(f"expected {N_UNITS} units, got {D.shape[1]}")
    out = np.empty(D.shape[0])
    for start in range(0, D.shape[0], _CHUNK):
        X = D[start:start + _CHUNK]
        n = X.shape[0]
        scale = np.abs(X).max(axis=1, keepdims=True)
        zero = scale[:, 0] == 0
        scale[zero] = 1.0
        X = X / scale
        T = np.abs(X.sum(axis=1))
        thr = T * (1.0 - _REL_TOL) - _REL_TOL
        A = X[:, :8] @ _SIGNS8.T                        # n x 256
        B = np.sort(X[:, 8:] @ _SIGNS8.T, axis=1)       # n x 256, |B| <= 8
        off = (np.arange(n) * 100.0)[:, None]           # rows separated by 100 > 2*24
        Bf = (B + off).ravel()
        base = (np.arange(n) * 256)[:, None]
        hi = np.searchsorted(Bf, ((thr[:, None] - A) + off).ravel(), side="left").reshape(n, 256) - base
        lo = np.searchsorted(Bf, ((-thr[:, None] - A) + off).ravel(), side="right").reshape(n, 256) - base
        cnt = (256 - np.clip(hi, 0, 256)) + np.clip(lo, 0, 256)
        p = cnt.sum(axis=1) / 65536.0
        p[(T <= _REL_TOL) | zero] = 1.0
        out[start:start + n] = np.minimum(p, 1.0)
    return out


def signflip_pvalue_bruteforce(d) -> float:
    """Reference implementation by full enumeration (self-tests only); same tie tolerance."""
    d = np.asarray(d, dtype=np.float64)
    m = np.abs(d).max()
    if m == 0:
        return 1.0
    x = d / m
    T = abs(x.sum())
    if T <= _REL_TOL:
        return 1.0
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=x.size)))
    S = signs @ x
    return float(np.mean(np.abs(S) >= T * (1.0 - _REL_TOL) - _REL_TOL))


def ttest_pvalues(D) -> np.ndarray:
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    n = D.shape[1]
    m = D.mean(axis=1)
    se = D.std(axis=1, ddof=1) / np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, m / se, np.where(m == 0, 0.0, np.inf * np.sign(m)))
    return 2.0 * stats.t.sf(np.abs(t), df=n - 1)


def domain_means(D) -> np.ndarray:
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    return D.reshape(D.shape[0], N_DOMAINS, SPLITS_PER_DOMAIN).mean(axis=2)


def decide(D, test: str = "signflip") -> Dict[str, np.ndarray]:
    """Three-condition confirmation rule, vectorized over rows of D (n x 16)."""
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if test == "signflip":
        p = signflip_pvalues(D)
    elif test == "ttest":
        p = ttest_pvalues(D)
    else:
        raise ValueError(test)
    T = D.mean(axis=1)
    n_pos = (domain_means(D) > 0).sum(axis=1)
    confirmed = (p < C.ALPHA) & (T > 0) & (n_pos >= C.MIN_DOMAINS_POSITIVE)
    return {"T": T, "p": p, "n_domains_positive": n_pos, "confirmed": confirmed}


def fixed_sequence(D1, D2, test: str = "signflip") -> Dict[str, np.ndarray]:
    r1, r2 = decide(D1, test), decide(D2, test)
    return {"P1": r1, "P2": r2, "P2_confirmatory": r1["confirmed"],
            "chain_confirmed": r1["confirmed"] & r2["confirmed"]}


def signflip_ci(d, level: float = 0.95, grid: int = 4001, refine: int = 50):
    """CI for a common shift by inverting the sign-flip test: {delta: p(d - delta) >= alpha}."""
    d = np.asarray(d, dtype=np.float64)
    alpha = 1.0 - level
    lo_b, hi_b = float(d.min()), float(d.max())
    if hi_b == lo_b:
        return {"lo": lo_b, "hi": hi_b, "contiguous": True}
    deltas = np.linspace(lo_b, hi_b, grid)
    acc = signflip_pvalues(d[None, :] - deltas[:, None]) >= alpha
    if not acc.any():
        return {"lo": float("nan"), "hi": float("nan"), "contiguous": False}
    idx = np.flatnonzero(acc)
    contiguous = bool(idx[-1] - idx[0] + 1 == idx.size)

    def p_at(x):
        return signflip_pvalues((d - x)[None, :])[0]

    def edge(inside, outside):
        for _ in range(refine):
            mid = 0.5 * (inside + outside)
            if p_at(mid) >= alpha:
                inside = mid
            else:
                outside = mid
        return inside

    lo = deltas[idx[0]] if idx[0] == 0 else edge(deltas[idx[0]], deltas[idx[0] - 1])
    hi = deltas[idx[-1]] if idx[-1] == grid - 1 else edge(deltas[idx[-1]], deltas[idx[-1] + 1])
    return {"lo": float(lo), "hi": float(hi), "contiguous": contiguous}


def t_ci(x: Sequence[float], level: float = 0.95):
    x = np.asarray(x, dtype=np.float64)
    m, n = float(x.mean()), x.size
    if n < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan")}
    h = stats.t.ppf(0.5 + level / 2, df=n - 1) * x.std(ddof=1) / np.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h}


def summarize_contrast(d, test: str = "signflip") -> Dict[str, object]:
    """Report for one contrast on the 16 units.

    test = "signflip": p and CI from the sign-flip test (CI by inversion under a common shift);
           "ttest":    p and CI from the one-sample t-test, df = 15;
           "estimate_only": point estimates and per-domain descriptives only,
                       no p-value and no pooled inferential CI (Protocol 6.1 fallback; C9 review).
    """
    d = np.asarray(d, dtype=np.float64)
    dm = d.reshape(N_DOMAINS, SPLITS_PER_DOMAIN)
    tags = [t for t, _ in unit_order()[::SPLITS_PER_DOMAIN]]
    out = {
        "T": float(d.mean()), "test": test,
        "n_domains_positive": int((dm.mean(axis=1) > 0).sum()),
        "n_splits_positive": int((d > 0).sum()),
        "domains": {t: {"splits": dm[i].tolist(), **t_ci(dm[i]), "ci_method": "t_df3"}
                    for i, t in enumerate(tags)},
    }
    if test == "estimate_only":
        out.update({"p": None, "confirmed_rule": None, "ci95": None,
                    "ci_method": "none (estimate_only lock)"})
        return out
    r = decide(d[None, :], test)
    out.update({"p": float(r["p"][0]), "confirmed_rule": bool(r["confirmed"][0])})
    if test == "signflip":
        out.update({"ci95": signflip_ci(d), "ci_method": "signflip_inversion_common_shift"})
    else:
        t = t_ci(d)
        out.update({"ci95": {"lo": t["lo"], "hi": t["hi"], "contiguous": True}, "ci_method": "t_df15"})
    return out
