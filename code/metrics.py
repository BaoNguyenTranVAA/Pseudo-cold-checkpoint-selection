"""
Full-catalog ranking metrics (Protocol V1.3 section 5, "Evaluation").

    ev = metrics.Evaluator(dev, device)
    row = ev.evaluate(model)      # {'pv_ndcg10', 'pv_recall10', 'dev_ndcg10', 'dev_recall10',
                                  #  'warmval_ndcg10', 'pv_noid_ndcg10'} + user counts

Definitions (identical to chance_floor.py / G8):
  * ranking over the WHOLE holdout catalog: PV = dev.pv_items (~1,334), cold-DEV =
    dev.cold_dev_items (700), warm-val = dev.warm_items; no sampled negatives;
  * PV and cold-DEV: no seen-item filtering (their catalogs contain no strict-train item);
  * warm-val: items the user has in strict train U PV are removed from the ranking (score -inf);
  * only users with >= 1 positive in the holdout; metrics averaged equally over users;
  * top-k by torch.topk(k=10, sorted=True);
    DCG = sum_{r=1..10} hit_r / log2(r+1); IDCG = sum_{r=1..min(m,10)} 1/log2(r+1); NDCG = DCG/IDCG;
    Recall@10 = hits / m, m = number of the user's positives in the holdout;
  * PV-noID: the same PV ranking computed inside model.id_disabled() (mask untouched).
Item embeddings are computed once per call (model.item_embeddings()); scores = U[users] @ I[catalog]^T,
in float32 on the model's device, users in chunks of EVAL_USER_BATCH. Results are float64.
Scores are continuous, so exact ties are not expected; if they occur, torch.topk decides.
evaluate() runs under torch.no_grad() and restores model.training.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

import constants as C

K = C.EVAL_K
_DISC = 1.0 / np.log2(np.arange(2, K + 2))


class Holdout:
    """Users with >= 1 positive, a dense truth matrix over the catalog, optional exclusions."""

    def __init__(self, pairs: np.ndarray, catalog: np.ndarray, num_items: int,
                 exclude_pairs: Optional[np.ndarray] = None):
        catalog = np.asarray(catalog, dtype=np.int64)
        col = np.full(num_items, -1, dtype=np.int64)
        col[catalog] = np.arange(catalog.size)
        if (col[pairs[:, 1]] < 0).any():
            raise ValueError("a holdout positive is outside its catalog")
        self.catalog = catalog
        self.users, inv = np.unique(pairs[:, 0], return_inverse=True)
        self.truth = np.zeros((self.users.size, catalog.size), dtype=bool)
        self.truth[inv, col[pairs[:, 1]]] = True
        self.m = self.truth.sum(1).astype(np.float64)
        self.idcg = np.array([_DISC[: int(min(mi, K))].sum() for mi in self.m])
        self.exclude = None
        if exclude_pairs is not None and len(exclude_pairs):
            row = np.full(int(max(self.users.max(), exclude_pairs[:, 0].max())) + 1, -1, dtype=np.int64)
            row[self.users] = np.arange(self.users.size)
            r, c = row[exclude_pairs[:, 0]], col[exclude_pairs[:, 1]]
            keep = (r >= 0) & (c >= 0)
            self.exclude = np.zeros_like(self.truth)
            self.exclude[r[keep], c[keep]] = True
            if (self.exclude & self.truth).any():
                raise ValueError("an excluded item is also a holdout positive")


def metrics_from_scores(score_fn: Callable[[np.ndarray], torch.Tensor], h: Holdout,
                        user_batch: int = C.EVAL_USER_BATCH) -> dict:
    """score_fn(user_rows) -> float tensor (len(user_rows), len(catalog)); user_rows index h.users."""
    nd = np.empty(h.users.size)
    rc = np.empty(h.users.size)
    for s in range(0, h.users.size, user_batch):
        rows = np.arange(s, min(s + user_batch, h.users.size))
        sc = score_fn(rows)
        if h.exclude is not None:
            ex = torch.as_tensor(h.exclude[rows], device=sc.device)
            sc = sc.masked_fill(ex, float("-inf"))
        top = torch.topk(sc, k=min(K, sc.shape[1]), dim=1, sorted=True).indices.cpu().numpy()
        hits = h.truth[rows[:, None], top]
        nd[rows] = (hits * _DISC[: top.shape[1]]).sum(1) / h.idcg[rows]
        rc[rows] = hits.sum(1) / h.m[rows]
    return {"ndcg": float(nd.mean()), "recall": float(rc.mean()), "n_users": int(h.users.size)}


class Evaluator:
    def __init__(self, dev, device="cpu"):
        self.device = torch.device(device)
        n = int(dev.num_items)
        self.pv = Holdout(dev.pv_pairs, dev.pv_items, n)
        self.cold = Holdout(dev.cold_dev_pairs, dev.cold_dev_items, n)
        seen = np.concatenate([dev.train_pairs, dev.pv_pairs], axis=0)
        self.warm = Holdout(dev.warm_val_pairs, dev.warm_items, n, exclude_pairs=seen)
        self._t = {}
        for name, h in (("pv", self.pv), ("cold", self.cold), ("warm", self.warm)):
            self._t[name] = (torch.as_tensor(h.users, device=self.device),
                             torch.as_tensor(h.catalog, device=self.device))

    def _scorer(self, model, name, item_emb):
        users, cat = self._t[name]
        U_ = model.user_embedding.weight
        I_ = item_emb[cat]

        def fn(rows):
            return U_[users[torch.as_tensor(rows, device=self.device)]] @ I_.T
        return fn

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        was_training = model.training
        model.eval()
        try:
            emb = model.item_embeddings()
            pv = metrics_from_scores(self._scorer(model, "pv", emb), self.pv)
            cold = metrics_from_scores(self._scorer(model, "cold", emb), self.cold)
            warm = metrics_from_scores(self._scorer(model, "warm", emb), self.warm)
            with model.id_disabled():
                emb_noid = model.item_embeddings()
                noid = metrics_from_scores(self._scorer(model, "pv", emb_noid), self.pv)
        finally:
            model.train(was_training)
        return {"pv_ndcg10": pv["ndcg"], "pv_recall10": pv["recall"],
                "dev_ndcg10": cold["ndcg"], "dev_recall10": cold["recall"],
                "warmval_ndcg10": warm["ndcg"], "pv_noid_ndcg10": noid["ndcg"],
                "n_users": {"pv": pv["n_users"], "dev": cold["n_users"], "warmval": warm["n_users"]}}
