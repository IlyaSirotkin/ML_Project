import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


class ItemKNNRecommender:
    def __init__(self, path):
        path = Path(path)
        self.sim = load_npz(path / "sim.npz").tocsr()
        self.hist = load_npz(path / "history.npz").tocsr()
        self.items = pd.read_csv(path / "items.csv").sort_values("iid").reset_index(drop=True)
        users = pd.read_csv(path / "users.csv")
        self.uid_of = dict(zip(users["userId"], users["uid"]))
        self.iid_of = dict(zip(self.items["movieId"], self.items["iid"]))
        self.meta = json.load(open(path / "meta.json"))

    def _top(self, scores, seen, n):
        s = scores.astype(np.float64)
        s[seen] = -np.inf
        n = min(n, len(s) - 1)
        top = np.argpartition(-s, n)[:n]
        top = top[np.argsort(-s[top])]
        top = top[s[top] > 0]
        out = self.items.loc[top, ["movieId", "title", "genres"]].copy()
        out["score"] = s[top]
        return out.reset_index(drop=True)

    def recommend_user(self, user_id, n=10):
        u = self.uid_of[user_id]
        b = np.zeros(self.sim.shape[0], dtype=np.float32)
        b[self.hist[u].indices] = 1
        return self._top(self.sim @ b, b > 0, n)

    def recommend_history(self, movie_ids, n=10):
        b = np.zeros(self.sim.shape[0], dtype=np.float32)
        b[[self.iid_of[m] for m in movie_ids if m in self.iid_of]] = 1
        return self._top(self.sim @ b, b > 0, n)

    def similar(self, movie_id, n=10):
        i = self.iid_of[movie_id]
        s = self.sim[i].toarray().ravel()
        seen = np.zeros(len(s), dtype=bool); seen[i] = True
        return self._top(s, seen, n)
