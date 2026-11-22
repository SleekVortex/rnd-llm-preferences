import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def quality(y, p):
    p = np.asarray(p, dtype=float)
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    p = p / p.sum(axis=1, keepdims=True)
    return {"log_loss": log_loss(y, p, labels=[0, 1, 2]),
            "accuracy": accuracy_score(y, p.argmax(1)),
            "macro_f1": f1_score(y, p.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)}


def paired_loss(y, first, second, group, level=0.95):
    def losses(p):
        p = np.asarray(p, dtype=float)
        p = p / p.sum(axis=1, keepdims=True)
        eps = np.finfo(p.dtype).eps
        return -np.log(np.clip(p[np.arange(len(y)), y], eps, 1-eps))
    delta = losses(first) - losses(second)
    g = pd.DataFrame({"group": np.asarray(group), "d": delta}).groupby("group").d
    totals, counts = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(42)
    means = []
    for _ in range(100):
        take = rng.integers(len(counts), size=(100, len(counts)))
        means.extend(totals[take].sum(1) / counts[take].sum(1))
    lo, hi = np.quantile(means, [(1-level)/2, (1+level)/2])
    return {"delta": delta.mean(), "low": lo, "high": hi, "level": level, "rows": len(y), "groups": len(counts)}
