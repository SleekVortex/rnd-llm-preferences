"""Признаки пар, срезы, калибровка и метрики устойчивости для ДЗ3."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["A", "B", "tie"]
HW2 = Path("experiments/reports/hw2/qwen")
HW3 = Path("experiments/reports/hw3")
LIMIT = 192  # столько токенов каждого ответа видела модель в ДЗ2

REFUSAL = re.compile(r"I'm sorry|I am sorry|I cannot|I can't|As an AI|I'm not able|I apologize", re.I)
UNKNOWN = re.compile(r"(couldn't|could not|can't|cannot|unable to) find|(don't|do not) have (any )?"
                     r"(specific |real-time )?information|not familiar with|no information|not aware of|"
                     r"I'm not sure who|not a (real|recognized|known)", re.I)


def build_features(valid_jsonl, tokens_csv, meta_csv="data/prepared/validation.csv"):
    """Признаки без текстов: длины в токенах, обрезка, отказы и ответы «не знаю»."""
    texts = pd.read_json(valid_jsonl, lines=True, dtype={"id": str}).set_index("id")
    tokens = pd.read_csv(tokens_csv, dtype={"id": str}).set_index("id")
    texts = texts.loc[tokens.index, ["prompt", "response_a", "response_b"]].map("\n".join)
    meta = pd.read_csv(meta_csv, dtype={"id": str}).set_index("id").loc[tokens.index]
    f = tokens.copy()
    f["turns"] = meta.turns
    f["same_responses"] = meta.same_responses
    f["cut"] = (f.response_a_tokens > LIMIT).astype(int) + (f.response_b_tokens > LIMIT)
    f["log_len_ratio"] = np.log((f.response_a_tokens + 1) / (f.response_b_tokens + 1))
    for side in "ab":
        head = texts["response_" + side].str[:400]
        f["refusal_" + side] = head.map(lambda s: bool(REFUSAL.search(s[:300])))
        f["unknown_" + side] = head.map(lambda s: bool(UNKNOWN.search(s)))
    return f


def load(features=HW3 / "features.csv", probs=HW2 / "large_lora.csv"):
    """Прогнозы выбранной модели ДЗ2 вместе с истинным классом, панелью и признаками."""
    panel = pd.read_csv(HW2 / "panel.csv", dtype={"id": str}).set_index("id")
    p = pd.read_csv(probs, dtype={"id": str}).set_index("id").loc[panel.index]
    f = pd.read_csv(features, dtype={"id": str}).set_index("id").loc[panel.index]
    d = panel.join(p).join(f)
    return add_scores(d, d[CLASSES].to_numpy())


def add_scores(d, p):
    y = d.target.to_numpy()
    d = d.copy()
    d["loss"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    d["pred"] = p.argmax(1)
    d["correct"] = d.pred == d.target
    d["confidence"] = p.max(1)
    return d


def bootstrap_mean(values, level=.95, resamples=2000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return lo, hi


def slice_table(d, by, order=None):
    """Размер группы, log loss с 95% интервалом, accuracy и доля ничьих.

    В выборке одна пара на группу component, поэтому бутстрап идет по парам."""
    rows = []
    for key, g in d.groupby(by, observed=True):
        lo, hi = bootstrap_mean(g.loss)
        rows.append({"group": key, "rows": len(g), "log_loss": g.loss.mean(), "low": lo, "high": hi,
                     "accuracy": g.correct.mean(), "tie_share": (g.target == 2).mean(),
                     "mean_p_tie": g.tie.mean()})
    t = pd.DataFrame(rows).set_index("group")
    return t.loc[order] if order is not None else t


def calibration(d, edges=(0, .4, .5, .6, .7, 1.)):
    bins = pd.cut(d.confidence, edges)
    t = d.groupby(bins, observed=True).agg(rows=("correct", "size"), confidence=("confidence", "mean"),
                                           accuracy=("correct", "mean"))
    ece = (t.rows * (t.confidence - t.accuracy).abs()).sum() / t.rows.sum()
    return t, ece


def swap_summary(original, swapped, y):
    """Сравнение прогнозов до и после перестановки ответов.

    swapped уже возвращен к исходной разметке: столбец A относится к исходному ответу A,
    который во втором прогоне стоял вторым. shift > 0 значит, что ответу выгодно стоять первым."""
    po, ps = original[CLASSES].to_numpy(), swapped[CLASSES].to_numpy()
    loss = lambda p: -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    shift = (po[:, 0] - ps[:, 0] + ps[:, 1] - po[:, 1]) / 2
    lo, hi = bootstrap_mean(shift)
    both = (po + ps) / 2
    return {"rows": len(y), "log_loss": loss(po).mean(), "log_loss_swapped": loss(ps).mean(),
            "log_loss_average": loss(both).mean(),
            "same_argmax": (po.argmax(1) == ps.argmax(1)).mean(),
            "mean_abs_change": np.abs(po - ps).sum(1).mean() / 2,
            "first_position_shift": shift.mean(), "shift_low": lo, "shift_high": hi,
            "pred_A_original": (po.argmax(1) == 0).mean(), "pred_A_swapped": (ps.argmax(1) == 0).mean()}
