"""MP7 evaluation: macro-F1, graded vs. dev metrics, bootstrap CIs, per-class precision/recall.

compute_graded_metric / compute_dev_metric are local stand-ins: the instructor versions
(and eval_mp7.jsonl) were not provided.
"""
import numpy as np

from prompt_template import LABEL_IDS


def macro_f1(gold, pred):
    f1s = []
    for lbl in LABEL_IDS:
        tp = sum(g == lbl and p == lbl for g, p in zip(gold, pred))
        fp = sum(g != lbl and p == lbl for g, p in zip(gold, pred))
        fn = sum(g == lbl and p != lbl for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def per_class_pr(gold, pred):
    rows = {}
    for lbl in LABEL_IDS:
        tp = sum(g == lbl and p == lbl for g, p in zip(gold, pred))
        fp = sum(g != lbl and p == lbl for g, p in zip(gold, pred))
        fn = sum(g == lbl and p != lbl for g, p in zip(gold, pred))
        rows[lbl] = (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0)
    return rows


def compute_graded_metric(rows, eval_ids):
    """Macro-F1 on the evaluation set (local stand-in; the instructor's version was not provided)."""
    rows = [r for r in rows if r["record_id"] in eval_ids]
    assert {r["record_id"] for r in rows} == set(eval_ids), "graded metric must cover the whole eval set"
    return {"macro_f1": macro_f1([r["gold"] for r in rows], [r["pred"] for r in rows]),
            "grading_eligible": True}


def compute_dev_metric(rows):
    """Macro-F1 on any non-evaluation split; advisory only."""
    return {"macro_f1": macro_f1([r["gold"] for r in rows], [r["pred"] for r in rows]),
            "grading_eligible": False}


def bootstrap_ci(rows_by_seed, eval_ids, B=1000, seed=0):
    """95% percentile CI: resample eval items with replacement, average macro-F1 over seeds."""
    rng = np.random.default_rng(seed)
    ids = list(eval_ids)
    lookup = [{r["record_id"]: r for r in rows} for rows in rows_by_seed]
    stats = []
    for _ in range(B):
        sample = rng.choice(ids, size=len(ids), replace=True)
        stats.append(np.mean([macro_f1([lk[i]["gold"] for i in sample], [lk[i]["pred"] for i in sample])
                              for lk in lookup]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))



def summarize(preds, eval_ids, B=1000):
    out = []
    order = {"prototype": 0, "knn": 1, "llm": 2}
    keys = sorted({(p["classifier"], p["k"]) for p in preds}, key=lambda x: (order.get(x[0], 9), x[1]))
    for clf, k in keys:
        rows = [p for p in preds if p["classifier"] == clf and p["k"] == k]
        seeds = sorted({p["seed"] for p in rows})
        by_seed = [[p for p in rows if p["seed"] == s] for s in seeds]
        f1s = [compute_graded_metric(r, eval_ids)["macro_f1"] for r in by_seed]
        lo, hi = bootstrap_ci(by_seed, eval_ids, B=B)
        out.append({"classifier": clf, "k": k, "mean": float(np.mean(f1s)), "std": float(np.std(f1s)),
                    "ci_lo": lo, "ci_hi": hi, "n_seeds": len(seeds)})
    return out


