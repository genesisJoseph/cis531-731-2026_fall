"""PS8 interim analysis for the IEEE-CIS Fraud Detection project (Genesis Joseph & Tatenda, CIS 731).

Produces the data card, three exploratory figures (missingness, weekly fraud-rate drift with
95% Wilson intervals, transaction amount by class), a UMAP sensitivity sweep with a fixed
random_state, and advisory naive baselines on a time-ordered validation split.
The test period is never touched here; it is reserved for the final evaluation.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
SEED = 42
CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + [f"M{i}" for i in range(1, 10)]
FILES = ["train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv"]
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "xtick.color": "#52514e", "ytick.color": "#52514e"})


# ------------------------------------------------------------------ loading
def load_train(data_dir=DATA):
    """Train transactions + identity (left join on TransactionID). Numerics as float32 to save memory."""
    data_dir = Path(data_dir)
    cols = pd.read_csv(data_dir / "train_transaction.csv", nrows=0).columns
    dtypes = {c: ("object" if c in CATEGORICAL else "float32") for c in cols
              if c not in ("TransactionID", "isFraud", "TransactionDT")}
    tx = pd.read_csv(data_dir / "train_transaction.csv", dtype=dtypes)
    idt = pd.read_csv(data_dir / "train_identity.csv")
    return tx, idt


def count_rows_cols(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip().split(",")
        rows = sum(1 for _ in f)
    return rows, len(header)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------- data card
def data_card(tx, idt, data_dir=DATA):
    data_dir = Path(data_dir)
    tables = {}
    for name in FILES:
        p = data_dir / name
        if p.exists():
            r, c = count_rows_cols(p)
            tables[name] = {"rows": r, "columns": c, "bytes": p.stat().st_size, "sha256": sha256(p)}
    total = sum(t["rows"] for t in tables.values())
    card = {
        "tables": tables, "total_rows": total,
        "train_fraud_count": int(tx["isFraud"].sum()),
        "train_fraud_rate": float(tx["isFraud"].mean()),
        "identity_coverage": float(tx["TransactionID"].isin(idt["TransactionID"]).mean()),
        "transaction_dt_days": float((tx["TransactionDT"].max() - tx["TransactionDT"].min()) / 86400),
        "validity_checks": {
            "TransactionID_unique": bool(tx["TransactionID"].is_unique),
            "TransactionID_non_null": bool(tx["TransactionID"].notna().all()),
            "isFraud_in_0_1": bool(tx["isFraud"].isin([0, 1]).all()),
            "TransactionAmt_positive": bool((tx["TransactionAmt"] > 0).all()),
            "identity_one_to_one": bool(idt["TransactionID"].is_unique),
        },
    }
    return card


def feature_group(col):
    for g in ("card", "addr", "dist", "P_email", "R_email"):
        if col.startswith(g):
            return g
    for g in ("C", "D", "M", "V"):
        if col[0] == g and col[1:].isdigit():
            return g
    return "other"


# ----------------------------------------------------------------- figures
def fig_missingness(tx):
    miss = tx.drop(columns=["TransactionID", "isFraud", "TransactionDT"]).isna().mean()
    grp = miss.groupby(miss.index.map(feature_group)).mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(grp.index, 100 * grp.values, color=BLUE, height=0.6)
    for y, v in enumerate(grp.values):
        ax.text(100 * v + 1, y, f"{100 * v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set(xlabel="mean % missing per feature group", xlim=(0, 105),
           title="Fig. 1: Missingness by feature group (train_transaction)")
    fig.tight_layout(); fig.savefig(FIG / "fig1_missingness.png", dpi=150); plt.close(fig)
    return {k: round(float(v), 4) for k, v in grp.items()}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def time_split(tx, train=0.7, val=0.1):
    """Time-ordered split: first 70% train, next 10% validation, last 20% test (held out)."""
    order = tx["TransactionDT"].sort_values()
    t1, t2 = order.iloc[int(len(order) * train)], order.iloc[int(len(order) * (train + val))]
    return t1, t2


def fig_drift(tx, t1, t2):
    week = (tx["TransactionDT"] // (7 * 86400)).astype(int)
    g = tx.groupby(week)["isFraud"].agg(["sum", "count"])
    lo, hi = wilson(g["sum"].values, g["count"].values)
    rate = g["sum"] / g["count"]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.fill_between(g.index, 100 * lo, 100 * hi, color=BLUE, alpha=0.2, lw=0, label="95% Wilson interval")
    ax.plot(g.index, 100 * rate, color=BLUE, lw=2, label="weekly fraud rate")
    for t, name in ((t1, "train | val"), (t2, "val | test")):
        ax.axvline(t / (7 * 86400), color=GREY, ls="--", lw=1)
        ax.text(t / (7 * 86400), ax.get_ylim()[1] * 0.97, f" {name}", fontsize=7, color="#52514e", va="top")
    ax.set(xlabel="week since start (TransactionDT / 7 days)", ylabel="% of transactions fraudulent",
           title="Fig. 2: Weekly fraud rate over time (cohort drift)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig2_drift.png", dpi=150); plt.close(fig)
    return {"weeks": int(len(g)), "min_rate": float(rate.min()), "max_rate": float(rate.max()),
            "first_week_rate": float(rate.iloc[0]), "last_week_rate": float(rate.iloc[-1])}


def fig_amount(tx):
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(-1, 4.5, 60)
    for lbl, color, name in ((0, BLUE, "not fraud"), (1, ORANGE, "fraud")):
        v = np.log10(tx.loc[tx["isFraud"] == lbl, "TransactionAmt"].clip(lower=0.1))
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=color, label=name)
    ax.set(xlabel="log10(TransactionAmt)", ylabel="density",
           title="Fig. 3: Transaction amount by class")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig3_amount.png", dpi=150); plt.close(fig)
    q = tx.groupby("isFraud")["TransactionAmt"].median()
    return {"median_amt_not_fraud": float(q.get(0, np.nan)), "median_amt_fraud": float(q.get(1, np.nan))}


# ---------------------------------------------------------- features + UMAP
def numeric_features(tx, max_missing=0.5):
    num = [c for c in tx.columns if c not in ("TransactionID", "isFraud", "TransactionDT")
           and c not in CATEGORICAL and tx[c].dtype != object]
    return [c for c in num if tx[c].isna().mean() <= max_missing]


def fig_umap_sweep(tx, n=6000, neighbors=(5, 15, 50), min_dists=(0.0, 0.1, 0.5)):
    import umap
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    parts = [tx[tx["isFraud"] == k] for k in (0, 1)]                   # balanced so fraud is visible
    sample = pd.concat([d.sample(min(len(d), n // 2), random_state=SEED) for d in parts])
    feats = numeric_features(tx)
    X = sample[feats].copy()
    X["TransactionAmt"] = np.log10(X["TransactionAmt"].clip(lower=0.1))
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    y = sample["isFraud"].values
    fig, axes = plt.subplots(len(neighbors), len(min_dists), figsize=(8, 8))
    for i, nn in enumerate(neighbors):
        for j, md in enumerate(min_dists):
            Z = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=SEED).fit_transform(X)
            ax = axes[i, j]
            ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=2, color=BLUE, alpha=0.4, lw=0)
            ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=2, color=ORANGE, alpha=0.6, lw=0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"n_neighbors={nn}, min_dist={md}", fontsize=8)
    fig.suptitle(f"Fig. 4: UMAP sensitivity sweep (random_state={SEED}; "
                 f"{len(sample):,} balanced sample; blue = not fraud, orange = fraud)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_umap_sweep.png", dpi=150); plt.close(fig)
    return {"sample_size": int(len(sample)), "n_features": len(feats), "random_state": SEED,
            "n_neighbors": list(neighbors), "min_dist": list(min_dists)}


# --------------------------------------------------------------- baselines
def baselines(tx, t1, t2):
    """Advisory metrics on the validation period only. The test period is not used."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    train = tx[tx["TransactionDT"] < t1]
    val = tx[(tx["TransactionDT"] >= t1) & (tx["TransactionDT"] < t2)]
    num = numeric_features(tx)
    cat = [c for c in ("ProductCD", "card4", "card6") if c in tx.columns]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), cat)])
    model = make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(train[num + cat], train["isFraud"])
    p = model.predict_proba(val[num + cat])[:, 1]
    yv = val["isFraud"].values
    return {
        "split": {"train_rows": int(len(train)), "val_rows": int(len(val)),
                  "test_rows_held_out": int((tx["TransactionDT"] >= t2).sum()),
                  "val_fraud_rate": float(yv.mean())},
        "majority_class": {"pr_auc": float(average_precision_score(yv, np.zeros_like(yv, dtype=float))),
                           "roc_auc": 0.5, "note": "always predicts not-fraud; PR-AUC equals the fraud rate"},
        "logistic_regression": {"pr_auc": float(average_precision_score(yv, p)),
                                "roc_auc": float(roc_auc_score(yv, p)),
                                "features": len(num) + len(cat), "class_weight": "balanced"},
        "grading_eligible": False,
    }


def manifest():
    import importlib.metadata as md
    vers = {}
    for p in ["pandas", "numpy", "scikit-learn", "umap-learn", "matplotlib", "xgboost", "kaggle"]:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = "not installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
            "random_state": SEED, "source": "Kaggle competition ieee-fraud-detection (IEEE-CIS / Vesta), "
                                             "original files, downloaded via Kaggle API"}


def run_all(data_dir=DATA):
    tx, idt = load_train(data_dir)
    t1, t2 = time_split(tx)
    out = {"data_card": data_card(tx, idt, data_dir),
           "fig1_missingness": fig_missingness(tx),
           "fig2_drift": fig_drift(tx, t1, t2),
           "fig3_amount": fig_amount(tx),
           "fig4_umap": fig_umap_sweep(tx),
           "baselines": baselines(tx, t1, t2),
           "manifest": manifest()}
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    return out
