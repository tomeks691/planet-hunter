#!/usr/bin/env python3
"""Train a reproducible baseline/weighted ML model for Planet Hunter.

Input:
  /app/data/ml_training.db (table: training_data)

Output artifacts:
  /app/data/ml/artifacts/baseline_model.joblib
  /app/data/ml/artifacts/metrics.json
  /app/data/ml/artifacts/confusion_matrix.csv
  /app/data/ml/artifacts/feature_columns.json
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

DB_PATH = Path("/app/data/ml_training.db")
OUT_DIR = Path("/app/data/ml/artifacts")
SEED = 42

FEATURES = [
    "period",
    "depth",
    "snr",
    "duration",
    "secondary_depth",
    "odd_even_sigma",
    "sinusoid_better",
    "sectors_ratio",
    "tmag",
    "teff",
    "star_radius",
]

TARGET = "label"
GROUP = "tic_id"


def load_dataset(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"Dataset not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT {GROUP}, {TARGET}, {', '.join(FEATURES)} FROM training_data"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No rows found in training_data")

    y = np.array([r[TARGET] for r in rows], dtype=object)
    groups = np.array([int(r[GROUP]) for r in rows], dtype=np.int64)

    X = np.array(
        [[float(r[c]) if r[c] is not None else np.nan for c in FEATURES] for r in rows],
        dtype=np.float64,
    )

    return X, y, groups


def split_groups(X, y, groups):
    # 70 train / 30 temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp, y_temp, g_temp = X[temp_idx], y[temp_idx], groups[temp_idx]

    # 15 val / 15 test from temp
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel_idx, test_rel_idx = next(gss2.split(X_temp, y_temp, groups=g_temp))

    X_val, y_val = X_temp[val_rel_idx], y_temp[val_rel_idx]
    X_test, y_test = X_temp[test_rel_idx], y_temp[test_rel_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model_weighted_tuned(X_train, y_train, X_val, y_val):
    # Inverse-frequency sample weights to help minority classes (KNOWN_PLANET/NOISE).
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    candidates = [
        {"learning_rate": 0.05, "max_depth": 8, "max_iter": 400, "l2_regularization": 0.1},
        {"learning_rate": 0.03, "max_depth": 8, "max_iter": 600, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_depth": 10, "max_iter": 500, "l2_regularization": 0.3},
        {"learning_rate": 0.07, "max_depth": 8, "max_iter": 350, "l2_regularization": 0.05},
    ]

    best = None
    best_score = -1.0

    for params in candidates:
        model = HistGradientBoostingClassifier(random_state=SEED, **params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_val_pred = model.predict(X_val)
        score = f1_score(y_val, y_val_pred, average="macro")
        if score > best_score:
            best_score = score
            best = (model, params)

    assert best is not None
    return best[0], best[1], float(best_score)


def save_confusion_matrix(path: Path, labels, cm):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for i, label in enumerate(labels):
            writer.writerow([label, *cm[i].tolist()])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_dataset(DB_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = split_groups(X, y, groups)

    model, best_params, val_best = train_model_weighted_tuned(X_train, y_train, X_val, y_val)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    labels = sorted(np.unique(y))

    metrics = {
        "seed": SEED,
        "rows_total": int(len(y)),
        "rows_train": int(len(y_train)),
        "rows_val": int(len(y_val)),
        "rows_test": int(len(y_test)),
        "features": FEATURES,
        "labels": labels,
        "class_distribution_total": dict(Counter(y.tolist())),
        "class_distribution_train": dict(Counter(y_train.tolist())),
        "best_params": best_params,
        "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "test_f1_macro": float(f1_score(y_test, y_test_pred, average="macro")),
        "val_f1_macro_best_during_search": val_best,
        "val_report": classification_report(y_val, y_val_pred, output_dict=True, zero_division=0),
        "test_report": classification_report(y_test, y_test_pred, output_dict=True, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_test_pred, labels=labels)

    joblib.dump(model, OUT_DIR / "baseline_model.joblib")
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(OUT_DIR / "confusion_matrix.csv", labels, cm)

    print("=== Training complete (weighted+tuned) ===")
    print(f"rows: total={len(y)} train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"best_params={best_params}")
    print(f"val_f1_macro={metrics['val_f1_macro']:.4f}")
    print(f"test_f1_macro={metrics['test_f1_macro']:.4f}")
    print(f"artifacts: {OUT_DIR}")


if __name__ == "__main__":
    main()
