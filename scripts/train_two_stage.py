#!/usr/bin/env python3
"""Train a two-stage classifier for Planet Hunter (with engineered features).

Stage A: PLANET vs NON_PLANET (threshold-tuned on validation)
Stage B: NON_PLANET subtype classifier (FALSE_POSITIVE / ECLIPSING_BINARY / NOISE)
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

DB_PATH = Path('/app/data/ml_training.db')
OUT_DIR = Path('/app/data/ml/artifacts')
SEED = 42

PLANET_LABEL = 'KNOWN_PLANET'

BASE_FEATURES = [
    'period', 'depth', 'snr', 'duration',
    'secondary_depth', 'odd_even_sigma', 'sinusoid_better', 'sectors_ratio',
    'tmag', 'teff', 'star_radius',
]


def safe_div(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill_value=default, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def build_feature_matrix(rows):
    # base columns
    base = {}
    for c in BASE_FEATURES:
        base[c] = np.array([float(r[c]) if r[c] is not None else np.nan for r in rows], dtype=np.float64)

    period = base['period']
    depth = base['depth']
    snr = base['snr']
    duration = base['duration']
    secondary_depth = base['secondary_depth']
    odd_even_sigma = base['odd_even_sigma']
    sectors_ratio = base['sectors_ratio']
    teff = base['teff']
    star_radius = base['star_radius']

    # engineered features focused on planet discrimination
    engineered = {
        'duty_cycle': safe_div(duration, period),
        'secondary_to_primary': safe_div(secondary_depth, depth),
        'depth_times_snr': depth * snr,
        'log1p_snr': np.log1p(np.clip(snr, a_min=0.0, a_max=None)),
        'inv_odd_even': safe_div(np.ones_like(odd_even_sigma), 1.0 + np.abs(odd_even_sigma)),
        'stability_index': sectors_ratio * np.log1p(np.clip(snr, a_min=0.0, a_max=None)),
        'teff_x_radius': teff * star_radius,
    }

    feature_names = BASE_FEATURES + list(engineered.keys())
    X = np.column_stack([base[c] for c in BASE_FEATURES] + [engineered[c] for c in engineered.keys()]).astype(np.float64)
    return X, feature_names


def load_dataset(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT tic_id, label, {', '.join(BASE_FEATURES)} FROM training_data"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError('No rows in training_data')

    X, feature_names = build_feature_matrix(rows)
    y = np.array([r['label'] for r in rows], dtype=object)
    groups = np.array([int(r['tic_id']) for r in rows], dtype=np.int64)
    return X, y, groups, feature_names


def split_groups(X, y, groups):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, tmp_idx = next(gss1.split(X, y, groups=groups))

    X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
    X_tmp, y_tmp, g_tmp = X[tmp_idx], y[tmp_idx], groups[tmp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_rel, te_rel = next(gss2.split(X_tmp, y_tmp, groups=g_tmp))

    X_va, y_va = X_tmp[va_rel], y_tmp[va_rel]
    X_te, y_te = X_tmp[te_rel], y_tmp[te_rel]
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def train_stage_a(X_train, y_train):
    y_bin = np.where(y_train == PLANET_LABEL, 1, 0)
    weights = compute_sample_weight(class_weight='balanced', y=y_bin)

    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=10,
        max_iter=700,
        l2_regularization=0.1,
        random_state=SEED,
    )
    model.fit(X_train, y_bin, sample_weight=weights)
    return model


def pick_threshold_for_planet_recall(model, X_val, y_val, min_recall=0.82):
    y_true = np.where(y_val == PLANET_LABEL, 1, 0)
    probs = model.predict_proba(X_val)[:, 1]

    candidates = np.linspace(0.12, 0.80, 69)
    best = None
    for th in candidates:
        y_pred = (probs >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        # prioritize planet f1 while holding minimum recall
        score = f1 if r >= min_recall else -1.0
        if best is None or score > best['score']:
            best = {'threshold': float(th), 'precision': float(p), 'recall': float(r), 'f1': float(f1), 'score': float(score)}

    if best is None or best['score'] < 0:
        # fallback to best planet-F1 regardless recall
        best = None
        for th in candidates:
            y_pred = (probs >= th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            if best is None or f1 > best['f1']:
                best = {'threshold': float(th), 'precision': float(p), 'recall': float(r), 'f1': float(f1), 'score': float(f1)}

    return best


def train_stage_b(X_train, y_train):
    mask = y_train != PLANET_LABEL
    X_np = X_train[mask]
    y_np = y_train[mask]

    weights = compute_sample_weight(class_weight='balanced', y=y_np)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        l2_regularization=0.1,
        random_state=SEED,
    )
    model.fit(X_np, y_np, sample_weight=weights)
    return model


def predict_two_stage(stage_a, stage_b, X, threshold):
    planet_prob = stage_a.predict_proba(X)[:, 1]
    is_planet = planet_prob >= threshold
    pred = np.empty(shape=(len(X),), dtype=object)
    pred[is_planet] = PLANET_LABEL

    if np.any(~is_planet):
        pred[~is_planet] = stage_b.predict(X[~is_planet])
    return pred


def save_cm(path: Path, labels, cm):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['true\\pred', *labels])
        for i, label in enumerate(labels):
            w.writerow([label, *cm[i].tolist()])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, feature_names = load_dataset(DB_PATH)
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_groups(X, y, groups)

    stage_a = train_stage_a(X_tr, y_tr)
    th = pick_threshold_for_planet_recall(stage_a, X_va, y_va, min_recall=0.82)
    stage_b = train_stage_b(X_tr, y_tr)

    y_va_pred = predict_two_stage(stage_a, stage_b, X_va, threshold=th['threshold'])
    y_te_pred = predict_two_stage(stage_a, stage_b, X_te, threshold=th['threshold'])

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_te, y_te_pred, labels=labels)

    metrics = {
        'seed': SEED,
        'features': feature_names,
        'rows_total': int(len(y)),
        'rows_train': int(len(y_tr)),
        'rows_val': int(len(y_va)),
        'rows_test': int(len(y_te)),
        'planet_threshold': th,
        'val_f1_macro': float(f1_score(y_va, y_va_pred, average='macro')),
        'test_f1_macro': float(f1_score(y_te, y_te_pred, average='macro')),
        'val_report': classification_report(y_va, y_va_pred, output_dict=True, zero_division=0),
        'test_report': classification_report(y_te, y_te_pred, output_dict=True, zero_division=0),
    }

    joblib.dump(stage_a, OUT_DIR / 'two_stage_stage_a.joblib')
    joblib.dump(stage_b, OUT_DIR / 'two_stage_stage_b.joblib')
    (OUT_DIR / 'two_stage_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    save_cm(OUT_DIR / 'two_stage_confusion_matrix.csv', labels, cm)

    print('=== Two-stage training complete (engineered) ===')
    print(f"planet threshold={th['threshold']:.3f} val_planet_recall={th['recall']:.3f} val_planet_precision={th['precision']:.3f} val_planet_f1={th['f1']:.3f}")
    print(f"val_f1_macro={metrics['val_f1_macro']:.4f}")
    print(f"test_f1_macro={metrics['test_f1_macro']:.4f}")


if __name__ == '__main__':
    main()
