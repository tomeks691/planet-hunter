#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
import joblib

DB_PATH = Path('/app/data/ml_training.db')
OUT_DIR = Path('/app/data/ml/artifacts')
SEED = 42
PLANET = 'KNOWN_PLANET'

BASE_FEATURES = [
    'period', 'depth', 'snr', 'duration',
    'secondary_depth', 'odd_even_sigma', 'sinusoid_better', 'sectors_ratio',
    'tmag', 'teff', 'star_radius',
]


def safe_div(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill_value=default, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[m] = a[m] / b[m]
    return out


def build_matrix(rows):
    base = {c: np.array([float(r[c]) if r[c] is not None else np.nan for r in rows], dtype=np.float64) for c in BASE_FEATURES}
    period = base['period']
    depth = base['depth']
    snr = base['snr']
    duration = base['duration']
    secondary = base['secondary_depth']
    odd = base['odd_even_sigma']
    sectors = base['sectors_ratio']
    teff = base['teff']
    radius = base['star_radius']

    engineered = {
        'duty_cycle': safe_div(duration, period),
        'secondary_to_primary': safe_div(secondary, depth),
        'depth_times_snr': depth * snr,
        'log1p_snr': np.log1p(np.clip(snr, 0.0, None)),
        'inv_odd_even': safe_div(np.ones_like(odd), 1.0 + np.abs(odd)),
        'stability_index': sectors * np.log1p(np.clip(snr, 0.0, None)),
        'teff_x_radius': teff * radius,
    }

    feats = BASE_FEATURES + list(engineered.keys())
    X = np.column_stack([base[c] for c in BASE_FEATURES] + [engineered[k] for k in engineered.keys()]).astype(np.float64)
    return X, feats


def clip_by_train_quantiles(X_train, X_val, X_test, low=0.005, high=0.995):
    lo = np.nanquantile(X_train, low, axis=0)
    hi = np.nanquantile(X_train, high, axis=0)
    return np.clip(X_train, lo, hi), np.clip(X_val, lo, hi), np.clip(X_test, lo, hi)


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT tic_id,label,{','.join(BASE_FEATURES)} FROM training_data").fetchall()
    conn.close()
    X, features = build_matrix(rows)
    y = np.array([r['label'] for r in rows], dtype=object)
    g = np.array([int(r['tic_id']) for r in rows], dtype=np.int64)
    return X, y, g, features


def split_groups(X, y, g):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr, tmp = next(gss1.split(X, y, groups=g))
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    Xtmp, ytmp, gtmp = X[tmp], y[tmp], g[tmp]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va, te = next(gss2.split(Xtmp, ytmp, groups=gtmp))
    return Xtr, ytr, Xtmp[va], ytmp[va], Xtmp[te], ytmp[te]


def oversample_planet(X, y, factor=3):
    mask = y == PLANET
    Xp, yp = X[mask], y[mask]
    if len(Xp) == 0:
        return X, y
    X_aug = np.concatenate([X] + [Xp] * (factor - 1), axis=0)
    y_aug = np.concatenate([y] + [yp] * (factor - 1), axis=0)
    idx = np.arange(len(y_aug))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    return X_aug[idx], y_aug[idx]


def choose_threshold(stage_a, Xv, yv, target_recall=0.85):
    p = stage_a.predict_proba(Xv)[:, 1]
    yt = (yv == PLANET).astype(int)
    best = None
    for th in np.linspace(0.05, 0.40, 71):
        yh = (p >= th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(yt, yh, average='binary', zero_division=0)
        # maximize recall first, then f1
        score = (rc, f1, pr)
        if best is None:
            best = (th, pr, rc, f1, score)
        else:
            if rc >= target_recall and best[2] < target_recall:
                best = (th, pr, rc, f1, score)
            elif (rc >= target_recall and best[2] >= target_recall and f1 > best[3]):
                best = (th, pr, rc, f1, score)
            elif best[2] < target_recall and rc > best[2]:
                best = (th, pr, rc, f1, score)
    return {'threshold': float(best[0]), 'precision': float(best[1]), 'recall': float(best[2]), 'f1': float(best[3])}


def predict_two_stage(stage_a, stage_b, X, th):
    p = stage_a.predict_proba(X)[:, 1]
    is_planet = p >= th
    pred = np.empty(len(X), dtype=object)
    pred[is_planet] = PLANET
    if np.any(~is_planet):
        pred[~is_planet] = stage_b.predict(X[~is_planet])
    return pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, g, features = load_data()
    Xtr, ytr, Xva, yva, Xte, yte = split_groups(X, y, g)
    Xtr, Xva, Xte = clip_by_train_quantiles(Xtr, Xva, Xte)

    # Stage A high-recall
    ytr_bin = np.where(ytr == PLANET, 1, 0)
    Xtr_a, ytr_a = oversample_planet(Xtr, ytr, factor=3)
    ytr_a_bin = np.where(ytr_a == PLANET, 1, 0)
    w_a = compute_sample_weight(class_weight='balanced', y=ytr_a_bin)
    stage_a = HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=10, max_iter=900, l2_regularization=0.05, random_state=SEED
    )
    stage_a.fit(Xtr_a, ytr_a_bin, sample_weight=w_a)

    th = choose_threshold(stage_a, Xva, yva, target_recall=0.85)

    # Stage B regular non-planet classifier
    mask_np = ytr != PLANET
    Xnp, ynp = Xtr[mask_np], ytr[mask_np]
    w_b = compute_sample_weight(class_weight='balanced', y=ynp)
    stage_b = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=600, l2_regularization=0.1, random_state=SEED
    )
    stage_b.fit(Xnp, ynp, sample_weight=w_b)

    yva_pred = predict_two_stage(stage_a, stage_b, Xva, th['threshold'])
    yte_pred = predict_two_stage(stage_a, stage_b, Xte, th['threshold'])

    m = {
        'mode': 'high_recall',
        'threshold': th,
        'val_macro_f1': float(f1_score(yva, yva_pred, average='macro')),
        'test_macro_f1': float(f1_score(yte, yte_pred, average='macro')),
        'val_report': classification_report(yva, yva_pred, output_dict=True, zero_division=0),
        'test_report': classification_report(yte, yte_pred, output_dict=True, zero_division=0),
        'features': features,
    }

    joblib.dump(stage_a, OUT_DIR / 'two_stage_hr_stage_a.joblib')
    joblib.dump(stage_b, OUT_DIR / 'two_stage_hr_stage_b.joblib')
    (OUT_DIR / 'two_stage_high_recall_metrics.json').write_text(json.dumps(m, indent=2), encoding='utf-8')

    kp = m['test_report'][PLANET]
    print('=== High-recall two-stage done ===')
    print('threshold', th)
    print('test_macro_f1', round(m['test_macro_f1'], 4))
    print('test_planet_precision', round(kp['precision'], 4))
    print('test_planet_recall', round(kp['recall'], 4))
    print('test_planet_f1', round(kp['f1-score'], 4))


if __name__ == '__main__':
    main()
