# Training Data Contract (v1)

Source DB: `/app/data/ml_training.db`  
Table: `training_data`

## Goal
Multi-class classification for signal type in Planet Hunter.

## Target label
- Column: `label` (TEXT)
- Allowed values:
  - `KNOWN_PLANET`
  - `FALSE_POSITIVE`
  - `ECLIPSING_BINARY`
  - `NOISE`

## Feature columns (v1)

### Core signal features (required)
| name | type | source | nulls |
|---|---|---|---|
| `period` | REAL | `build_training_db.py` from `analyses.period` | no |
| `depth` | REAL | `build_training_db.py` from `analyses.depth` | no |
| `snr` | REAL | `build_training_db.py` from `analyses.snr` | no |
| `duration` | REAL | `build_training_db.py` from `analyses.duration` | no |
| `secondary_depth` | REAL | `build_training_db.py` from `analyses.secondary_depth` | no |
| `odd_even_sigma` | REAL | `build_training_db.py` from `analyses.odd_even_sigma` | no |
| `sinusoid_better` | INTEGER (0/1) | `build_training_db.py` from `analyses.sinusoid_better` | no |
| `sectors_ratio` | REAL | `build_training_db.py`, derived from `sectors_detected / sectors_checked` | no |

### Stellar/context features (allowed sparse)
| name | type | source | nulls |
|---|---|---|---|
| `tmag` | REAL | `build_training_db.py` from `stars.tmag` | yes (rare) |
| `teff` | REAL | `build_training_db.py` from `stars.teff` | yes |
| `star_radius` | REAL | `build_training_db.py` from `stars.radius` | yes |

### Excluded from v1 training (too sparse)
| name | reason |
|---|---|
| `planet_radius` | ~86% null |
| `equilibrium_temp` | ~87% null |

## Identifier / split key
- `tic_id` is an object identifier (not a feature for model fitting).
- Splits MUST be group-based by `tic_id` to avoid leakage.

## Split strategy (v1)
- Group split by `tic_id`:
  - train: 70%
  - validation: 15%
  - test: 15%
- Seed: fixed (`42`) for reproducibility.

## Metrics (v1)
- Primary: Macro F1 (`f1_macro`)
- Secondary:
  - per-class precision/recall/F1
  - weighted F1
  - confusion matrix

## Artifacts (v1)
- `data/ml/artifacts/baseline_model.joblib`
- `data/ml/artifacts/metrics.json`
- `data/ml/artifacts/confusion_matrix.csv`
- `data/ml/artifacts/feature_columns.json`

## Repro notes
- Run data build first:
  - `docker exec planet-hunter python3 /app/build_training_db.py`
- Then train:
  - `docker exec planet-hunter python3 /app/scripts/train_baseline.py`
