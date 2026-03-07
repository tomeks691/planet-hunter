# ML Production Plan (Planet Hunter)

Owner: Nero  
Status: IN PROGRESS

## Target (production-ready)
- Primary objective: maximize `KNOWN_PLANET` recall while keeping review queue manageable.
- Hard target (v1-prod):
  - `planet recall >= 0.82` on test split
  - `planet precision >= 0.55` on test split
  - `macro F1 >= 0.86`

## Current snapshot
- Best test macro F1 so far: **0.8630** (two-stage engineered).
- Bottleneck: `KNOWN_PLANET` class quality/coverage.

## Workstreams

### 1) Label & data quality audit (Day 1)
- [ ] Audit all `KNOWN_PLANET` rows for feature outliers / inconsistent entries.
- [ ] Add explicit training dataset version stamp + row counts by class.
- [ ] Confirm leakage guards (`tic_id`-group split).
- [ ] Produce `data/ml/reports/data_audit_v1.md`.

### 2) Planet-focused feature engineering (Day 1-2)
- [ ] Add transit-oriented features to training export:
  - duty-cycle variants
  - transit count proxy
  - peak prominence / SDE-like score proxy (from available BLS outputs)
  - odd/even depth ratio variants
- [ ] Keep only features with stable non-null coverage or controlled imputation.

### 3) Modeling strategy (Day 2)
- [ ] Keep two-stage architecture:
  - Stage A: `PLANET vs NON_PLANET`
  - Stage B: `NON_PLANET` subclassifier
- [ ] Add threshold optimization for Stage A based on target recall/precision.
- [ ] Evaluate calibrated probability threshold and abstain window.

### 4) Inference safety (Day 2-3)
- [ ] Add confidence policy:
  - high confidence => auto label
  - low confidence => `MANUAL_REVIEW`
- [ ] Keep rule-based fallback path.

### 5) Acceptance and rollout (Day 3)
- [ ] Comparison report (baseline vs weighted vs two-stage vs engineered).
- [ ] Select locked model + threshold.
- [ ] Document retraining command and artifact locations.

## Immediate next actions (started now)
1. Produce data-audit report focused on `KNOWN_PLANET` quality and missingness.
2. Extend feature set in training scripts with additional transit proxies.
3. Run full benchmark matrix and choose new threshold policy.
