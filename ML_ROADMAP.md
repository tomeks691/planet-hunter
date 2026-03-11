# Planet Hunter — ML Roadmap (v1)

Goal: replace/augment rule-based planet signal analysis with an ML-assisted classifier.

## Phase 0 — Define target (must do first)
- Problem type: multi-class classification of candidate signals.
- Initial classes (v1):
  - `PLANET_CANDIDATE`
  - `ECLIPSING_BINARY`
  - `VARIABLE_STAR`
  - `NOISE/ARTIFACT`
- Success metric target (v1): macro F1 >= 0.80 on validation set.

## Phase 1 — Data foundation
- Build training dataset from historical runs + known labeled objects.
- Store as a reproducible dataset file (CSV/Parquet).
- Ensure no data leakage (split by object/star, not random rows).

## Phase 2 — Feature engineering
- Keep existing signal/statistical features from current pipeline.
- Add robust features from periodogram + transit-shape stats.
- Version feature schema.

## Phase 3 — Baseline ML
- Start with strong baseline model: XGBoost or LightGBM.
- Train/val/test split with fixed random seed.
- Log metrics + confusion matrix.

## Phase 4 — Inference integration
- Add model inference as optional stage in current pipeline.
- Keep fallback to old rule-based path.
- Add confidence threshold + abstain logic.

## Phase 5 — Monitoring and iteration
- Track false positives / false negatives.
- Weekly retrain loop from newly verified labels.
- Keep model/version registry in project.

---

## First task (do now)
Create a data contract for training:
1. New file: `data/ml/training_schema.md`
2. Define every training column:
   - name
   - type
   - source (which script/module)
   - allowed nulls
3. Define target label mapping and split strategy.

Why first: without a stable schema, ML work becomes chaos and non-reproducible.
