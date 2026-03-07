#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planet_hunter.models import AnalysisResult, Classification
from planet_hunter.pipeline import ml_classifier as ml


class DummyStageA:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        import numpy as np

        return np.array([[1.0 - self.p, self.p]], dtype=float)


class DummyStageB:
    def __init__(self, label: str):
        self.label = label

    def predict(self, X):
        return [self.label]


def make_result() -> AnalysisResult:
    return AnalysisResult(
        tic_id=123,
        period=5.0,
        depth=0.01,
        snr=15.0,
        duration=2.0,
        secondary_depth=0.001,
        odd_even_sigma=0.2,
        sinusoid_better=False,
        sectors_checked=3,
        sectors_detected=2,
    )


def run_case(p_planet: float, non_planet_label: str, threshold: float, margin: float):
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        metrics = td_path / "metrics.json"
        metrics.write_text(json.dumps({"threshold": {"threshold": threshold}}), encoding="utf-8")

        original_load = ml.joblib.load

        def fake_load(path):
            path = str(path)
            if "stage_a" in path:
                return DummyStageA(p_planet)
            return DummyStageB(non_planet_label)

        ml.joblib.load = fake_load
        try:
            c = ml.TwoStageMLClassifier(
                stage_a_path=td_path / "dummy_stage_a.joblib",
                stage_b_path=td_path / "dummy_stage_b.joblib",
                metrics_path=metrics,
                uncertainty_margin=margin,
            )
            return c.predict(make_result())
        finally:
            ml.joblib.load = original_load


def main():
    c1 = run_case(p_planet=0.90, non_planet_label="FALSE_POSITIVE", threshold=0.5, margin=0.03)
    assert c1 == Classification.PLANET_CANDIDATE, c1

    c2 = run_case(p_planet=0.20, non_planet_label="FALSE_POSITIVE", threshold=0.5, margin=0.03)
    assert c2 == Classification.FALSE_POSITIVE, c2

    c3 = run_case(p_planet=0.51, non_planet_label="NOISE", threshold=0.5, margin=0.03)
    assert c3 == Classification.MANUAL_REVIEW, c3

    print("ML classifier checks passed")


if __name__ == "__main__":
    main()
