import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from planet_hunter.models import AnalysisResult, Classification
from planet_hunter.config import ML_MODEL_VERSION

log = logging.getLogger(__name__)

PLANET_LABEL = "KNOWN_PLANET"


class TwoStageMLClassifier:
    def __init__(
        self,
        stage_a_path: Path,
        stage_b_path: Path,
        metrics_path: Path,
        uncertainty_margin: float = 0.03,
    ):
        self.stage_a_path = stage_a_path
        self.stage_b_path = stage_b_path
        self.metrics_path = metrics_path
        self.uncertainty_margin = float(uncertainty_margin)

        self._loaded = False
        self._stage_a = None
        self._stage_b = None
        self._threshold = 0.14

    def _load(self):
        if self._loaded:
            return
        self._stage_a = joblib.load(self.stage_a_path)
        self._stage_b = joblib.load(self.stage_b_path)

        payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        th = payload.get("threshold", {}).get("threshold")
        if th is not None:
            self._threshold = float(th)
        self._loaded = True
        log.info("ML classifier loaded (threshold=%.3f)", self._threshold)

    @staticmethod
    def _safe_div(a: float, b: float, default: float = 0.0) -> float:
        if a is None or b is None or abs(b) < 1e-12:
            return default
        return float(a) / float(b)

    def _build_features(self, result: AnalysisResult) -> np.ndarray:
        period = float(result.period) if result.period is not None else np.nan
        depth = float(result.depth) if result.depth is not None else np.nan
        snr = float(result.snr) if result.snr is not None else np.nan
        duration = float(result.duration) if result.duration is not None else np.nan
        secondary = float(result.secondary_depth) if result.secondary_depth is not None else np.nan
        odd = float(result.odd_even_sigma) if result.odd_even_sigma is not None else np.nan
        sinusoid = 1.0 if bool(result.sinusoid_better) else 0.0

        sectors_ratio = np.nan
        if result.sectors_checked and result.sectors_checked > 0:
            sectors_ratio = float(result.sectors_detected) / float(result.sectors_checked)

        # Not currently available on AnalysisResult -> keep NaN
        tmag = np.nan
        teff = np.nan
        star_radius = np.nan

        duty_cycle = self._safe_div(duration, period)
        secondary_to_primary = self._safe_div(secondary, depth)
        depth_times_snr = (depth * snr) if np.isfinite(depth) and np.isfinite(snr) else np.nan
        log1p_snr = np.log1p(max(snr, 0.0)) if np.isfinite(snr) else np.nan
        inv_odd_even = self._safe_div(1.0, 1.0 + abs(odd)) if np.isfinite(odd) else np.nan
        stability_index = sectors_ratio * log1p_snr if np.isfinite(sectors_ratio) and np.isfinite(log1p_snr) else np.nan
        teff_x_radius = teff * star_radius if np.isfinite(teff) and np.isfinite(star_radius) else np.nan

        row = [
            period,
            depth,
            snr,
            duration,
            secondary,
            odd,
            sinusoid,
            sectors_ratio,
            tmag,
            teff,
            star_radius,
            duty_cycle,
            secondary_to_primary,
            depth_times_snr,
            log1p_snr,
            inv_odd_even,
            stability_index,
            teff_x_radius,
        ]
        return np.array([row], dtype=np.float64)

    def predict(self, result: AnalysisResult) -> Optional[Classification]:
        if result.period is None or result.depth is None or result.snr is None or result.duration is None:
            return None

        try:
            self._load()
            X = self._build_features(result)
            p_planet = float(self._stage_a.predict_proba(X)[0, 1])
            result.ml_planet_score = p_planet
            result.ml_model_version = ML_MODEL_VERSION
            result.ml_decision_source = "two_stage"

            # Uncertainty band around decision threshold -> manual review
            if abs(p_planet - self._threshold) <= self.uncertainty_margin:
                return Classification.MANUAL_REVIEW

            if p_planet >= self._threshold:
                return Classification.PLANET_CANDIDATE

            other = self._stage_b.predict(X)[0]
            if other == "FALSE_POSITIVE":
                return Classification.FALSE_POSITIVE
            if other == "ECLIPSING_BINARY":
                return Classification.ECLIPSING_BINARY
            if other == "NOISE":
                return Classification.NOISE
            return Classification.MANUAL_REVIEW
        except Exception as e:
            log.warning("ML prediction failed, fallback to rules: %s", e)
            result.ml_decision_source = "fallback_rules"
            return None
