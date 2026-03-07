import logging
from planet_hunter.models import Classification, AnalysisResult
from planet_hunter.config import (
    SNR_MINIMUM, DEPTH_MAX_PLANET, DEPTH_MIN_SIGNAL,
    SECONDARY_ECLIPSE_RATIO, ODD_EVEN_SIGMA,
    ML_CLASSIFIER_ENABLED, ML_CLASSIFIER_UNCERTAINTY_MARGIN,
    ML_STAGE_A_PATH, ML_STAGE_B_PATH, ML_METRICS_PATH,
)
from planet_hunter.pipeline.ml_classifier import TwoStageMLClassifier

log = logging.getLogger(__name__)

_ml = None
if ML_CLASSIFIER_ENABLED:
    _ml = TwoStageMLClassifier(
        stage_a_path=ML_STAGE_A_PATH,
        stage_b_path=ML_STAGE_B_PATH,
        metrics_path=ML_METRICS_PATH,
        uncertainty_margin=ML_CLASSIFIER_UNCERTAINTY_MARGIN,
    )


def classify(result: AnalysisResult) -> Classification:
    """Decision-tree classifier for transit signals.

    Modifies result in-place with the classification and returns it.
    """
    if not result.ml_decision_source:
        result.ml_decision_source = "rules"

    log.info(
        "Classifying TIC %d: SNR=%.1f, depth=%.5f, sinusoid=%s, "
        "secondary=%.3f, odd_even=%.1fσ, sectors=%d/%d",
        result.tic_id,
        result.snr or 0,
        result.depth or 0,
        result.sinusoid_better,
        result.secondary_depth or 0,
        result.odd_even_sigma or 0,
        result.sectors_detected,
        result.sectors_checked,
    )

    # 0. Physically impossible: transit duration > 20% of period -> FALSE_POSITIVE
    if (result.period and result.duration and
            (result.duration / 24) / result.period > 0.20):
        log.info("Duration/period=%.1f%% > 20%% -> FALSE_POSITIVE",
                 (result.duration / 24) / result.period * 100)
        return Classification.FALSE_POSITIVE

    # 1. SNR too low -> NOISE
    if result.snr is None or result.snr < SNR_MINIMUM:
        log.info("SNR %.1f < %.1f -> NOISE", result.snr or 0, SNR_MINIMUM)
        return Classification.NOISE

    # 2. ML classifier (if enabled) can override remaining heuristic tree
    if _ml is not None:
        ml_cls = _ml.predict(result)
        if ml_cls is not None:
            log.info("ML classifier -> %s", ml_cls.value)
            return ml_cls

    # 3. Sinusoidal variation fits better -> VARIABLE_STAR
    if result.sinusoid_better:
        log.info("Sinusoid fits better -> VARIABLE_STAR")
        return Classification.VARIABLE_STAR

    # 3. Transit too deep -> ECLIPSING_BINARY
    if result.depth is not None and result.depth > DEPTH_MAX_PLANET:
        log.info("Depth %.4f > %.4f -> ECLIPSING_BINARY", result.depth, DEPTH_MAX_PLANET)
        return Classification.ECLIPSING_BINARY

    # 4. Significant secondary eclipse -> ECLIPSING_BINARY
    if result.secondary_depth is not None and result.secondary_depth > SECONDARY_ECLIPSE_RATIO:
        log.info("Secondary/primary ratio %.3f > %.3f -> ECLIPSING_BINARY",
                 result.secondary_depth, SECONDARY_ECLIPSE_RATIO)
        return Classification.ECLIPSING_BINARY

    # 5. Odd-even depth difference -> ECLIPSING_BINARY
    if result.odd_even_sigma is not None and result.odd_even_sigma > ODD_EVEN_SIGMA:
        log.info("Odd-even %.1fσ > %.1fσ -> ECLIPSING_BINARY",
                 result.odd_even_sigma, ODD_EVEN_SIGMA)
        return Classification.ECLIPSING_BINARY

    # 6. Depth too shallow -> NOISE
    if result.depth is not None and result.depth < DEPTH_MIN_SIGNAL:
        log.info("Depth %.5f < %.5f -> NOISE", result.depth, DEPTH_MIN_SIGNAL)
        return Classification.NOISE

    # 7. Multi-sector validation
    if result.sectors_checked >= 2:
        if result.sectors_detected >= 2:
            log.info("Detected in %d/%d sectors -> PLANET_CANDIDATE",
                     result.sectors_detected, result.sectors_checked)
            return Classification.PLANET_CANDIDATE
        elif result.sectors_detected == 0:
            log.info("Not detected in other sectors -> FALSE_POSITIVE")
            return Classification.FALSE_POSITIVE
        else:
            # 1 sector only
            log.info("Only 1 sector detection -> MANUAL_REVIEW")
            return Classification.MANUAL_REVIEW
    else:
        # Single sector available
        log.info("Only 1 sector available -> MANUAL_REVIEW")
        return Classification.MANUAL_REVIEW
