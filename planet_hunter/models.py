from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Classification(str, Enum):
    PLANET_CANDIDATE = "PLANET_CANDIDATE"
    KNOWN_PLANET = "KNOWN_PLANET"           # Known TOI / confirmed exoplanet
    ECLIPSING_BINARY = "ECLIPSING_BINARY"
    VARIABLE_STAR = "VARIABLE_STAR"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    NOISE = "NOISE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"


class QueueSource(str, Enum):
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    ML_TRAINING = "ML_TRAINING"


class QueueStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class StarInfo:
    tic_id: int
    ra: Optional[float] = None
    dec: Optional[float] = None
    tmag: Optional[float] = None
    radius: Optional[float] = None  # solar radii
    teff: Optional[float] = None    # Kelvin


@dataclass
class AnalysisResult:
    tic_id: int
    classification: Classification = Classification.PENDING
    period: Optional[float] = None          # days
    depth: Optional[float] = None           # fractional
    snr: Optional[float] = None
    duration: Optional[float] = None        # hours
    planet_radius: Optional[float] = None   # earth radii
    equilibrium_temp: Optional[float] = None  # Kelvin
    sectors_checked: int = 0
    sectors_detected: int = 0
    sinusoid_better: bool = False
    secondary_depth: Optional[float] = None
    odd_even_sigma: Optional[float] = None
    plot_lightcurve: Optional[str] = None
    plot_periodogram: Optional[str] = None
    plot_phase_fold: Optional[str] = None
    plot_diagnostic: Optional[str] = None
    review_notes: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class QueueItem:
    tic_id: int
    source: QueueSource = QueueSource.MANUAL
    priority: int = 1
    status: QueueStatus = QueueStatus.QUEUED
