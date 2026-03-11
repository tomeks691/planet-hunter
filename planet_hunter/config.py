import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "planet_hunter.db"
ML_DB_PATH = DATA_DIR / "ml_training.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
ML_DB_URL = os.getenv("ML_DB_URL", f"sqlite:///{ML_DB_PATH}")
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Pipeline thresholds
SNR_MINIMUM = 7.0
DEPTH_MAX_PLANET = 0.05        # 5% - above this -> EB
DEPTH_MIN_SIGNAL = 0.0003      # 0.03% - below this -> NOISE
SECONDARY_ECLIPSE_RATIO = 0.30 # secondary > 30% of primary -> EB
ODD_EVEN_SIGMA = 3.0           # odd-even difference threshold

# BLS search range
PERIOD_MIN = 0.5   # days
PERIOD_MAX = 30.0  # days

# Queue priorities
PRIORITY_MANUAL = 1
PRIORITY_AUTO = 5

# Scanner
SCANNER_INTERVAL = 300  # seconds between auto-scans
SCANNER_BATCH_SIZE = 5
# If True, AUTO scanner will pause while ML training backlog exists (queued/running)
PAUSE_AUTO_SCANNER_WHEN_ML_BACKLOG = True
# Ignore ML RUNNING items older than this age when deciding backlog pause.
ML_BACKLOG_RUNNING_MAX_AGE_MINUTES = 45

# Queue recovery / stale RUNNING safeguards
# Aggressive recovery for ML queue to avoid AUTO scanner lock-ups.
STUCK_RUNNING_ML_MINUTES = 45
# Conservative recovery for non-ML queue items.
STUCK_RUNNING_OTHER_MINUTES = 360
STUCK_SWEEP_INTERVAL_SECONDS = 300
# Hard timeout for a single TIC processing attempt (prevents one stuck target
# from blocking the only worker forever).
PIPELINE_ITEM_TIMEOUT_SECONDS = 900

# ML inference (optional runtime classifier)
ML_CLASSIFIER_ENABLED = True
ML_CLASSIFIER_UNCERTAINTY_MARGIN = 0.03
ML_MODEL_VERSION = "v1.0-best"
ML_STAGE_A_PATH = DATA_DIR / "ml/artifacts/two_stage_hr_stage_a.joblib"
ML_STAGE_B_PATH = DATA_DIR / "ml/artifacts/two_stage_hr_stage_b.joblib"
ML_METRICS_PATH = DATA_DIR / "ml/artifacts/two_stage_high_recall_metrics.json"

# Web
HOST = "0.0.0.0"
PORT = 8420
