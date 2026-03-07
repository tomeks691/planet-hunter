from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "planet_hunter.db"
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

# Web
HOST = "0.0.0.0"
PORT = 8420
