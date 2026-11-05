import logging
from pathlib import Path

# Base project directory (always resolves relative to this project, regardless of working directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data & Model Directories
DATASET_ROOT = BASE_DIR / "dataSet"
USER_IMAGES_DIR = DATASET_ROOT / "user"
RECOGNIZER_DIR = BASE_DIR / "recognizer"
EXPORT_DIR = BASE_DIR / "exports"

# File Paths
DB_PATH = DATASET_ROOT / "FaceBase.db"
MODEL_PATH = RECOGNIZER_DIR / "training_data.yml"
LOCAL_CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

# Face Processing & Recognition Defaults
FACE_SIZE = (200, 200)  # Standardized face crop dimensions for consistent LBPH histograms
DEFAULT_SAMPLES_PER_USER = 50
SAMPLE_INTERVAL_MS = 150  # Minimum interval between saved enrollment samples (ms)

# LBPH Distance Threshold:
# In OpenCV LBPH, `distance` (often called confidence) is a Chi-Square distance where 0 = exact match.
# Lower values mean higher similarity. Typical same-person matches are between 25 and 65.
DEFAULT_DISTANCE_THRESHOLD = 68.0

# Detection Parameters
DETECT_SCALE_FACTOR = 1.2
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE = (60, 60)

# Attendance / Audit Log Cooldown (seconds before logging the same person again)
LOG_COOLDOWN_SECONDS = 30


def ensure_directories() -> None:
    """Ensure all required application directories exist."""
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    USER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RECOGNIZER_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FaceRecognition")

