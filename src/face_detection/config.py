from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Nếu file config.py nằm trong:
# D:/Hoctap/CK_KHDL/EventPick_v1/src/face_detection/config.py
# thì parents[2] = D:/Hoctap/CK_KHDL/EventPick_v1
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

# Folder chứa các class/person folder n000002, n000003, ...
RAW_EVENTS_DIR = DATA_DIR / "val" / "Event_Public"

# Metadata riêng cho pipeline detect/crop
METADATA_DIR = DATA_DIR / "metadata" / "Event_Public"
IMAGES_METADATA_PATH = METADATA_DIR / "images_metadata.csv"
FACES_METADATA_PATH = METADATA_DIR / "faces_metadata.csv"

# Output face crop / aligned face
DETECTED_FACES_DIR = DATA_DIR / "detected_faces" / "Event_Public"
ALIGNED_FACES_DIR = DATA_DIR / "aligned_faces" / "Event_Public"

REPORTS_DIR = PROJECT_ROOT / "reports"
FACE_DETECTION_REPORT_DIR = REPORTS_DIR / "face_detection" / "Event_Public"

VISUALIZATION_DIR = REPORTS_DIR / "visualizations" / "face_detection" / "Event_Public"
ANNOTATED_SOURCES_DIR = VISUALIZATION_DIR / "annotated_sources"
CORRECT_EXAMPLES_DIR = VISUALIZATION_DIR / "correct_examples"
SUSPICIOUS_EXAMPLES_DIR = VISUALIZATION_DIR / "suspicious_examples"
CROWDED_SCENES_DIR = VISUALIZATION_DIR / "crowded_scenes"
ALIGNMENT_COMPARISON_DIR = VISUALIZATION_DIR / "alignment_comparison"

DETECTION_FAILED_IMAGES_PATH = (
    FACE_DETECTION_REPORT_DIR / "failed_images.csv"
)


# ============================================================
# 2. RETINAFACE / INSIGHTFACE CONFIG
# ============================================================

INSIGHTFACE_MODEL_NAME = "buffalo_l"

ALLOWED_MODULES = ["detection"]

CTX_ID = 0

PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

DET_SIZE = (640, 640)


# ============================================================
# 3. DETECTION THRESHOLD
# ============================================================

DETECTION_THRESHOLD = 0.50
ACCEPTED_THRESHOLD = 0.80
SUSPICIOUS_THRESHOLD = 0.50


# ============================================================
# 4. FACE CROP CONFIG
# ============================================================

CROP_MARGIN_RATIO = 0.15

OVERWRITE_DETECTED_FACES = False


# ============================================================
# 5. ALIGNMENT OUTPUT CONFIG
# ============================================================

ALIGNED_FACE_SIZE = (224, 224)

OVERWRITE_ALIGNED_FACES = False


# ============================================================
# 6. VISUALIZATION CONFIG
# ============================================================

MAX_ANNOTATED_SOURCE_IMAGES = 20
MAX_CORRECT_EXAMPLES = 20
MAX_SUSPICIOUS_EXAMPLES = 20
MAX_CROWDED_SCENE_EXAMPLES = 12
MAX_ALIGNMENT_COMPARISONS = 30