from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_EVENTS_DIR = DATA_DIR / "raw" / "events"

METADATA_DIR = DATA_DIR / "metadata"
IMAGES_METADATA_PATH = METADATA_DIR / "images_metadata.csv"
FACES_METADATA_PATH = METADATA_DIR / "faces_metadata.csv"

DETECTED_FACES_DIR = DATA_DIR / "detected_faces"
ALIGNED_FACES_DIR = DATA_DIR / "aligned_faces"

REPORTS_DIR = PROJECT_ROOT / "reports"
FACE_DETECTION_REPORT_DIR = REPORTS_DIR / "face_detection"

VISUALIZATION_DIR = REPORTS_DIR / "visualizations" / "face_detection"
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

# Model pack chính thức của InsightFace.
# buffalo_l dùng RetinaFace-10GF cho detection.
INSIGHTFACE_MODEL_NAME = "buffalo_l"

# Chỉ load detection, không load recognition, gender, age...
ALLOWED_MODULES = ["detection"]

# InsightFace dùng ctx_id = -1 cho CPU.
CTX_ID = 0

PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

# Kích thước input khi detector xử lý ảnh.
# 640x640 là lựa chọn cân bằng.
DET_SIZE = (640, 640)


# ============================================================
# 3. DETECTION THRESHOLD
# ============================================================

# Ngưỡng để detector trả ra face.
# Bên dưới 0.50 xem như bỏ qua.
DETECTION_THRESHOLD = 0.50

# Mặt confidence >= 0.80 được xem là accepted.
ACCEPTED_THRESHOLD = 0.80

# 0.50 <= confidence < 0.80 được xem là suspicious.
SUSPICIOUS_THRESHOLD = 0.50


# ============================================================
# 4. FACE CROP CONFIG
# ============================================================

# Crop thêm 15% margin quanh bbox để không cắt quá sát mặt.
CROP_MARGIN_RATIO = 0.15
# Nếu chạy lại crop:
# False = nếu ảnh crop đã tồn tại thì bỏ qua, không ghi đè
# True  = crop lại và ghi đè file cũ
OVERWRITE_DETECTED_FACES = False

# ============================================================
# 5. ALIGNMENT OUTPUT CONFIG
# ============================================================

ALIGNED_FACE_SIZE = (224, 224)
# Nếu chạy lại alignment:
# False = ảnh align đã có thì bỏ qua
# True  = align lại và ghi đè
OVERWRITE_ALIGNED_FACES = False
# ============================================================
# 6. VISUALIZATION CONFIG
# ============================================================

# Số lượng ví dụ lưu cho mỗi nhóm visualization.
MAX_ANNOTATED_SOURCE_IMAGES = 20
MAX_CORRECT_EXAMPLES = 20
MAX_SUSPICIOUS_EXAMPLES = 20
MAX_CROWDED_SCENE_EXAMPLES = 12
MAX_ALIGNMENT_COMPARISONS = 30