from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

# File này nằm tại:
# EventPick/src/cluster_refinement/config.py
# parents[2] = thư mục gốc EventPick
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. TEST CLUSTER CONFIG
# ============================================================

# TEST_EVENT_VERSION = "Event_LopTai"
# TEST_CLUSTER_NAME = "person_20_no"
#
# TEST_CLUSTER_DIR = (
#     PROJECT_ROOT
#     / "data"
#     / TEST_EVENT_VERSION
#     / TEST_CLUSTER_NAME
# )
# ============================================================
# 2. TEST CLUSTER CONFIG
# ============================================================

# Đây là tên dùng để tổ chức folder output report/log,
# không nhất thiết phải là tên event thật.
TEST_EVENT_VERSION = "_manual_test"

# Tên cluster test
TEST_CLUSTER_NAME = "test"

# Folder input thực tế đang chứa ảnh lẫn 4 người
TEST_CLUSTER_DIR = (
    PROJECT_ROOT
    / "data"
    / "test"
)


# ============================================================
# 3. OUTPUT: EMBEDDINGS
# ============================================================

EMBEDDING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "cluster_refinement_test"
    / TEST_EVENT_VERSION
    / TEST_CLUSTER_NAME
)

FACENET_EMBEDDINGS_NPY_PATH = (
    EMBEDDING_OUTPUT_DIR
    / f"{TEST_CLUSTER_NAME}_facenet_embeddings.npy"
)

FACENET_EMBEDDING_METADATA_CSV_PATH = (
    EMBEDDING_OUTPUT_DIR
    / f"{TEST_CLUSTER_NAME}_facenet_embedding_metadata.csv"
)


# ============================================================
# 4. OUTPUT: REPORTS
# ============================================================

REPORT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cluster_refinement_test"
    / TEST_EVENT_VERSION
    / TEST_CLUSTER_NAME
)

FACENET_EMBEDDING_SUMMARY_JSON_PATH = (
    REPORT_OUTPUT_DIR
    / f"{TEST_CLUSTER_NAME}_facenet_embedding_summary.json"
)


# ============================================================
# 5. OUTPUT: LOGS
# ============================================================

LOG_OUTPUT_DIR = (
    PROJECT_ROOT
    / "logs"
    / "cluster_refinement_test"
    / TEST_EVENT_VERSION
    / TEST_CLUSTER_NAME
)

FACENET_EMBEDDING_LOG_PATH = (
    LOG_OUTPUT_DIR
    / "facenet_embedding_extraction.log"
)


# ============================================================
# 6. FACENET CONFIG
# ============================================================

FACENET_PRETRAINED = "vggface2"

FACENET_IMAGE_SIZE = (160, 160)

FACENET_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
FACENET_NORMALIZE_STD = [0.5, 0.5, 0.5]


# ============================================================
# 7. IMAGE EXTENSIONS
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
    ".BMP",
}
# ============================================================
# 8. FINAL SPLIT CONFIG FOR CURRENT TEST CLUSTER
# ============================================================

FINAL_SPLIT_METHOD = "agglomerative"
FINAL_SPLIT_LINKAGE = "average"
FINAL_SPLIT_DISTANCE_THRESHOLD = 0.24

# Ký hiệu threshold trong tên folder/file đã sinh ở bước sweep
FINAL_SPLIT_THRESHOLD_FOLDER_NAME = "0p24"

# File assignments đã được sinh từ sweep:
# average_thr_0p28/person_20_no_average_thr_0p28_assignments.csv
FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH = (
    REPORT_OUTPUT_DIR
    / "subclustering_agglomerative_sweep"
    / f"{FINAL_SPLIT_LINKAGE}_thr_{FINAL_SPLIT_THRESHOLD_FOLDER_NAME}"
    / f"{TEST_CLUSTER_NAME}_{FINAL_SPLIT_LINKAGE}_thr_{FINAL_SPLIT_THRESHOLD_FOLDER_NAME}_assignments.csv"
)

# Output refined cluster thật
FINAL_SPLIT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "refined_clusters"
    / TEST_EVENT_VERSION
    / f"{TEST_CLUSTER_NAME}_refined"
)

# Report final split
FINAL_SPLIT_REPORT_DIR = (
    REPORT_OUTPUT_DIR
    / "final_split"
)

FINAL_SPLIT_ASSIGNMENTS_CSV_PATH = (
    FINAL_SPLIT_REPORT_DIR
    / f"{TEST_CLUSTER_NAME}_final_split_assignments.csv"
)

FINAL_SPLIT_SUMMARY_JSON_PATH = (
    FINAL_SPLIT_REPORT_DIR
    / f"{TEST_CLUSTER_NAME}_final_split_summary.json"
)

FINAL_SPLIT_LOG_PATH = (
    FINAL_SPLIT_REPORT_DIR
    / "final_split_creation.log"
)

# An toàn dữ liệu:
# False = nếu folder refined đã tồn tại thì dừng, tránh trộn dữ liệu cũ/mới
# True  = xóa folder refined cũ rồi tạo lại
OVERWRITE_FINAL_SPLIT_OUTPUT = False