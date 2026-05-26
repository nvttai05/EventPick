from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd

from src.cluster_refinement.config import (
    PROJECT_ROOT,
    TEST_EVENT_VERSION,
    TEST_CLUSTER_NAME,
    FINAL_SPLIT_METHOD,
    FINAL_SPLIT_LINKAGE,
    FINAL_SPLIT_DISTANCE_THRESHOLD,
    FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH,
    FINAL_SPLIT_OUTPUT_DIR,
    FINAL_SPLIT_REPORT_DIR,
    FINAL_SPLIT_ASSIGNMENTS_CSV_PATH,
    FINAL_SPLIT_SUMMARY_JSON_PATH,
    FINAL_SPLIT_LOG_PATH,
    OVERWRITE_FINAL_SPLIT_OUTPUT,
)


# ============================================================
# 1. LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Tạo logger:
    - In ra terminal
    - Ghi vào final_split_creation.log
    """
    FINAL_SPLIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("create_final_split_one_cluster")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        FINAL_SPLIT_LOG_PATH,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# 2. PATH UTILITIES
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    """
    Chuyển path relative trong CSV thành absolute path.
    """
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def to_project_relative_path(path: Path) -> str:
    """
    Chuyển absolute path về relative path tính từ project root.
    """
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


# ============================================================
# 3. PREPARE OUTPUT DIR
# ============================================================

def prepare_output_directory(logger: logging.Logger) -> None:
    """
    Tạo hoặc làm mới thư mục refined output.
    """
    if FINAL_SPLIT_OUTPUT_DIR.exists():
        if OVERWRITE_FINAL_SPLIT_OUTPUT:
            logger.warning(
                "Folder output đã tồn tại và OVERWRITE=True. Đang xóa folder cũ: %s",
                FINAL_SPLIT_OUTPUT_DIR,
            )
            shutil.rmtree(FINAL_SPLIT_OUTPUT_DIR)
        else:
            raise FileExistsError(
                "Folder refined output đã tồn tại. "
                "Để tránh trộn dữ liệu cũ/mới, script dừng lại.\n"
                f"Folder: {FINAL_SPLIT_OUTPUT_DIR}\n"
                "Nếu muốn tạo lại từ đầu, đặt "
                "OVERWRITE_FINAL_SPLIT_OUTPUT = True trong config.py."
            )

    FINAL_SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. LABEL MAPPING
# ============================================================

def build_subperson_mapping(assignments_df: pd.DataFrame) -> dict[int, str]:
    """
    Map subcluster_label -> subperson_x.

    Quy ước:
    - Sắp subcluster theo số ảnh giảm dần
    - Cụm lớn nhất = subperson_0
    - Cụm lớn thứ hai = subperson_1
    - ...

    Cách này giúp output final gọn và dễ đọc,
    không bị phụ thuộc vào label gốc 0 / 1 / 2 của thuật toán.
    """
    label_counts = (
        assignments_df["subcluster_label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    mapping: dict[int, str] = {}

    for subperson_index, subcluster_label in enumerate(label_counts.index.tolist()):
        mapping[int(subcluster_label)] = f"subperson_{subperson_index}"

    return mapping


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def create_final_split_one_cluster() -> None:
    logger = setup_logging()

    logger.info("=" * 100)
    logger.info("BẮT ĐẦU TẠO FINAL SPLIT CHO CLUSTER %s", TEST_CLUSTER_NAME)
    logger.info("=" * 100)
    logger.info("Event version: %s", TEST_EVENT_VERSION)
    logger.info("Source cluster: %s", TEST_CLUSTER_NAME)
    logger.info(
        "Chosen config: method=%s | linkage=%s | distance_threshold=%.2f",
        FINAL_SPLIT_METHOD,
        FINAL_SPLIT_LINKAGE,
        FINAL_SPLIT_DISTANCE_THRESHOLD,
    )
    logger.info("Source assignments CSV: %s", FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH)
    logger.info("Final split output dir: %s", FINAL_SPLIT_OUTPUT_DIR)

    if not FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy assignments CSV đã chốt: "
            f"{FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH}"
        )

    assignments_df = pd.read_csv(
        FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH,
        keep_default_na=False,
    )

    required_columns = {
        "image_name",
        "image_path",
        "subcluster_label",
    }

    missing_columns = required_columns - set(assignments_df.columns)
    if missing_columns:
        raise ValueError(
            f"Assignments CSV thiếu các cột bắt buộc: {missing_columns}"
        )

    logger.info("Tổng ảnh trong assignments: %d", len(assignments_df))

    # Chuẩn bị folder output
    prepare_output_directory(logger)

    # Map label thuật toán -> subperson final
    subperson_mapping = build_subperson_mapping(assignments_df)

    logger.info("Subcluster -> Final subperson mapping:")
    for subcluster_label, subperson_name in subperson_mapping.items():
        logger.info("  subcluster %s -> %s", subcluster_label, subperson_name)

    final_records: list[dict] = []

    copy_success_count = 0
    copy_error_count = 0

    for _, row in assignments_df.iterrows():
        image_name = str(row["image_name"])
        image_path = resolve_project_path(row["image_path"])
        original_subcluster_label = int(row["subcluster_label"])

        final_subperson_name = subperson_mapping[original_subcluster_label]

        final_subperson_dir = FINAL_SPLIT_OUTPUT_DIR / final_subperson_name
        final_subperson_dir.mkdir(parents=True, exist_ok=True)

        output_image_path = final_subperson_dir / image_name

        record = {
            "event_version": TEST_EVENT_VERSION,
            "source_cluster_name": TEST_CLUSTER_NAME,
            "image_name": image_name,
            "source_image_path": to_project_relative_path(image_path),
            "original_subcluster_label": original_subcluster_label,
            "final_subperson_id": final_subperson_name,
            "final_output_path": "",
            "copy_status": "pending",
            "copy_error": "",
        }

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Không tìm thấy ảnh nguồn: {image_path}")

            shutil.copy2(image_path, output_image_path)

            record["final_output_path"] = to_project_relative_path(output_image_path)
            record["copy_status"] = "success"
            record["copy_error"] = ""

            copy_success_count += 1

        except Exception as error:
            record["copy_status"] = "error"
            record["copy_error"] = str(error)

            copy_error_count += 1

            logger.error(
                "Lỗi copy ảnh %s | %s",
                image_name,
                str(error),
            )

        final_records.append(record)

    # ============================================================
    # 6. SAVE FINAL ASSIGNMENTS CSV
    # ============================================================

    final_df = pd.DataFrame(final_records)

    FINAL_SPLIT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        FINAL_SPLIT_ASSIGNMENTS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 7. SUMMARY
    # ============================================================

    final_counts = (
        final_df[final_df["copy_status"] == "success"]["final_subperson_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    original_subcluster_counts = (
        assignments_df["subcluster_label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "event_version": TEST_EVENT_VERSION,
        "source_cluster_name": TEST_CLUSTER_NAME,
        "chosen_refinement_config": {
            "method": FINAL_SPLIT_METHOD,
            "linkage": FINAL_SPLIT_LINKAGE,
            "distance_threshold": FINAL_SPLIT_DISTANCE_THRESHOLD,
        },
        "input_assignments_csv": str(FINAL_SPLIT_SOURCE_ASSIGNMENTS_CSV_PATH),
        "final_split_output_dir": str(FINAL_SPLIT_OUTPUT_DIR),
        "num_images_in_assignments": int(len(assignments_df)),
        "num_images_copied_successfully": int(copy_success_count),
        "num_images_copy_error": int(copy_error_count),
        "num_final_subpersons": int(len(subperson_mapping)),
        "original_subcluster_counts": {
            str(k): int(v)
            for k, v in original_subcluster_counts.items()
        },
        "subcluster_to_final_subperson_mapping": {
            str(k): v
            for k, v in subperson_mapping.items()
        },
        "final_subperson_counts": {
            str(k): int(v)
            for k, v in final_counts.items()
        },
        "outputs": {
            "final_assignments_csv": str(FINAL_SPLIT_ASSIGNMENTS_CSV_PATH),
            "final_summary_json": str(FINAL_SPLIT_SUMMARY_JSON_PATH),
            "final_split_log": str(FINAL_SPLIT_LOG_PATH),
        },
    }

    with open(
            FINAL_SPLIT_SUMMARY_JSON_PATH,
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    # ============================================================
    # 8. FINAL LOG
    # ============================================================

    logger.info("=" * 100)
    logger.info("HOÀN TẤT FINAL SPLIT CHO %s", TEST_CLUSTER_NAME)
    logger.info("=" * 100)
    logger.info("Tổng ảnh assignments: %d", len(assignments_df))
    logger.info("Copy thành công: %d", copy_success_count)
    logger.info("Copy lỗi: %d", copy_error_count)
    logger.info("Số final subperson: %d", len(subperson_mapping))
    logger.info("Phân bố ảnh theo final subperson:")

    for subperson_name, count in final_counts.items():
        logger.info("  %s: %d ảnh", subperson_name, count)

    logger.info("Đã lưu output split thật: %s", FINAL_SPLIT_OUTPUT_DIR)
    logger.info("Đã lưu final assignments CSV: %s", FINAL_SPLIT_ASSIGNMENTS_CSV_PATH)
    logger.info("Đã lưu summary JSON: %s", FINAL_SPLIT_SUMMARY_JSON_PATH)
    logger.info("Đã lưu log: %s", FINAL_SPLIT_LOG_PATH)
    logger.info("=" * 100)


if __name__ == "__main__":
    create_final_split_one_cluster()