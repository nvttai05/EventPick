from pathlib import Path
import math

import cv2
import pandas as pd
from tqdm import tqdm

from src.face_detection.config import (
    PROJECT_ROOT,
    FACES_METADATA_PATH,
    DETECTED_FACES_DIR,
    FACE_DETECTION_REPORT_DIR,
    CROP_MARGIN_RATIO,
    OVERWRITE_DETECTED_FACES,
)


# ============================================================
# 1. REPORT PATH
# ============================================================

FACE_CROP_FAILED_PATH = FACE_DETECTION_REPORT_DIR / "face_crop_failed.csv"


# ============================================================
# 2. PATH UTILITIES
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    """
    Chuyển relative path trong metadata thành absolute path.
    """
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def to_project_relative_path(absolute_path: Path) -> str:
    """
    Chuyển absolute path về relative path để lưu trong metadata.
    """
    return str(absolute_path.relative_to(PROJECT_ROOT))


# ============================================================
# 3. CROP UTILITIES
# ============================================================

def expand_bbox_with_margin(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """
    Nới bounding box thêm margin_ratio quanh face.

    Ví dụ margin_ratio = 0.15:
    - nới thêm 15% chiều rộng bbox sang trái/phải
    - nới thêm 15% chiều cao bbox lên/xuống
    """

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    margin_x = bbox_width * margin_ratio
    margin_y = bbox_height * margin_ratio

    crop_x1 = math.floor(x1 - margin_x)
    crop_y1 = math.floor(y1 - margin_y)
    crop_x2 = math.ceil(x2 + margin_x)
    crop_y2 = math.ceil(y2 + margin_y)

    # Clamp về giới hạn ảnh
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(image_width, crop_x2)
    crop_y2 = min(image_height, crop_y2)

    return crop_x1, crop_y1, crop_x2, crop_y2


def build_detected_face_output_path(
    event_id: str,
    image_id: str,
    face_index: int,
) -> Path:
    """
    Tạo path output cho face crop.

    Ví dụ:
    data/detected_faces/Event_PSH/abc123_face01.jpg
    """
    image_stem = Path(image_id).stem
    face_index_str = f"{int(face_index):02d}"

    output_filename = f"{image_stem}_face{face_index_str}.jpg"

    output_dir = DETECTED_FACES_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / output_filename


# ============================================================
# 4. MAIN CROP PIPELINE
# ============================================================

def run_face_crop() -> None:
    """
    Crop toàn bộ face từ bounding box đã có trong faces_metadata.csv.

    Output:
    - data/detected_faces/{event_id}/*.jpg
    - Cập nhật detected_face_path trong faces_metadata.csv
    - Bổ sung crop_status, crop_error
    - Tạo face_crop_failed.csv
    """

    if not FACES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy faces_metadata.csv tại: {FACES_METADATA_PATH}"
        )

    faces_df = pd.read_csv(
        FACES_METADATA_PATH,
        keep_default_na=False
    )

    required_columns = {
        "face_id",
        "image_id",
        "event_id",
        "face_index",
        "source_image_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    }

    missing_columns = required_columns - set(faces_df.columns)
    if missing_columns:
        raise ValueError(
            f"faces_metadata.csv thiếu các cột bắt buộc: {missing_columns}"
        )

    # Bổ sung cột nếu chưa có
    if "detected_face_path" not in faces_df.columns:
        faces_df["detected_face_path"] = ""

    if "crop_status" not in faces_df.columns:
        faces_df["crop_status"] = "pending"

    if "crop_error" not in faces_df.columns:
        faces_df["crop_error"] = ""

    if "crop_margin_ratio" not in faces_df.columns:
        faces_df["crop_margin_ratio"] = CROP_MARGIN_RATIO

    # Ép các cột text về kiểu object để có thể gán chuỗi an toàn
    faces_df["detected_face_path"] = faces_df["detected_face_path"].astype("object")
    faces_df["crop_status"] = faces_df["crop_status"].astype("object")
    faces_df["crop_error"] = faces_df["crop_error"].astype("object")

    failed_crop_records: list[dict] = []

    # Group theo ảnh gốc để tránh đọc lại cùng một ảnh hàng chục lần
    grouped_faces = faces_df.groupby(
        ["image_id", "event_id", "source_image_path"],
        sort=False,
    )

    print("=" * 70)
    print("BẮT ĐẦU FACE CROP")
    print("=" * 70)
    print(f"Tổng face cần xử lý: {len(faces_df)}")
    print(f"Tổng ảnh nguồn có face: {len(grouped_faces)}")
    print(f"Crop margin ratio: {CROP_MARGIN_RATIO}")
    print("=" * 70)
    print()

    for (image_id, event_id, source_image_path), group in tqdm(
        grouped_faces,
        total=len(grouped_faces),
        desc="Cropping faces by source image",
    ):
        absolute_source_path = resolve_project_path(source_image_path)

        image = cv2.imread(str(absolute_source_path))

        if image is None:
            # Nếu ảnh gốc không đọc được, đánh lỗi toàn bộ face trong ảnh đó
            for row_idx, row in group.iterrows():
                error_message = "OpenCV không đọc được ảnh nguồn."

                faces_df.at[row_idx, "crop_status"] = "error"
                faces_df.at[row_idx, "crop_error"] = error_message

                failed_crop_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

            continue

        image_height, image_width = image.shape[:2]

        for row_idx, row in group.iterrows():
            try:
                face_id = str(row["face_id"])
                face_index = int(row["face_index"])

                x1 = float(row["bbox_x1"])
                y1 = float(row["bbox_y1"])
                x2 = float(row["bbox_x2"])
                y2 = float(row["bbox_y2"])

                crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox_with_margin(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_width=image_width,
                    image_height=image_height,
                    margin_ratio=CROP_MARGIN_RATIO,
                )

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    raise ValueError(
                        f"Crop bbox không hợp lệ: "
                        f"({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})"
                    )

                face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_crop.size == 0:
                    raise ValueError("Face crop rỗng.")

                output_path = build_detected_face_output_path(
                    event_id=str(event_id),
                    image_id=str(image_id),
                    face_index=face_index,
                )

                # Nếu file đã có và không muốn ghi đè thì chỉ cập nhật metadata
                if output_path.exists() and not OVERWRITE_DETECTED_FACES:
                    detected_face_relative_path = to_project_relative_path(output_path)

                    faces_df.at[row_idx, "detected_face_path"] = detected_face_relative_path
                    faces_df.at[row_idx, "crop_status"] = "exists_skipped"
                    faces_df.at[row_idx, "crop_error"] = ""
                    faces_df.at[row_idx, "crop_margin_ratio"] = CROP_MARGIN_RATIO
                    continue

                success = cv2.imwrite(str(output_path), face_crop)

                if not success:
                    raise IOError(f"Không thể ghi ảnh crop ra file: {output_path}")

                detected_face_relative_path = to_project_relative_path(output_path)

                faces_df.at[row_idx, "detected_face_path"] = detected_face_relative_path
                faces_df.at[row_idx, "crop_status"] = "success"
                faces_df.at[row_idx, "crop_error"] = ""
                faces_df.at[row_idx, "crop_margin_ratio"] = CROP_MARGIN_RATIO

            except Exception as error:
                error_message = str(error)

                faces_df.at[row_idx, "crop_status"] = "error"
                faces_df.at[row_idx, "crop_error"] = error_message

                failed_crop_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

    # Lưu lại face metadata đã cập nhật crop path
    faces_df.to_csv(
        FACES_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu report lỗi
    failed_crop_df = pd.DataFrame(
        failed_crop_records,
        columns=[
            "face_id",
            "image_id",
            "event_id",
            "source_image_path",
            "error",
        ],
    )

    failed_crop_df.to_csv(
        FACE_CROP_FAILED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Thống kê
    crop_status_counts = faces_df["crop_status"].value_counts(dropna=False)

    print()
    print("=" * 70)
    print("HOÀN TẤT FACE CROP")
    print("=" * 70)
    print(f"Tổng face trong metadata: {len(faces_df)}")
    print()
    print("Phân bố crop_status:")
    print(crop_status_counts)
    print()
    print(f"Số face crop lỗi: {len(failed_crop_df)}")
    print(f"Đã cập nhật: {FACES_METADATA_PATH}")
    print(f"Đã tạo report lỗi: {FACE_CROP_FAILED_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_face_crop()