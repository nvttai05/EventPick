from pathlib import Path
import math

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.crop_face_val.config import (
    PROJECT_ROOT,
    RAW_EVENTS_DIR,
    FACES_METADATA_PATH,
    VAL_PROCESSED_DIR,
    VAL_PROCESSED_METADATA_PATH,
    OVERWRITE_VAL_PROCESSED,
)


# ============================================================
# 1. CONFIG
# ============================================================

OUTPUT_SIZE = (224, 224)

# True = mỗi ảnh gốc chỉ lấy 1 face tốt nhất.
# Khuyến nghị cho tập val vì mỗi folder n000xxx là 1 người.
# Nếu lấy tất cả face, có thể đưa nhầm người nền vào class đó.
USE_BEST_FACE_PER_IMAGE = True

# Ưu tiên face lớn và confidence cao.
# score = confidence * sqrt(face_area)
BEST_FACE_SCORE_MODE = "confidence_area"


# Template 5 landmarks chuẩn cho ảnh 224x224
REFERENCE_LANDMARKS_224 = np.array(
    [
        [76.5892, 103.3926],   # left eye
        [147.0636, 103.0028],  # right eye
        [112.0504, 143.4732],  # nose
        [83.0986, 184.7310],   # mouth left
        [141.4598, 184.4082],  # mouth right
    ],
    dtype=np.float32,
)


# ============================================================
# 2. PATH UTILS
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def to_project_relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def build_output_path(source_image_path: str, image_id: str, face_index: int) -> Path:
    """
    Giữ lại cấu trúc folder gốc.

    Input:
        data/val/Event_Public/n000002/abc.jpg

    Output:
        data/val_processed/Event_Public/n000002/abc_face01.jpg
    """
    absolute_source_path = resolve_project_path(source_image_path)

    try:
        relative_parent = absolute_source_path.parent.relative_to(RAW_EVENTS_DIR)
    except ValueError:
        relative_parent = Path(absolute_source_path.parent.name)

    image_stem = Path(str(image_id)).stem
    face_index_str = f"{int(face_index):02d}"

    output_filename = f"{image_stem}_face{face_index_str}.jpg"

    output_dir = VAL_PROCESSED_DIR / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / output_filename


# ============================================================
# 3. FACE SELECTION
# ============================================================

def compute_face_area(row: pd.Series) -> float:
    x1 = float(row["bbox_x1"])
    y1 = float(row["bbox_y1"])
    x2 = float(row["bbox_x2"])
    y2 = float(row["bbox_y2"])

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def compute_best_face_score(row: pd.Series) -> float:
    confidence = float(row["confidence"])
    area = compute_face_area(row)

    return confidence * math.sqrt(area)


def select_faces_to_process(faces_df: pd.DataFrame) -> pd.DataFrame:
    """
    Nếu USE_BEST_FACE_PER_IMAGE=True:
        mỗi ảnh gốc chỉ giữ face tốt nhất.
    Nếu False:
        xử lý toàn bộ face.
    """
    if not USE_BEST_FACE_PER_IMAGE:
        return faces_df.copy()

    df = faces_df.copy()
    df["face_area"] = df.apply(compute_face_area, axis=1)
    df["best_face_score"] = df.apply(compute_best_face_score, axis=1)

    selected_rows = []

    grouped = df.groupby(
        ["image_id", "event_id", "source_image_path"],
        sort=False,
    )

    for _, group in grouped:
        best_idx = group["best_face_score"].idxmax()
        selected_rows.append(df.loc[best_idx])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    return selected_df


# ============================================================
# 4. ALIGNMENT
# ============================================================

def extract_landmarks(row: pd.Series) -> np.ndarray:
    landmarks = np.array(
        [
            [float(row["left_eye_x"]), float(row["left_eye_y"])],
            [float(row["right_eye_x"]), float(row["right_eye_y"])],
            [float(row["nose_x"]), float(row["nose_y"])],
            [float(row["mouth_left_x"]), float(row["mouth_left_y"])],
            [float(row["mouth_right_x"]), float(row["mouth_right_y"])],
        ],
        dtype=np.float32,
    )

    return landmarks


def align_face_from_source(
    source_image: np.ndarray,
    source_landmarks: np.ndarray,
) -> np.ndarray:
    if source_landmarks.shape != (5, 2):
        raise ValueError(f"Landmarks sai shape: {source_landmarks.shape}")

    if not np.isfinite(source_landmarks).all():
        raise ValueError("Landmarks chứa NaN hoặc Inf.")

    transform_matrix, _ = cv2.estimateAffinePartial2D(
        source_landmarks,
        REFERENCE_LANDMARKS_224,
        method=cv2.LMEDS,
    )

    if transform_matrix is None:
        raise ValueError("Không ước lượng được affine transform.")

    output_width, output_height = OUTPUT_SIZE

    aligned_face = cv2.warpAffine(
        source_image,
        transform_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if aligned_face is None or aligned_face.size == 0:
        raise ValueError("Ảnh aligned rỗng.")

    return aligned_face


# ============================================================
# 5. MAIN
# ============================================================

def build_val_aligned_from_metadata() -> None:
    if not FACES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy faces_metadata.csv: {FACES_METADATA_PATH}"
        )

    VAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VAL_PROCESSED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    faces_df = pd.read_csv(
        FACES_METADATA_PATH,
        keep_default_na=False,
    )

    required_columns = {
        "face_id",
        "image_id",
        "event_id",
        "face_index",
        "source_image_path",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "left_eye_x",
        "left_eye_y",
        "right_eye_x",
        "right_eye_y",
        "nose_x",
        "nose_y",
        "mouth_left_x",
        "mouth_left_y",
        "mouth_right_x",
        "mouth_right_y",
    }

    missing_columns = required_columns - set(faces_df.columns)

    if missing_columns:
        raise ValueError(
            f"faces_metadata.csv thiếu các cột bắt buộc: {missing_columns}"
        )

    # Chỉ lấy face detect thành công nếu có cột detection_status
    if "detection_status" in faces_df.columns:
        faces_df = faces_df[
            faces_df["detection_status"].isin(["accepted", "suspicious"])
        ].copy()

    selected_df = select_faces_to_process(faces_df)

    print("=" * 80)
    print("BẮT ĐẦU BUILD VAL ALIGNED")
    print("=" * 80)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_EVENTS_DIR: {RAW_EVENTS_DIR}")
    print(f"FACES_METADATA_PATH: {FACES_METADATA_PATH}")
    print(f"VAL_PROCESSED_DIR: {VAL_PROCESSED_DIR}")
    print(f"Tổng face trong metadata: {len(faces_df)}")
    print(f"Số face sẽ xử lý: {len(selected_df)}")
    print(f"USE_BEST_FACE_PER_IMAGE: {USE_BEST_FACE_PER_IMAGE}")
    print(f"OUTPUT_SIZE: {OUTPUT_SIZE}")
    print("=" * 80)

    output_records = []
    failed_records = []

    grouped = selected_df.groupby(
        ["image_id", "event_id", "source_image_path"],
        sort=False,
    )

    for (image_id, event_id, source_image_path), group in tqdm(
        grouped,
        total=len(grouped),
        desc="Building aligned validation faces",
    ):
        source_path = resolve_project_path(source_image_path)
        source_image = cv2.imread(str(source_path))

        if source_image is None:
            for _, row in group.iterrows():
                error_message = "OpenCV không đọc được ảnh nguồn."

                failed_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

            continue

        for _, row in group.iterrows():
            try:
                face_index = int(row["face_index"])

                output_path = build_output_path(
                    source_image_path=str(source_image_path),
                    image_id=str(image_id),
                    face_index=face_index,
                )

                if output_path.exists() and not OVERWRITE_VAL_PROCESSED:
                    status = "exists_skipped"
                else:
                    landmarks = extract_landmarks(row)

                    aligned_face = align_face_from_source(
                        source_image=source_image,
                        source_landmarks=landmarks,
                    )

                    success = cv2.imwrite(str(output_path), aligned_face)

                    if not success:
                        raise IOError(f"Không thể ghi ảnh: {output_path}")

                    status = "success"

                output_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "confidence": row["confidence"],
                    "face_index": face_index,
                    "output_path": to_project_relative_path(output_path),
                    "output_width": OUTPUT_SIZE[0],
                    "output_height": OUTPUT_SIZE[1],
                    "process_status": status,
                    "process_error": "",
                })

            except Exception as error:
                error_message = str(error)

                failed_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

                output_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "confidence": row["confidence"],
                    "face_index": row["face_index"],
                    "output_path": "",
                    "output_width": OUTPUT_SIZE[0],
                    "output_height": OUTPUT_SIZE[1],
                    "process_status": "error",
                    "process_error": error_message,
                })

    output_df = pd.DataFrame(output_records)

    output_df.to_csv(
        VAL_PROCESSED_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    status_counts = output_df["process_status"].value_counts(dropna=False)

    print()
    print("=" * 80)
    print("HOÀN TẤT BUILD VAL ALIGNED")
    print("=" * 80)
    print(f"Tổng ảnh output records: {len(output_df)}")
    print()
    print("Phân bố process_status:")
    print(status_counts)
    print()
    print(f"Số lỗi: {len(failed_records)}")
    print(f"Đã lưu metadata: {VAL_PROCESSED_METADATA_PATH}")
    print(f"Output folder: {VAL_PROCESSED_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    build_val_aligned_from_metadata()