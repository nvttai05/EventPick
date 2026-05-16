from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.face_detection.config import (
    PROJECT_ROOT,
    FACES_METADATA_PATH,
    ALIGNED_FACES_DIR,
    FACE_DETECTION_REPORT_DIR,
    ALIGNED_FACE_SIZE,
    OVERWRITE_ALIGNED_FACES,
)


# ============================================================
# 1. REPORT PATH
# ============================================================

FACE_ALIGNMENT_FAILED_PATH = (
    FACE_DETECTION_REPORT_DIR / "face_alignment_failed.csv"
)


# ============================================================
# 2. ALIGNMENT TEMPLATE
# ============================================================

# Template 5 landmarks chuẩn cho ảnh 224x224
# Scale từ template face alignment phổ biến 112x112 lên gấp đôi.
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
# 3. PATH UTILITIES
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


def build_aligned_face_output_path(
    event_id: str,
    image_id: str,
    face_index: int,
) -> Path:
    """
    Tạo path output cho ảnh aligned face.

    Ví dụ:
    data/aligned_faces/Event_PSH/abc123_face01.jpg
    """
    image_stem = Path(str(image_id)).stem
    face_index_str = f"{int(face_index):02d}"

    output_filename = f"{image_stem}_face{face_index_str}.jpg"

    output_dir = ALIGNED_FACES_DIR / str(event_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / output_filename


# ============================================================
# 4. LANDMARK / ALIGNMENT UTILITIES
# ============================================================

def extract_source_landmarks(row: pd.Series) -> np.ndarray:
    """
    Lấy 5 landmarks từ một dòng faces_metadata.csv.
    Shape output: (5, 2)
    """
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


def align_face_from_source_image(
    source_image: np.ndarray,
    source_landmarks: np.ndarray,
) -> np.ndarray:
    """
    Align face từ ảnh gốc bằng 5 landmarks.

    Dùng cv2.estimateAffinePartial2D để ước lượng phép biến đổi
    similarity-like transform từ landmarks gốc -> template landmarks chuẩn.
    """

    if source_landmarks.shape != (5, 2):
        raise ValueError(
            f"Landmarks không đúng shape (5, 2): {source_landmarks.shape}"
        )

    if not np.isfinite(source_landmarks).all():
        raise ValueError("Landmarks chứa NaN hoặc Inf.")

    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        source_landmarks,
        REFERENCE_LANDMARKS_224,
        method=cv2.LMEDS,
    )

    if transform_matrix is None:
        raise ValueError("Không ước lượng được affine transform.")

    output_width, output_height = ALIGNED_FACE_SIZE

    aligned_face = cv2.warpAffine(
        source_image,
        transform_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if aligned_face is None or aligned_face.size == 0:
        raise ValueError("Ảnh aligned face rỗng.")

    return aligned_face


# ============================================================
# 5. MAIN ALIGNMENT PIPELINE
# ============================================================

def run_face_alignment() -> None:
    """
    Align toàn bộ khuôn mặt từ 5 landmarks.

    Output:
    - data/aligned_faces/{event_id}/*.jpg
    - Cập nhật faces_metadata.csv
    - Tạo face_alignment_failed.csv
    """

    if not FACES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy faces_metadata.csv tại: {FACES_METADATA_PATH}"
        )

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

    # Bổ sung cột nếu chưa có
    if "aligned_face_path" not in faces_df.columns:
        faces_df["aligned_face_path"] = ""

    if "alignment_applied" not in faces_df.columns:
        faces_df["alignment_applied"] = False

    if "alignment_status" not in faces_df.columns:
        faces_df["alignment_status"] = "pending"

    if "alignment_error" not in faces_df.columns:
        faces_df["alignment_error"] = ""

    if "output_width" not in faces_df.columns:
        faces_df["output_width"] = ""

    if "output_height" not in faces_df.columns:
        faces_df["output_height"] = ""

    # Ép kiểu các cột text về object để gán chuỗi an toàn
    faces_df["aligned_face_path"] = faces_df["aligned_face_path"].astype("object")
    faces_df["alignment_status"] = faces_df["alignment_status"].astype("object")
    faces_df["alignment_error"] = faces_df["alignment_error"].astype("object")
    # Ép output_width/output_height về kiểu số nguyên nullable
    # để có thể gán 224 mà không lỗi dtype
    faces_df["output_width"] = pd.to_numeric(
        faces_df["output_width"].replace("", pd.NA),
        errors="coerce"
    ).astype("Int64")

    faces_df["output_height"] = pd.to_numeric(
        faces_df["output_height"].replace("", pd.NA),
        errors="coerce"
    ).astype("Int64")

    failed_alignment_records: list[dict] = []

    # Group theo ảnh nguồn để mỗi ảnh gốc chỉ đọc 1 lần
    grouped_faces = faces_df.groupby(
        ["image_id", "event_id", "source_image_path"],
        sort=False,
    )

    print("=" * 70)
    print("BẮT ĐẦU FACE ALIGNMENT")
    print("=" * 70)
    print(f"Tổng face cần align: {len(faces_df)}")
    print(f"Tổng ảnh nguồn có face: {len(grouped_faces)}")
    print(f"Output size: {ALIGNED_FACE_SIZE[0]}x{ALIGNED_FACE_SIZE[1]}")
    print("=" * 70)
    print()

    for (image_id, event_id, source_image_path), group in tqdm(
        grouped_faces,
        total=len(grouped_faces),
        desc="Aligning faces by source image",
    ):
        absolute_source_path = resolve_project_path(source_image_path)

        source_image = cv2.imread(str(absolute_source_path))

        if source_image is None:
            for row_idx, row in group.iterrows():
                error_message = "OpenCV không đọc được ảnh nguồn."

                faces_df.at[row_idx, "alignment_applied"] = False
                faces_df.at[row_idx, "alignment_status"] = "error"
                faces_df.at[row_idx, "alignment_error"] = error_message

                failed_alignment_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

            continue

        for row_idx, row in group.iterrows():
            try:
                face_id = str(row["face_id"])
                face_index = int(row["face_index"])

                output_path = build_aligned_face_output_path(
                    event_id=str(event_id),
                    image_id=str(image_id),
                    face_index=face_index,
                )

                # Nếu file đã có và không muốn ghi đè thì chỉ cập nhật metadata
                if output_path.exists() and not OVERWRITE_ALIGNED_FACES:
                    aligned_relative_path = to_project_relative_path(output_path)

                    faces_df.at[row_idx, "aligned_face_path"] = aligned_relative_path
                    faces_df.at[row_idx, "alignment_applied"] = True
                    faces_df.at[row_idx, "alignment_status"] = "exists_skipped"
                    faces_df.at[row_idx, "alignment_error"] = ""
                    faces_df.at[row_idx, "output_width"] = ALIGNED_FACE_SIZE[0]
                    faces_df.at[row_idx, "output_height"] = ALIGNED_FACE_SIZE[1]
                    continue

                source_landmarks = extract_source_landmarks(row)

                aligned_face = align_face_from_source_image(
                    source_image=source_image,
                    source_landmarks=source_landmarks,
                )

                success = cv2.imwrite(str(output_path), aligned_face)

                if not success:
                    raise IOError(f"Không thể ghi ảnh aligned face: {output_path}")

                aligned_relative_path = to_project_relative_path(output_path)

                faces_df.at[row_idx, "aligned_face_path"] = aligned_relative_path
                faces_df.at[row_idx, "alignment_applied"] = True
                faces_df.at[row_idx, "alignment_status"] = "success"
                faces_df.at[row_idx, "alignment_error"] = ""
                faces_df.at[row_idx, "output_width"] = ALIGNED_FACE_SIZE[0]
                faces_df.at[row_idx, "output_height"] = ALIGNED_FACE_SIZE[1]

            except Exception as error:
                error_message = str(error)

                faces_df.at[row_idx, "alignment_applied"] = False
                faces_df.at[row_idx, "alignment_status"] = "error"
                faces_df.at[row_idx, "alignment_error"] = error_message

                failed_alignment_records.append({
                    "face_id": row["face_id"],
                    "image_id": image_id,
                    "event_id": event_id,
                    "source_image_path": source_image_path,
                    "error": error_message,
                })

    # Lưu metadata
    faces_df.to_csv(
        FACES_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu report lỗi
    failed_alignment_df = pd.DataFrame(
        failed_alignment_records,
        columns=[
            "face_id",
            "image_id",
            "event_id",
            "source_image_path",
            "error",
        ],
    )

    failed_alignment_df.to_csv(
        FACE_ALIGNMENT_FAILED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Thống kê cuối
    alignment_status_counts = faces_df["alignment_status"].value_counts(dropna=False)

    print()
    print("=" * 70)
    print("HOÀN TẤT FACE ALIGNMENT")
    print("=" * 70)
    print(f"Tổng face trong metadata: {len(faces_df)}")
    print()
    print("Phân bố alignment_status:")
    print(alignment_status_counts)
    print()
    print(f"Số face align lỗi: {len(failed_alignment_df)}")
    print(f"Đã cập nhật: {FACES_METADATA_PATH}")
    print(f"Đã tạo report lỗi: {FACE_ALIGNMENT_FAILED_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_face_alignment()