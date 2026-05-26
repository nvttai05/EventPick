from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.face_detection.config import (
    PROJECT_ROOT,
    IMAGES_METADATA_PATH,
    FACES_METADATA_PATH,
    ANNOTATED_SOURCES_DIR,
    CORRECT_EXAMPLES_DIR,
    SUSPICIOUS_EXAMPLES_DIR,
    CROWDED_SCENES_DIR,
    ALIGNMENT_COMPARISON_DIR,
    MAX_ANNOTATED_SOURCE_IMAGES,
    MAX_CORRECT_EXAMPLES,
    MAX_SUSPICIOUS_EXAMPLES,
    MAX_CROWDED_SCENE_EXAMPLES,
    MAX_ALIGNMENT_COMPARISONS,
)


# ============================================================
# 1. PATH UTILITIES
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    """
    Chuyển relative path trong metadata thành absolute path.
    """
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# 2. DRAWING UTILITIES
# ============================================================

def draw_detection_annotations(
    image: np.ndarray,
    faces_in_image: pd.DataFrame,
) -> np.ndarray:
    """
    Vẽ bbox + face index + confidence lên ảnh nguồn.
    """
    annotated = image.copy()

    for _, face in faces_in_image.iterrows():
        x1 = int(round(float(face["bbox_x1"])))
        y1 = int(round(float(face["bbox_y1"])))
        x2 = int(round(float(face["bbox_x2"])))
        y2 = int(round(float(face["bbox_y2"])))

        confidence = float(face["confidence"])
        face_index = int(face["face_index"])
        status = str(face["detection_status"])

        # Màu tự nhiên theo status
        # OpenCV dùng BGR
        if status == "accepted":
            box_color = (0, 255, 0)
        else:
            box_color = (0, 165, 255)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness=3,
        )

        label = f"F{face_index:02d} | {confidence:.2f}"

        label_x = x1
        label_y = max(25, y1 - 10)

        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def resize_for_report(
    image: np.ndarray,
    max_width: int = 1800,
) -> np.ndarray:
    """
    Resize ảnh lớn xuống để file visualization nhẹ hơn,
    nhưng vẫn đủ rõ để đưa vào báo cáo.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def save_image(path: Path, image: np.ndarray) -> None:
    """
    Ghi ảnh an toàn ra disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Không thể ghi ảnh: {path}")


# ============================================================
# 3. ANNOTATED SOURCE VISUALIZATION
# ============================================================

def create_annotated_source_examples(
    images_df: pd.DataFrame,
    faces_df: pd.DataFrame,
) -> None:
    """
    Tạo một số ảnh nguồn đại diện có vẽ bbox + confidence.
    Ưu tiên ảnh có số mặt trung bình-khá để trực quan đẹp.
    """
    candidate_images = images_df[
        (images_df["face_count"] >= 3)
        & (images_df["face_count"] <= 15)
    ].copy()

    if len(candidate_images) == 0:
        candidate_images = images_df[images_df["face_count"] > 0].copy()

    candidate_images = candidate_images.sample(
        n=min(MAX_ANNOTATED_SOURCE_IMAGES, len(candidate_images)),
        random_state=42,
    )

    print("Đang tạo annotated source examples...")

    for _, image_row in tqdm(
        candidate_images.iterrows(),
        total=len(candidate_images),
        desc="Annotated source images",
    ):
        image_id = str(image_row["image_id"])
        event_id = str(image_row["event_id"])
        source_path = resolve_project_path(image_row["path"])

        image = cv2.imread(str(source_path))

        if image is None:
            continue

        faces_in_image = faces_df[
            (faces_df["image_id"] == image_id)
            & (faces_df["event_id"] == event_id)
        ]

        annotated = draw_detection_annotations(image, faces_in_image)
        annotated = resize_for_report(annotated)

        output_name = f"{event_id}__{Path(image_id).stem}__annotated.jpg"
        output_path = ANNOTATED_SOURCES_DIR / output_name

        save_image(output_path, annotated)


# ============================================================
# 4. HIGH-CONFIDENCE DETECTION EXAMPLES
# ============================================================

def create_correct_candidate_examples(
    faces_df: pd.DataFrame,
) -> None:
    """
    Tạo ảnh crop riêng lẻ cho các detection confidence cao.
    Đây là ứng viên để người làm báo cáo chọn làm 'detect đúng'.
    """
    candidates = faces_df[
        (faces_df["detection_status"] == "accepted")
        & (faces_df["confidence"] >= 0.90)
        & (faces_df["detected_face_path"] != "")
    ].copy()

    candidates = candidates.sort_values(
        by="confidence",
        ascending=False,
    ).head(MAX_CORRECT_EXAMPLES)

    print("Đang tạo high-confidence detection examples...")

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Correct candidate examples",
    ):
        detected_path = resolve_project_path(row["detected_face_path"])

        image = cv2.imread(str(detected_path))

        if image is None:
            continue

        confidence = float(row["confidence"])
        face_id = str(row["face_id"])

        output_name = f"{face_id}__conf_{confidence:.3f}.jpg"
        output_path = CORRECT_EXAMPLES_DIR / output_name

        save_image(output_path, image)


# ============================================================
# 5. SUSPICIOUS DETECTION EXAMPLES
# ============================================================

def create_suspicious_candidate_examples(
    faces_df: pd.DataFrame,
) -> None:
    """
    Tạo crop cho các detection confidence thấp nhất.
    Đây là ứng viên để người làm báo cáo kiểm tra:
    - mặt quá nhỏ
    - detect mơ hồ
    - false positive tiềm năng
    """
    candidates = faces_df[
        (faces_df["detection_status"] == "suspicious")
        & (faces_df["detected_face_path"] != "")
    ].copy()

    candidates = candidates.sort_values(
        by="confidence",
        ascending=True,
    ).head(MAX_SUSPICIOUS_EXAMPLES)

    print("Đang tạo suspicious detection examples...")

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Suspicious examples",
    ):
        detected_path = resolve_project_path(row["detected_face_path"])

        image = cv2.imread(str(detected_path))

        if image is None:
            continue

        confidence = float(row["confidence"])
        face_id = str(row["face_id"])

        output_name = f"{face_id}__conf_{confidence:.3f}.jpg"
        output_path = SUSPICIOUS_EXAMPLES_DIR / output_name

        save_image(output_path, image)


# ============================================================
# 6. CROWDED SCENES
# ============================================================

def create_crowded_scene_examples(
    images_df: pd.DataFrame,
    faces_df: pd.DataFrame,
) -> None:
    """
    Chọn top ảnh đông người nhất và vẽ toàn bộ bbox.
    """
    crowded_images = images_df.sort_values(
        by="face_count",
        ascending=False,
    ).head(MAX_CROWDED_SCENE_EXAMPLES)

    print("Đang tạo crowded scene examples...")

    for _, image_row in tqdm(
        crowded_images.iterrows(),
        total=len(crowded_images),
        desc="Crowded scenes",
    ):
        image_id = str(image_row["image_id"])
        event_id = str(image_row["event_id"])
        face_count = int(image_row["face_count"])

        source_path = resolve_project_path(image_row["path"])

        image = cv2.imread(str(source_path))

        if image is None:
            continue

        faces_in_image = faces_df[
            (faces_df["image_id"] == image_id)
            & (faces_df["event_id"] == event_id)
        ]

        annotated = draw_detection_annotations(image, faces_in_image)
        annotated = resize_for_report(annotated)

        output_name = (
            f"{event_id}__{Path(image_id).stem}"
            f"__faces_{face_count:03d}.jpg"
        )
        output_path = CROWDED_SCENES_DIR / output_name

        save_image(output_path, annotated)


# ============================================================
# 7. ALIGNMENT COMPARISON
# ============================================================

def create_alignment_comparison_examples(
    faces_df: pd.DataFrame,
) -> None:
    """
    Tạo ảnh so sánh:
    crop chưa align | face đã align

    detected_faces thường kích thước khác nhau,
    nên resize về 224x224 để ghép cạnh nhau.
    """
    candidates = faces_df[
        (faces_df["crop_status"] == "success")
        & (faces_df["alignment_status"] == "success")
        & (faces_df["detected_face_path"] != "")
        & (faces_df["aligned_face_path"] != "")
        & (faces_df["detection_status"] == "accepted")
    ].copy()

    # Sample ngẫu nhiên để có đa dạng hơn,
    # tránh chỉ lấy toàn mặt giống nhau trong event đầu.
    candidates = candidates.sample(
        n=min(MAX_ALIGNMENT_COMPARISONS, len(candidates)),
        random_state=42,
    )

    print("Đang tạo alignment comparison examples...")

    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Alignment comparisons",
    ):
        detected_path = resolve_project_path(row["detected_face_path"])
        aligned_path = resolve_project_path(row["aligned_face_path"])

        before = cv2.imread(str(detected_path))
        after = cv2.imread(str(aligned_path))

        if before is None or after is None:
            continue

        before = cv2.resize(
            before,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )

        after = cv2.resize(
            after,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )

        comparison = np.hstack([before, after])

        # Gắn nhãn text
        canvas = np.full(
            (comparison.shape[0] + 40, comparison.shape[1], 3),
            255,
            dtype=np.uint8,
        )

        canvas[40:, :, :] = comparison

        cv2.putText(
            canvas,
            "Before Align",
            (35, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "After Align",
            (260, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        face_id = str(row["face_id"])
        output_name = f"{face_id}__alignment_comparison.jpg"
        output_path = ALIGNMENT_COMPARISON_DIR / output_name

        save_image(output_path, canvas)


# ============================================================
# 8. MAIN
# ============================================================

def create_face_detection_visualizations() -> None:
    if not IMAGES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy images_metadata.csv: {IMAGES_METADATA_PATH}"
        )

    if not FACES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy faces_metadata.csv: {FACES_METADATA_PATH}"
        )

    images_df = pd.read_csv(
        IMAGES_METADATA_PATH,
        keep_default_na=False,
    )

    faces_df = pd.read_csv(
        FACES_METADATA_PATH,
        keep_default_na=False,
    )

    # Đảm bảo folder tồn tại
    visualization_dirs = [
        ANNOTATED_SOURCES_DIR,
        CORRECT_EXAMPLES_DIR,
        SUSPICIOUS_EXAMPLES_DIR,
        CROWDED_SCENES_DIR,
        ALIGNMENT_COMPARISON_DIR,
    ]

    for directory in visualization_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU TẠO FACE DETECTION VISUALIZATIONS")
    print("=" * 70)
    print()

    create_annotated_source_examples(images_df, faces_df)
    create_correct_candidate_examples(faces_df)
    create_suspicious_candidate_examples(faces_df)
    create_crowded_scene_examples(images_df, faces_df)
    create_alignment_comparison_examples(faces_df)

    print()
    print("=" * 70)
    print("HOÀN TẤT VISUALIZATION")
    print("=" * 70)
    print(f"Annotated sources:      {ANNOTATED_SOURCES_DIR}")
    print(f"Correct candidates:     {CORRECT_EXAMPLES_DIR}")
    print(f"Suspicious candidates:  {SUSPICIOUS_EXAMPLES_DIR}")
    print(f"Crowded scenes:         {CROWDED_SCENES_DIR}")
    print(f"Alignment comparisons:  {ALIGNMENT_COMPARISON_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    create_face_detection_visualizations()