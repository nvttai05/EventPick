from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis

from src.face_detection.gpu_runtime import setup_onnx_gpu_runtime
from src.face_detection.config import (
    PROJECT_ROOT,
    IMAGES_METADATA_PATH,
    FACES_METADATA_PATH,
    DETECTION_FAILED_IMAGES_PATH,
    INSIGHTFACE_MODEL_NAME,
    ALLOWED_MODULES,
    PROVIDERS,
    CTX_ID,
    DET_SIZE,
    DETECTION_THRESHOLD,
    ACCEPTED_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
)


# ============================================================
# 1. GPU RUNTIME
# ============================================================

setup_onnx_gpu_runtime(debug=False)


# ============================================================
# 2. PATH UTILITIES
# ============================================================

def resolve_project_path(path_value: str) -> Path:
    """
    Chuyển path trong metadata thành đường dẫn tuyệt đối.

    Metadata của dự án lưu relative path, ví dụ:
    data\\raw\\events\\Event_LopTai\\abc.jpg
    """
    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# 3. DETECTION STATUS
# ============================================================

def classify_detection_status(confidence: float) -> str:
    """
    Gắn trạng thái cho từng face detection.
    """
    if confidence >= ACCEPTED_THRESHOLD:
        return "accepted"

    if confidence >= SUSPICIOUS_THRESHOLD:
        return "suspicious"

    return "rejected"


# ============================================================
# 4. LOAD RETINAFACE
# ============================================================

def load_retinaface_detector() -> FaceAnalysis:
    """
    Khởi tạo RetinaFace detector thông qua InsightFace.
    Chỉ dùng module detection.
    """
    print("Đang khởi tạo RetinaFace detector...")

    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL_NAME,
        allowed_modules=ALLOWED_MODULES,
        providers=PROVIDERS,
    )

    app.prepare(
        ctx_id=CTX_ID,
        det_size=DET_SIZE,
        det_thresh=DETECTION_THRESHOLD,
    )

    print("[OK] RetinaFace detector đã sẵn sàng.")
    return app


# ============================================================
# 5. FACE METADATA RECORD
# ============================================================

def build_face_record(
    row: pd.Series,
    face: Any,
    face_index: int,
    num_faces_in_source: int,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Tạo một record face-level metadata cho 1 khuôn mặt.
    """

    image_id = str(row["image_id"])
    event_id = str(row["event_id"])

    image_stem = Path(image_id).stem
    face_index_str = f"{face_index:02d}"

    face_id = f"{event_id}__{image_stem}__face{face_index_str}"

    bbox = face.bbox.astype(float)
    landmarks = face.kps.astype(float)
    confidence = float(face.det_score)

    x1, y1, x2, y2 = bbox.tolist()

    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    bbox_area = bbox_width * bbox_height

    image_area = float(image_width * image_height)
    face_area_ratio = bbox_area / image_area if image_area > 0 else np.nan

    detection_status = classify_detection_status(confidence)

    return {
        # A. Identity
        "face_id": face_id,
        "image_id": image_id,
        "event_id": event_id,
        "face_index": face_index,

        # B. Paths - crop/align sẽ cập nhật ở bước sau
        "source_image_path": str(row["path"]),
        "detected_face_path": "",
        "aligned_face_path": "",

        # C. Detector output
        "detector": "retinaface_det_10g",
        "confidence": confidence,
        "detection_status": detection_status,
        "num_faces_in_source": num_faces_in_source,

        # D. Source image size
        "image_width": image_width,
        "image_height": image_height,

        # E. Bounding box
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "face_area_ratio": face_area_ratio,

        # F. Five landmarks
        "left_eye_x": float(landmarks[0][0]),
        "left_eye_y": float(landmarks[0][1]),

        "right_eye_x": float(landmarks[1][0]),
        "right_eye_y": float(landmarks[1][1]),

        "nose_x": float(landmarks[2][0]),
        "nose_y": float(landmarks[2][1]),

        "mouth_left_x": float(landmarks[3][0]),
        "mouth_left_y": float(landmarks[3][1]),

        "mouth_right_x": float(landmarks[4][0]),
        "mouth_right_y": float(landmarks[4][1]),

        # G. Alignment - sẽ cập nhật ở bước align
        "alignment_applied": False,
        "alignment_status": "pending",
        "alignment_error": "",
        "output_width": "",
        "output_height": "",

        # H. Dành cho các bước sau
        "face_blur_score": "",
        "quality_status": "pending",
        "duplicate_group_id": "",
        "cluster_id": "",
        "label": "",
        "split": "",
    }


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def run_face_detection() -> None:
    """
    Chạy RetinaFace trên toàn bộ ảnh trong images_metadata.csv.

    Output:
    - faces_metadata.csv
    - images_metadata.csv được cập nhật face_count
    - failed_images.csv
    """

    if not IMAGES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy images_metadata.csv tại: {IMAGES_METADATA_PATH}"
        )

    images_df = pd.read_csv(IMAGES_METADATA_PATH)

    required_columns = {
        "image_id",
        "event_id",
        "path",
    }

    missing_columns = required_columns - set(images_df.columns)
    if missing_columns:
        raise ValueError(
            f"images_metadata.csv thiếu các cột bắt buộc: {missing_columns}"
        )

    # Đảm bảo các cột cập nhật tồn tại
    if "face_count" not in images_df.columns:
        images_df["face_count"] = 0

    if "accepted_face_count" not in images_df.columns:
        images_df["accepted_face_count"] = 0

    if "suspicious_face_count" not in images_df.columns:
        images_df["suspicious_face_count"] = 0

    if "detection_status" not in images_df.columns:
        images_df["detection_status"] = "pending"

    app = load_retinaface_detector()

    all_face_records: list[dict] = []
    failed_image_records: list[dict] = []

    print()
    print("Bắt đầu chạy face detection toàn bộ dataset...")
    print()

    for idx, row in tqdm(
        images_df.iterrows(),
        total=len(images_df),
        desc="Detecting faces",
    ):
        image_id = str(row["image_id"])
        event_id = str(row["event_id"])
        metadata_path = str(row["path"])

        absolute_image_path = resolve_project_path(metadata_path)

        try:
            image = cv2.imread(str(absolute_image_path))

            if image is None:
                raise ValueError("OpenCV không đọc được ảnh.")

            image_height, image_width = image.shape[:2]

            faces = app.get(image)

            # Sắp xếp face theo confidence giảm dần
            faces = sorted(
                faces,
                key=lambda face: float(face.det_score),
                reverse=True,
            )

            num_faces = len(faces)

            accepted_count = 0
            suspicious_count = 0

            for face_index, face in enumerate(faces, start=1):
                confidence = float(face.det_score)
                detection_status = classify_detection_status(confidence)

                if detection_status == "accepted":
                    accepted_count += 1
                elif detection_status == "suspicious":
                    suspicious_count += 1

                record = build_face_record(
                    row=row,
                    face=face,
                    face_index=face_index,
                    num_faces_in_source=num_faces,
                    image_width=image_width,
                    image_height=image_height,
                )

                all_face_records.append(record)

            # Cập nhật image-level metadata
            images_df.at[idx, "face_count"] = num_faces
            images_df.at[idx, "accepted_face_count"] = accepted_count
            images_df.at[idx, "suspicious_face_count"] = suspicious_count

            if num_faces > 0:
                images_df.at[idx, "detection_status"] = "has_face"
            else:
                images_df.at[idx, "detection_status"] = "no_face"

        except Exception as error:
            images_df.at[idx, "face_count"] = 0
            images_df.at[idx, "accepted_face_count"] = 0
            images_df.at[idx, "suspicious_face_count"] = 0
            images_df.at[idx, "detection_status"] = "error"

            failed_image_records.append({
                "image_id": image_id,
                "event_id": event_id,
                "path": metadata_path,
                "error": str(error),
            })

    # Lưu face metadata
    faces_df = pd.DataFrame(all_face_records)
    faces_df.to_csv(
        FACES_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Cập nhật images metadata
    images_df.to_csv(
        IMAGES_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Lưu log ảnh lỗi
    failed_df = pd.DataFrame(failed_image_records)
    failed_df.to_csv(
        DETECTION_FAILED_IMAGES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("HOÀN TẤT FACE DETECTION")
    print("=" * 70)
    print(f"Tổng ảnh xử lý: {len(images_df)}")
    print(f"Tổng khuôn mặt detect: {len(faces_df)}")
    print(f"Ảnh lỗi khi xử lý: {len(failed_df)}")
    print()
    print(f"Đã tạo: {FACES_METADATA_PATH}")
    print(f"Đã cập nhật: {IMAGES_METADATA_PATH}")
    print(f"Đã tạo log lỗi: {DETECTION_FAILED_IMAGES_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_face_detection()