from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
from tqdm import tqdm

from src.crop_face_val.config import (
    PROJECT_ROOT,
    RAW_EVENTS_DIR,
    METADATA_DIR,
    IMAGES_METADATA_PATH,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
}


def to_project_relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def build_images_metadata() -> None:
    if not RAW_EVENTS_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy folder input: {RAW_EVENTS_DIR}")

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = []

    # RAW_EVENTS_DIR = data/val/Event_Public
    # Bên trong có n000002, n000003, ...
    for class_dir in sorted(RAW_EVENTS_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix in SUPPORTED_EXTENSIONS:
                image_paths.append(image_path)

    records = []

    print("=" * 70)
    print("BẮT ĐẦU TẠO images_metadata.csv")
    print("=" * 70)
    print(f"Input folder: {RAW_EVENTS_DIR}")
    print(f"Tổng ảnh tìm thấy: {len(image_paths)}")
    print("=" * 70)

    for image_path in tqdm(image_paths, desc="Reading images"):
        class_id = image_path.parent.name
        image_id = image_path.name

        image = cv2.imread(str(image_path))

        if image is None:
            width = ""
            height = ""
            status = "error"
            error = "OpenCV không đọc được ảnh"
        else:
            height, width = image.shape[:2]
            status = "valid"
            error = ""

        records.append({
            "image_id": image_id,
            "event_id": class_id,  # ở dataset này event_id chính là folder n000002, n000003...
            "path": to_project_relative_path(image_path),
            "url": "local_import",
            "width": width,
            "height": height,
            "brightness": "",
            "blur_score": "",
            "face_count": 0,
            "accepted_face_count": 0,
            "suspicious_face_count": 0,
            "detection_status": "pending",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metadata_status": status,
            "metadata_error": error,
        })

    df = pd.DataFrame(records)

    df.to_csv(
        IMAGES_METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("HOÀN TẤT TẠO images_metadata.csv")
    print("=" * 70)
    print(f"Tổng ảnh ghi metadata: {len(df)}")
    print(f"Đã lưu tại: {IMAGES_METADATA_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    build_images_metadata()