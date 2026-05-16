import pandas as pd

from src.face_detection.config import (
    IMAGES_METADATA_PATH,
    FACES_METADATA_PATH,
    DETECTION_FAILED_IMAGES_PATH,
)


def check_detection_outputs() -> None:
    images_df = pd.read_csv(IMAGES_METADATA_PATH)
    faces_df = pd.read_csv(FACES_METADATA_PATH)
    try:
        failed_df = pd.read_csv(DETECTION_FAILED_IMAGES_PATH)
    except pd.errors.EmptyDataError:
        failed_df = pd.DataFrame(
            columns=["image_id", "event_id", "path", "error"]
        )

    print("=" * 70)
    print("KIỂM TRA KẾT QUẢ FACE DETECTION")
    print("=" * 70)

    print(f"Tổng ảnh trong images_metadata.csv: {len(images_df)}")
    print(f"Tổng face trong faces_metadata.csv: {len(faces_df)}")
    print(f"Tổng ảnh lỗi trong failed_images.csv: {len(failed_df)}")
    print()

    # Kiểm tra tổng face_count ở image-level có khớp số dòng face-level không
    total_face_count_from_images = int(images_df["face_count"].sum())
    total_faces_from_faces_metadata = len(faces_df)

    print(f"Tổng face_count từ images_metadata.csv: {total_face_count_from_images}")
    print(f"Tổng dòng faces_metadata.csv: {total_faces_from_faces_metadata}")

    if total_face_count_from_images == total_faces_from_faces_metadata:
        print("[OK] Tổng face_count khớp với faces_metadata.csv.")
    else:
        print("[WARNING] Tổng face_count KHÔNG khớp faces_metadata.csv.")
    print()

    # Thống kê detection status theo ảnh
    print("Phân bố detection_status ở mức ảnh:")
    print(images_df["detection_status"].value_counts(dropna=False))
    print()

    # Thống kê detection status theo face
    print("Phân bố detection_status ở mức khuôn mặt:")
    print(faces_df["detection_status"].value_counts(dropna=False))
    print()

    # Confidence
    print("Thống kê confidence:")
    print(faces_df["confidence"].describe())
    print()

    # Face count
    print("Thống kê số mặt trên mỗi ảnh:")
    print(images_df["face_count"].describe())
    print()

    # 10 ảnh đông người nhất
    print("Top 10 ảnh có nhiều khuôn mặt nhất:")
    top_crowded = images_df.sort_values(
        by="face_count",
        ascending=False
    )[["image_id", "event_id", "face_count", "path"]].head(10)

    print(top_crowded.to_string(index=False))

    print("=" * 70)


if __name__ == "__main__":
    check_detection_outputs()