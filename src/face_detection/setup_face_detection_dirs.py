from src.face_detection.config import (
    DETECTED_FACES_DIR,
    ALIGNED_FACES_DIR,
    METADATA_DIR,
    FACE_DETECTION_REPORT_DIR,
    ANNOTATED_SOURCES_DIR,
    CORRECT_EXAMPLES_DIR,
    SUSPICIOUS_EXAMPLES_DIR,
    CROWDED_SCENES_DIR,
    ALIGNMENT_COMPARISON_DIR,
)


def create_face_detection_directories() -> None:
    """
    Tạo toàn bộ thư mục cần thiết cho pipeline face detection.
    Nếu thư mục đã tồn tại thì không gây lỗi.
    """

    directories = [
        DETECTED_FACES_DIR,
        ALIGNED_FACES_DIR,
        METADATA_DIR,

        FACE_DETECTION_REPORT_DIR,

        ANNOTATED_SOURCES_DIR,
        CORRECT_EXAMPLES_DIR,
        SUSPICIOUS_EXAMPLES_DIR,
        CROWDED_SCENES_DIR,
        ALIGNMENT_COMPARISON_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("Đã tạo xong cấu trúc thư mục cho face detection pipeline.")


if __name__ == "__main__":
    create_face_detection_directories()