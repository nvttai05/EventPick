from pathlib import Path
import cv2
import pandas as pd
import onnxruntime as ort

from src.face_detection.gpu_runtime import setup_onnx_gpu_runtime

# Setup GPU runtime trước khi InsightFace khởi tạo ONNX session
setup_onnx_gpu_runtime(debug=True)

from insightface.app import FaceAnalysis


from src.face_detection.config import (
    PROJECT_ROOT,
    IMAGES_METADATA_PATH,
    INSIGHTFACE_MODEL_NAME,
    ALLOWED_MODULES,
    PROVIDERS,
    CTX_ID,
    DET_SIZE,
)


def print_onnxruntime_providers() -> None:
    """
    Kiểm tra ONNX Runtime hiện nhìn thấy những execution provider nào.
    Nếu có CUDAExecutionProvider thì môi trường GPU đã được nhận diện.
    """
    providers = ort.get_available_providers()

    print("=" * 70)
    print("ONNX Runtime available providers:")
    for provider in providers:
        print(f" - {provider}")
    print("=" * 70)

    if "CUDAExecutionProvider" in providers:
        print("[OK] ONNX Runtime đã nhận CUDAExecutionProvider.")
    else:
        print("[WARNING] Không thấy CUDAExecutionProvider. Có thể InsightFace sẽ chạy CPU.")
    print()


def load_first_valid_image_path() -> Path:
    """
    Đọc images_metadata.csv và lấy ảnh đầu tiên còn tồn tại trên ổ đĩa.

    Trong metadata, cột path đang lưu dạng tương đối, ví dụ:
    data\\raw\\events\\Event_LopTai\\abc.jpg

    Vì vậy cần nối thêm PROJECT_ROOT để ra đường dẫn tuyệt đối.
    """
    if not IMAGES_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file metadata: {IMAGES_METADATA_PATH}"
        )

    df = pd.read_csv(IMAGES_METADATA_PATH)

    if "path" not in df.columns:
        raise ValueError("images_metadata.csv không có cột 'path'.")

    for raw_path in df["path"].dropna():
        raw_path = str(raw_path).strip()

        # Chuyển path trong metadata thành Path.
        # Path trên Windows xử lý tốt chuỗi có dấu \.
        metadata_path = Path(raw_path)

        # Nếu metadata_path là đường dẫn tuyệt đối thì dùng luôn.
        # Nếu là đường dẫn tương đối thì nối với PROJECT_ROOT.
        if metadata_path.is_absolute():
            image_path = metadata_path
        else:
            image_path = PROJECT_ROOT / metadata_path

        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        "Không tìm thấy ảnh hợp lệ nào từ cột 'path' trong images_metadata.csv."
    )

def load_retinaface_detector() -> FaceAnalysis:
    """
    Khởi tạo InsightFace chỉ với module detection.
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
    )

    print("[OK] Khởi tạo detector xong.")
    print()
    return app


def test_one_image() -> None:
    print_onnxruntime_providers()

    image_path = load_first_valid_image_path()
    print(f"Ảnh dùng để test: {image_path}")
    print()

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")

    print(f"Kích thước ảnh: width={image.shape[1]}, height={image.shape[0]}")
    print()

    app = load_retinaface_detector()

    print("Đang chạy detect face trên 1 ảnh...")
    faces = app.get(image)
    print("[OK] Detect xong.")
    print()

    print("=" * 70)
    print(f"Tổng số khuôn mặt phát hiện được: {len(faces)}")
    print("=" * 70)

    if len(faces) == 0:
        print("Không phát hiện được khuôn mặt nào trong ảnh test.")
        return

    for idx, face in enumerate(faces, start=1):
        bbox = face.bbox
        landmarks = face.kps
        confidence = face.det_score

        print(f"\nFace {idx}")
        print(f"  Confidence: {confidence:.4f}")
        print(
            "  BBox: "
            f"x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, "
            f"x2={bbox[2]:.2f}, y2={bbox[3]:.2f}"
        )
        print("  Landmarks:")
        print(f"    Left eye:     ({landmarks[0][0]:.2f}, {landmarks[0][1]:.2f})")
        print(f"    Right eye:    ({landmarks[1][0]:.2f}, {landmarks[1][1]:.2f})")
        print(f"    Nose:         ({landmarks[2][0]:.2f}, {landmarks[2][1]:.2f})")
        print(f"    Mouth left:   ({landmarks[3][0]:.2f}, {landmarks[3][1]:.2f})")
        print(f"    Mouth right:  ({landmarks[4][0]:.2f}, {landmarks[4][1]:.2f})")


if __name__ == "__main__":
    test_one_image()