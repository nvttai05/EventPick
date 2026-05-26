import cv2
import numpy as np
from pathlib import Path

def get_image_metrics(image_path: Path):
    """Tính toán Brightness, Contrast và Blur Score cho ảnh aligned face."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    contrast = gray.std()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2)
    }

def validate_landmark_semantics(row):
    """
    Kiểm tra cấu trúc hình học tương đối của các điểm landmarks so với Bounding Box gốc.
    Loại bỏ các trường hợp nhận diện lỗi hoặc bị che khuất dị thường.
    """
    try:
        # Lấy tọa độ bounding box gốc để tính tỷ lệ phân bổ tương đối
        x1, y1 = float(row["bbox_x1"]), float(row["bbox_y1"])
        x2, y2 = float(row["bbox_x2"]), float(row["bbox_y2"])
        bh = y2 - y1
        
        if bh <= 0:
            return False, "invalid_bbox_dimensions"

        # Tải tọa độ trục Y của các điểm landmark chính
        le_y = float(row["left_eye_y"])
        re_y = float(row["right_eye_y"])
        n_y = float(row["nose_y"])
        ml_y = float(row["mouth_left_y"])
        mr_y = float(row["mouth_right_y"])

        # 1. Kiểm tra thứ tự dọc: Mắt phải nằm TRÊN mũi, Mũi phải nằm TRÊN miệng
        if le_y > n_y or re_y > n_y:
            return False, "geometry_eyes_below_nose"
        if n_y > ml_y or n_y > mr_y:
            return False, "geometry_nose_below_mouth"

        # 2. Kiểm tra vị trí tương đối trong Box (Mắt ở nửa trên, miệng ở nửa dưới)
        le_y_rel = (le_y - y1) / bh
        re_y_rel = (re_y - y1) / bh
        ml_y_rel = (ml_y - y1) / bh
        mr_y_rel = (mr_y - y1) / bh

        if le_y_rel > 0.70 or re_y_rel > 0.70:
            return False, "geometry_eyes_too_low"
        if ml_y_rel < 0.40 or mr_y_rel < 0.40:
            return False, "geometry_mouth_too_high"

        return True, "passed"
    except Exception as e:
        return False, f"geometry_error_{str(e)}"

def estimate_pose_ratio(row) -> float:
    """Tận dụng 5 landmarks để ước lượng góc nghiêng mặt (Pose ratio)."""
    try:
        nx, ny = float(row["nose_x"]), float(row["nose_y"])
        lex, ley = float(row["left_eye_x"]), float(row["left_eye_y"])
        rex, rey = float(row["right_eye_x"]), float(row["right_eye_y"])
        
        dist_left = np.sqrt((nx - lex)**2 + (ny - ley)**2)
        dist_right = np.sqrt((nx - rex)**2 + (ny - rey)**2)
        
        if dist_left == 0 or dist_right == 0:
            return 999.0
            
        return round(max(dist_left, dist_right) / min(dist_left, dist_right), 2)
    except Exception:
        return 999.0