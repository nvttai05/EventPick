import cv2
import numpy as np
from pathlib import Path

def get_image_metrics(image_path: Path):
    """
    Đọc ảnh aligned face và tính toán các chỉ số: kích thước, độ sáng, độ tương phản, độ nhòe.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tính Brightness (Độ sáng trung bình)
    brightness = gray.mean()
    
    # 2. Tính Contrast (Độ tương phản - Dùng độ lệch chuẩn std)
    contrast = gray.std()
    
    # 3. Tính Blur Score (Variance of Laplacian) [cite: 414]
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2)
    }

def estimate_pose_ratio(row) -> float:
    """
    Tận dụng 5 landmarks để tính toán độ bất đối xứng hình học của khuôn mặt.
    Mặt thẳng: Khoảng cách từ Mũi -> Mắt trái tương đương Mũi -> Mắt phải.
    Mặt nghiêng mạnh: Tỷ lệ này sẽ bị lệch rất lớn (ví dụ lệch > 2.0 lần).
    """
    try:
        nx, ny = float(row["nose_x"]), float(row["nose_y"])
        lex, ley = float(row["left_eye_x"]), float(row["left_eye_y"])
        rex, rey = float(row["right_eye_x"]), float(row["right_eye_y"])
        
        # Khoảng cách Euclide từ mũi đến 2 mắt
        dist_left = np.sqrt((nx - lex)**2 + (ny - ley)**2)
        dist_right = np.sqrt((nx - rex)**2 + (ny - rey)**2)
        
        if dist_left == 0 or dist_right == 0:
            return 999.0 # Lỗi dữ liệu landmarks
            
        # Trả về tỷ lệ lệch góc mặt
        ratio = max(dist_left, dist_right) / min(dist_left, dist_right)
        return round(ratio, 2)
    except Exception:
        return 999.0