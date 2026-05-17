import cv2
import numpy as np

def compute_dhash(image_path, hash_size=8) -> str:
    """
    Tính dHash của khuôn mặt để phát hiện các ảnh trùng lặp tuyệt đối hoặc gần như nhau.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    
    # Chuyển sang ảnh xám và resize về kích thước (hash_size + 1, hash_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
    # Tính toán sự khác biệt giữa các điểm ảnh liền kề theo chiều ngang
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Chuyển mảng Booleans thành chuỗi ký tự nhị phân đại diện cho Hash
    return "".join(f"{int(b)}" for b in diff.flatten())