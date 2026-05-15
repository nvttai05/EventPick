import cv2

def get_image_metrics(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tính Brightness (0-255)
    brightness = gray.mean()
    
    # Tính Blur (Càng cao càng nét, dưới 100 thường là nhòe)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        "width": w,
        "height": h,
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2)
    }