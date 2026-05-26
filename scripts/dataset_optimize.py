import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def optimize_dataset(csv_path="data/metadata/images_metadata.csv"):
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # --- THỰC HIỆN TỐI ƯU ---
    
    # 1. Loại bỏ trùng lặp MD5
    df = df.drop_duplicates(subset=['image_id'], keep='first')
    
    # 2. Lọc theo tiêu chuẩn kỹ thuật (Có thể điều chỉnh ngưỡng ở đây)
    # Loại ảnh quá tối (bình thường < 30 là đen kịt) hoặc quá nhỏ
    df = df[(df['brightness'] >= 25) & (df['width'] >= 400)]
    
    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"\n[KẾT QUẢ TỐI ƯU]")
    print(f"- Đã loại bỏ: {removed} ảnh (Trùng lặp hoặc lỗi kỹ thuật)")
    print(f"- Số lượng ảnh sạch còn lại: {final_count}")
    
    # Chỉ lưu khi có sự thay đổi
    if removed > 0:
        confirm = input("Bạn có chắc chắn muốn cập nhật lại file CSV? (y/n): ")
        if confirm.lower() == 'y':
            df.to_csv(csv_path, index=False)
            print("[Xong] Metadata đã được tối ưu hóa.")
        else:
            print("[Hủy] Không có thay đổi nào được lưu.")
    else:
        print("[Info] Dataset của bạn đã rất sạch, không cần tối ưu thêm.")

if __name__ == "__main__":
    optimize_dataset()