import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class MetadataHandler:
    def __init__(self, csv_path="data/metadata/images_metadata.csv"):
        self.csv_path = Path(csv_path)
        self.columns = [
            'image_id', 'event_id', 'path', 'url', 
            'width', 'height', 'brightness', 'blur_score', 
            'face_count', 'timestamp'
        ]
        self._init_csv()

    def _init_csv(self):
        """Khởi tạo file CSV nếu chưa tồn tại"""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Tạo DataFrame trống với các cột đã định nghĩa và lưu xuống kèm header
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_path, index=False) 
            print(f"[INFO] Created metadata file with headers at {self.csv_path}")

    def add_image_entry(self, data_dict):
        """Thêm dữ liệu ảnh mới vào file, đảm bảo đúng thứ tự cột"""
        # 1. Tự động thêm timestamp nếu thiếu
        if 'timestamp' not in data_dict:
            data_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. Đảm bảo face_count luôn có giá trị (mặc định là 0 nếu chưa detect)
        if 'face_count' not in data_dict:
            data_dict['face_count'] = 0

        # 3. Ép dữ liệu theo đúng thứ tự các cột đã khai báo ở self.columns
        # Cách này giúp dữ liệu không bao giờ bị nhảy cột trong Excel
        row_data = []
        for col in self.columns:
            row_data.append(data_dict.get(col, ""))

        df_new = pd.DataFrame([row_data])
        df_new.to_csv(self.csv_path, mode='a', header=False, index=False)

    def get_stats(self):
        """Đọc nhanh thống kê hiện tại"""
        df = pd.read_csv(self.csv_path)
        return {
            "total_images": len(df),
            "events": df['event_id'].nunique(),
            "avg_brightness": df['brightness'].mean() if len(df) > 0 else 0
        }

# Để test nhanh module này
if __name__ == "__main__":
    handler = MetadataHandler()
    print(handler.get_stats())