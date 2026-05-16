import pandas as pd
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_stats(csv_path="data/metadata/images_metadata.csv", json_path="data/metadata/dataset_statistics.json"):
    if not os.path.exists(csv_path):
        print(f"[Lỗi] Không tìm thấy file {csv_path}")
        return

    # 1. Đọc dữ liệu từ CSV sạch
    df = pd.read_csv(csv_path)
    
    # 2. Tính toán các thông số
    total_records = len(df)
    unique_images = df['image_id'].nunique()
    event_distribution = df['event_id'].value_counts().to_dict()
    
    # Tính toán chất lượng trung bình
    avg_metrics = {
        "avg_brightness": round(float(df['brightness'].mean()), 2),
        "avg_blur_score": round(float(df['blur_score'].mean()), 2),
        "avg_width": int(df['width'].mean()),
        "avg_height": int(df['height'].mean())
    }

    # 3. Cấu trúc dữ liệu JSON
    stats_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "total_images": total_records,
            "unique_images": unique_images,
            "duplicate_count": total_records - unique_images
        },
        "quality_summary": avg_metrics,
        "event_breakdown": event_distribution
    }

    # 4. Lưu vào file JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=4, ensure_ascii=False)

    # 5. Vẫn in ra màn hình để Toàn theo dõi
    print("\n" + "="*40)
    print("BÁO CÁO THỐNG KÊ DATASET")
    print("="*40)
    print(f"- Tổng số ảnh: {total_records}")
    print(f"- Độ sáng TB: {avg_metrics['avg_brightness']}")
    print(f"- Độ nét TB: {avg_metrics['avg_blur_score']}")
    print(f"\n[Thông tin đã được lưu vào: {json_path}]")
    print("="*40)

if __name__ == "__main__":
    generate_stats()