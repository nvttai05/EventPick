import os
import hashlib
from pathlib import Path
import sys

# Thêm đường dẫn để Python nhận diện thư mục 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metadata_handler import MetadataHandler
from src.quality.processor import get_image_metrics

def run_local_pipeline():
    handler = MetadataHandler()
    temp_root = Path("data/raw/temp")
    
    # Duyệt qua từng thư mục event trong temp
    for event_dir in temp_root.iterdir():
        if event_dir.is_dir():
            event_id = event_dir.name
            final_event_dir = Path(f"data/raw/events/{event_id}")
            final_event_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n--- Đang xử lý Event: {event_id} ---")
            
            # Duyệt qua tất cả ảnh trong event đó (kể cả thư mục con lồng nhau)
            for img_path in event_dir.rglob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        # 1. Tính toán chất lượng
                        metrics = get_image_metrics(str(img_path))
                        if metrics is None: continue
                        
                        # 2. Tạo Image ID (MD5) để đặt tên file
                        with open(img_path, "rb") as f:
                            img_id = hashlib.md5(f.read()).hexdigest()
                        
                        final_path = final_event_dir / f"{img_id}{img_path.suffix}"
                        
                        # 3. Di chuyển/Copy vào kho chính thức (data/raw/events/)
                        if not final_path.exists():
                            import shutil
                            shutil.copy(img_path, final_path)
                        
                        # 4. Ghi Metadata (đã có cơ chế chống lệch cột bạn vừa sửa)
                        meta_entry = {
                            "image_id": img_id,
                            "event_id": event_id,
                            "path": str(final_path),
                            "url": "local_import", # Vì mình tải tay
                            **metrics
                        }
                        handler.add_image_entry(meta_entry)
                        
                    except Exception as e:
                        print(f"Lỗi file {img_path.name}: {e}")

if __name__ == "__main__":
    run_local_pipeline()
    print("\n--- HOÀN THÀNH XỬ LÝ TẤT CẢ DỮ LIỆU CỤC BỘ ---")