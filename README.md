
- `data/raw/events/`: Chứa ảnh gốc các sự kiện (đã đổi tên sang mã MD5).
- `data/metadata/`: Chứa các file quản lý thông tin (`images_metadata.csv`, `dataset_statistics.json`).

src/quality/processor.py: Tính độ sáng, độ nét.

src/utils/metadata_handler.py: Xử lý đọc/ghi file CSV chuẩn, không lệch cột.

scripts/process_local.py: Tự động đổi tên ảnh sang mã MD5 và đẩy vào kho events/.

scripts/dataset_stats.py: Xuất báo cáo thống kê ra Terminal và file .json.

scripts/dataset_optimize.py: Lọc ảnh trùng, ảnh lỗi, đảm bảo dataset "sạch".
