import os
import sys
import shutil
from pathlib import Path

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
# ============================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2] 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

# Gọi trực tiếp các hàm từ 2 file processor.py và deduplicator.py ông đã viết
from src.quality.processor import get_image_metrics, estimate_pose_ratio
from src.quality.deduplicator import compute_dhash


def run_data_cleaning():
    DATA_DIR = PROJECT_ROOT / "data"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    FACES_METADATA_PATH = DATA_DIR / "metadata" / "faces_metadata.csv"
    ALIGNED_FACES_DIR = DATA_DIR / "interim" / "aligned_faces"
    
    # Các thư mục Output chuẩn theo file Word
    CLEANED_FACES_DIR = DATA_DIR / "interim" / "cleaned_faces"
    QUALITY_REPORTS_DIR = REPORTS_DIR / "quality_reports"
    DUPLICATE_REPORTS_DIR = REPORTS_DIR / "duplicate_reports"

    CLEANED_FACES_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DUPLICATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not FACES_METADATA_PATH.exists():
        print(f"[LỖI] Không tìm thấy file metadata tại: {FACES_METADATA_PATH}")
        return

    if not ALIGNED_FACES_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục ảnh tại: {ALIGNED_FACES_DIR}")
        return

    faces_df = pd.read_csv(FACES_METADATA_PATH)

    # Khởi tạo các cột lưu thông số chạy thực tế
    faces_df["blur_score"] = 0.0
    faces_df["brightness"] = 0.0
    faces_df["contrast"] = 0.0
    faces_df["pose_ratio"] = 0.0
    faces_df["dHash"] = ""
    faces_df["quality_status"] = "pending"
    faces_df["filter_reason"] = ""
    faces_df["cleaned_face_path"] = ""

    stats = {
        "total_processed": 0,
        "total_passed": 0,
        "total_filtered": 0,
        "reason_blurry": 0,
        "reason_bad_lighting": 0,
        "reason_low_contrast": 0,
        "reason_extreme_pose": 0,
        "reason_duplicate": 0,
        "reason_near_duplicate": 0,
        "reason_low_confidence": 0,
        "file_not_found": 0,
        "corrupted_image": 0
    }

    event_hashes = {}  # Lưu danh sách hash theo từng sự kiện
    duplicate_records = []

    print(f"[OK] Đang đọc dữ liệu ảnh từ: {ALIGNED_FACES_DIR}")
    print("BẮT ĐẦU QUY TRÌNH LỌC...")

    for idx, row in tqdm(faces_df.iterrows(), total=len(faces_df), desc="Processing"):
        stats["total_processed"] += 1

        raw_path = str(row.get("aligned_face_path", ""))
        filename = Path(raw_path.replace('\\', '/')).name
        event_id = str(row.get("event_id", ""))

        aligned_abs_path = ALIGNED_FACES_DIR / event_id / filename

        if not aligned_abs_path.exists():
            stats["file_not_found"] += 1
            stats["total_filtered"] += 1
            faces_df.at[idx, "quality_status"] = "filtered"
            faces_df.at[idx, "filter_reason"] = "file_not_found"
            continue

        metrics = get_image_metrics(aligned_abs_path)
        pose_ratio = estimate_pose_ratio(row)
        dhash_str = compute_dhash(aligned_abs_path)

        if metrics is None:
            faces_df.at[idx, "quality_status"] = "filtered"
            faces_df.at[idx, "filter_reason"] = "corrupted_image"
            stats["corrupted_image"] += 1
            stats["total_filtered"] += 1
            continue

        faces_df.at[idx, "blur_score"] = metrics["blur_score"]
        faces_df.at[idx, "brightness"] = metrics["brightness"]
        faces_df.at[idx, "contrast"] = metrics["contrast"]
        faces_df.at[idx, "pose_ratio"] = pose_ratio
        faces_df.at[idx, "dHash"] = dhash_str

        # ============================================================
    
        # ============================================================
        reasons = []
        
        # 1. Hạ độ tự tin nhận diện xuống mức sàn (0.50)
        if float(row.get("confidence", 1.0)) < 0.50:
            reasons.append("low_detector_confidence")
            stats["reason_low_confidence"] += 1

        # 2. Hạ ngưỡng lọc nhòe từ 80.0 xuống 35.0 
        if metrics["blur_score"] < 35.0:
            reasons.append("blurry")
            stats["reason_blurry"] += 1

        # 3. Mở rộng biên độ ánh sáng từ [40-225] thành [25-242] (Tránh lọc oan ảnh hội trường)
        if metrics["brightness"] < 25.0 or metrics["brightness"] > 242.0:
            reasons.append("bad_lighting")
            stats["reason_bad_lighting"] += 1

        # 4. Hạ ngưỡng tương phản tối thiểu xuống 8.0
        if metrics["contrast"] < 8.0:
            reasons.append("low_contrast")
            stats["reason_low_contrast"] += 1

        # 5. Tăng tỷ lệ lệch góc mặt từ 2.0 lên 2.8 (Giữ lại các khuôn mặt nghiêng vừa phải)
        if pose_ratio > 2.8:
            reasons.append("extreme_pose")
            stats["reason_extreme_pose"] += 1

        # 6. Kiểm tra Trùng lặp (Duplicate) & Gần trùng lặp (Near-duplicate) bằng Hamming Distance
        if event_id not in event_hashes:
            event_hashes[event_id] = set()

        is_dup = False
        is_near_dup = False
        
        if dhash_str != "":
            for existing_hash in event_hashes[event_id]:
                # Tính khoảng cách Hamming giữa 2 chuỗi nhị phân dHash
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(dhash_str, existing_hash))
                
                if hamming_dist == 0:
                    is_dup = True
                    break
                elif hamming_dist <= 4:  # Khác nhau dưới 4 bit => Ảnh chụp liên thanh (Near-duplicate)
                    is_near_dup = True
                    break

        if is_dup:
            reasons.append("duplicate_hash")
            stats["reason_duplicate"] += 1
            duplicate_records.append({"face_id": row["face_id"], "event_id": event_id, "type": "exact_duplicate", "dHash": dhash_str})
        elif is_near_dup:
            reasons.append("near_duplicate_hash")
            stats["reason_near_duplicate"] += 1
            duplicate_records.append({"face_id": row["face_id"], "event_id": event_id, "type": "near_duplicate", "dHash": dhash_str})

        # ============================================================
        # PHÂN LOẠI KẾT QUẢ
        # ============================================================
        if len(reasons) > 0:
            faces_df.at[idx, "quality_status"] = "filtered"
            faces_df.at[idx, "filter_reason"] = "|".join(reasons)
            stats["total_filtered"] += 1
        else:
            # Sao chép ảnh sạch sang cleaned_faces/
            event_clean_dir = CLEANED_FACES_DIR / event_id
            event_clean_dir.mkdir(parents=True, exist_ok=True)
            clean_dest_path = event_clean_dir / filename

            shutil.copy(aligned_abs_path, clean_dest_path)

            clean_rel_path = str(clean_dest_path.relative_to(PROJECT_ROOT))
            faces_df.at[idx, "quality_status"] = "passed"
            faces_df.at[idx, "cleaned_face_path"] = clean_rel_path

            event_hashes[event_id].add(dhash_str)
            stats["total_passed"] += 1

    # Lưu kết quả
    faces_df.to_csv(FACES_METADATA_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_records).to_csv(DUPLICATE_REPORTS_DIR / "duplicate_faces_report.csv", index=False, encoding="utf-8-sig")

    # ĐÚC KẾT THÔNG SỐ CHẠY CHI TIẾT
    summary_text = f"""======================================================================
THÔNG SỐ CHẠY PIPELINE LÀM SẠCH DỮ LIỆU (DATA CLEANING REPORT)
======================================================================
[Tổng quan tập dữ liệu sau khi nới lỏng bộ lọc]:
- Tổng số khuôn mặt đưa vào xử lý: {stats['total_processed']}
- Tổng số khuôn mặt ĐẠT CHUẨN (Passed): {stats['total_passed']} (Tỷ lệ: {round(stats['total_passed']/stats['total_processed']*100, 2)}%)
- Tổng số khuôn mặt BỊ LOẠI BỎ (Filtered): {stats['total_filtered']} (Tỷ lệ: {round(stats['total_filtered']/stats['total_processed']*100, 2)}%)

[Thống kê chi tiết các tiêu chí loại bỏ]:
(Lưu ý: Một khuôn mặt có thể vi phạm nhiều lỗi cùng lúc)
1. Không tìm thấy file ảnh vật lý (File Not Found): {stats['file_not_found']} mặt
2. Ảnh bị nhòe/mất nét (Blurry Face):      {stats['reason_blurry']} mặt
3. Ánh sáng lỗi (Quá tối/Cháy sáng):       {stats['reason_bad_lighting']} mặt
4. Độ tương phản kém (Low Contrast):       {stats['reason_low_contrast']} mặt
5. Góc nghiêng quá lớn (Extreme Pose):     {stats['reason_extreme_pose']} mặt
6. Trùng lặp tuyệt đối (Exact Duplicate):  {stats['reason_duplicate']} mặt
7. Gần trùng lặp chụp liên thanh (Near-Dup): {stats['reason_near_duplicate']} mặt
8. Nhiễu Detector (Confidence < 0.50):     {stats['reason_low_confidence']} mặt
9. Ảnh lỗi không đọc được (Corrupted):     {stats['corrupted_image']} mặt
======================================================================
"""
    print(summary_text)
    with open(QUALITY_REPORTS_DIR / "cleaning_runtime_stats.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[SUCCESS] Đã xuất thông số chạy cụ thể tại: {QUALITY_REPORTS_DIR / 'cleaning_runtime_stats.txt'}")


if __name__ == "__main__":
    run_data_cleaning()