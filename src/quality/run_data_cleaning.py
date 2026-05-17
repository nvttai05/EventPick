import os
import sys
import shutil
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2] 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.quality.processor import get_image_metrics, estimate_pose_ratio, validate_landmark_semantics
from src.quality.deduplicator import compute_dhash


def run_data_cleaning():
    DATA_DIR = PROJECT_ROOT / "data"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    FACES_METADATA_PATH = DATA_DIR / "metadata" / "faces_metadata.csv"
    ALIGNED_FACES_DIR = DATA_DIR / "interim" / "aligned_faces"
    
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
        "reason_invalid_geometry": 0,
        "reason_duplicate": 0,
        "reason_low_confidence": 0,
        "file_not_found": 0,
        "corrupted_image": 0
    }

    event_hashes = {}
    duplicate_records = []

    print(f"[OK] Đang đọc dữ liệu ảnh thực tế từ: {ALIGNED_FACES_DIR}")
    print("BẮT ĐẦU CHẠY PIPELINE LÒC: SÀN CONFIDENCE >= 0.70 + ADAPTIVE FILTER...")

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
        is_geom_valid, geom_reason = validate_landmark_semantics(row)

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
        # THIẾT LẬP BỘ LỌC THÍCH ỨNG TRÊN SÀN CHẶN CỨNG 0.70
        # ============================================================
        conf = float(row.get("confidence", 1.0))
        reasons = []

        #  Sàn chặn cứng detector nâng lên hẳn 0.70 để diệt nhiễu triệt để
        if conf < 0.70:
            reasons.append("low_detector_confidence")
            stats["reason_low_confidence"] += 1
            
        else:
            # Phân tầng thích ứng cho nhóm vượt qua sàn 0.70
            if conf >= 0.85:
                # Ảnh người thật siêu chắc chắn -> Bảo vệ ảnh thần thái, nới lỏng để giữ ảnh mờ nhẹ/nghiêng
                min_blur = 35.0
                max_pose = 2.4
                check_geometry = False # Độ tự tin quá cao, miễn kiểm tra hình học tránh lỗi landmarks rìa
            else:
                # Vùng biên 0.70 <= confidence < 0.85 -> Áp ngưỡng tiêu chuẩn để lọc kỹ
                min_blur = 60.0
                max_pose = 1.9
                check_geometry = True

            # Kiểm tra hình học ngữ nghĩa
            if check_geometry and not is_geom_valid:
                reasons.append(geom_reason)
                stats["reason_invalid_geometry"] += 1

            # Lọc thích ứng độ nhòe (Blur score)
            if metrics["blur_score"] < min_blur:
                reasons.append("blurry")
                stats["reason_blurry"] += 1

            # Lọc thích ứng góc nghiêng mặt (Pose ratio)
            if pose_ratio > max_pose:
                reasons.append("extreme_pose")
                stats["reason_extreme_pose"] += 1

        # Các quy tắc ánh sáng và tương phản giữ ổn định đa nền tảng
        if metrics["brightness"] < 35.0 or metrics["brightness"] > 235.0:
            reasons.append("bad_lighting")
            stats["reason_bad_lighting"] += 1

        if metrics["contrast"] < 12.0:
            reasons.append("low_contrast")
            stats["reason_low_contrast"] += 1

        if event_id not in event_hashes:
            event_hashes[event_id] = set()

        if dhash_str in event_hashes[event_id] and dhash_str != "":
            reasons.append("duplicate_hash")
            stats["reason_duplicate"] += 1
            duplicate_records.append({
                "face_id": row["face_id"],
                "event_id": event_id,
                "image_id": row["image_id"],
                "dHash": dhash_str
            })

        # Xử lý phân loại đầu ra
        if len(reasons) > 0:
            faces_df.at[idx, "quality_status"] = "filtered"
            faces_df.at[idx, "filter_reason"] = "|".join(reasons)
            stats["total_filtered"] += 1
        else:
            event_clean_dir = CLEANED_FACES_DIR / event_id
            event_clean_dir.mkdir(parents=True, exist_ok=True)
            clean_dest_path = event_clean_dir / filename

            shutil.copy(aligned_abs_path, clean_dest_path)

            clean_rel_path = str(clean_dest_path.relative_to(PROJECT_ROOT))
            faces_df.at[idx, "quality_status"] = "passed"
            faces_df.at[idx, "cleaned_face_path"] = clean_rel_path

            event_hashes[event_id].add(dhash_str)
            stats["total_passed"] += 1

    # Đồng bộ lưu trữ kết quả cho Sơn dùng
    faces_df.to_csv(FACES_METADATA_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_records).to_csv(DUPLICATE_REPORTS_DIR / "duplicate_faces_report.csv", index=False, encoding="utf-8-sig")

    summary_text = f"""======================================================================
THÔNG SỐ PIPELINE LÀM SẠCH 
======================================================================
[Tổng quan tập dữ liệu sau khi nâng sàn 0.70 & Adaptive]:
- Tổng số khuôn mặt đưa vào xử lý: {stats['total_processed']}
- Tổng số khuôn mặt ĐẠT CHUẨN (Passed): {stats['total_passed']} (Tỷ lệ: {round(stats['total_passed']/stats['total_processed']*100, 2)}%)
- Tổng số khuôn mặt BỊ LOẠI BỎ (Filtered): {stats['total_filtered']} (Tỷ lệ: {round(stats['total_filtered']/stats['total_processed']*100, 2)}%)

[Thống kê chi tiết các tiêu chí loại bỏ]:
(Lưu ý: Một khuôn mặt có thể vi phạm nhiều tiêu chí cùng lúc)
1. Nhiễu cứng Detector (Confidence < 0.70):   {stats['reason_low_confidence']} mặt
2. Ảnh bị nhòe/mất nét thích ứng (Blurry):     {stats['reason_blurry']} mặt
3. Lỗi cấu trúc hình học (Invalid Geometry):   {stats['reason_invalid_geometry']} mặt
4. Góc nghiêng quá lớn thích ứng (Pose):       {stats['reason_extreme_pose']} mặt
5. Ánh sáng lỗi (Quá tối/Cháy sáng):          {stats['reason_bad_lighting']} mặt
6. Độ tương phản kém (Low Contrast):          {stats['reason_low_contrast']} mặt
7. Trùng lặp hình ảnh (Duplicate dHash):      {stats['reason_duplicate']} mặt
8. Lỗi đọc file vật lý/Ảnh lỗi kỹ thuật:       {stats['file_not_found'] + stats['corrupted_image']} mặt
======================================================================
"""
    print(summary_text)
    with open(QUALITY_REPORTS_DIR / "cleaning_runtime_stats.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[SUCCESS] Đã xuất thông số chạy cụ thể tại: {QUALITY_REPORTS_DIR / 'cleaning_runtime_stats.txt'}")


if __name__ == "__main__":
    run_data_cleaning()