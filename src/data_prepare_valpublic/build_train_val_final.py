from pathlib import Path
import shutil
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"D:\Hoctap\CK_KHDL\EventPick_v1")

TRAIN_CLEAN_DIR = PROJECT_ROOT / "data" / "train_clean"
PUBLIC_TRAIN_DIR = PROJECT_ROOT / "data" / "public_split" / "train" / "Event_Public"
PUBLIC_VAL_DIR = PROJECT_ROOT / "data" / "public_split" / "val" / "Event_Public"

TRAIN_FINAL_DIR = PROJECT_ROOT / "data" / "train_final"
VAL_FINAL_DIR = PROJECT_ROOT / "data" / "val_final"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OVERWRITE_OUTPUT = True


# ============================================================
# UTILS
# ============================================================

def get_images(folder: Path):
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def copy_folder_images(source_person_dir: Path, output_person_dir: Path):
    images = get_images(source_person_dir)

    output_person_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        shutil.copy2(image_path, output_person_dir / image_path.name)

    return len(images)


# ============================================================
# MAIN
# ============================================================

def main():
    if not TRAIN_CLEAN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy TRAIN_CLEAN_DIR: {TRAIN_CLEAN_DIR}")

    if not PUBLIC_TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy PUBLIC_TRAIN_DIR: {PUBLIC_TRAIN_DIR}")

    if not PUBLIC_VAL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy PUBLIC_VAL_DIR: {PUBLIC_VAL_DIR}")

    if TRAIN_FINAL_DIR.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa train_final cũ: {TRAIN_FINAL_DIR}")
        shutil.rmtree(TRAIN_FINAL_DIR)

    if VAL_FINAL_DIR.exists() and OVERWRITE_OUTPUT:
        print(f"Đang xóa val_final cũ: {VAL_FINAL_DIR}")
        shutil.rmtree(VAL_FINAL_DIR)

    TRAIN_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    train_final_images = 0
    train_final_classes = 0

    val_final_images = 0
    val_final_classes = 0

    print("=" * 80)
    print("BẮT ĐẦU BUILD TRAIN_FINAL / VAL_FINAL")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Copy self-collected train_clean
    # Structure:
    # train_clean/event/person/images
    #
    # Output:
    # train_final/SelfCollected/event__person/images
    # để tránh dataset.py hiểu event cũ là event và bị lẫn tầng.
    # Nhưng dataset.py cần root/event/person/image,
    # nên ta dùng event = SelfCollected, person = event__person.
    # ------------------------------------------------------------
    self_output_event_dir = TRAIN_FINAL_DIR / "SelfCollected"

    self_person_dirs = []

    for event_dir in sorted(TRAIN_CLEAN_DIR.iterdir()):
        if not event_dir.is_dir():
            continue

        for person_dir in sorted(event_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            self_person_dirs.append((event_dir.name, person_dir.name, person_dir))

    for event_name, person_name, person_dir in tqdm(self_person_dirs, desc="Copy self-collected"):
        final_person_name = f"{event_name}__{person_name}"
        output_person_dir = self_output_event_dir / final_person_name

        count = copy_folder_images(person_dir, output_person_dir)

        if count > 0:
            train_final_images += count
            train_final_classes += 1

    # ------------------------------------------------------------
    # 2. Copy public train
    # Output:
    # train_final/Event_Public/n000xxx/images
    # ------------------------------------------------------------
    public_train_person_dirs = [
        p for p in sorted(PUBLIC_TRAIN_DIR.iterdir())
        if p.is_dir()
    ]

    for person_dir in tqdm(public_train_person_dirs, desc="Copy public train"):
        output_person_dir = TRAIN_FINAL_DIR / "Event_Public" / person_dir.name

        count = copy_folder_images(person_dir, output_person_dir)

        if count > 0:
            train_final_images += count
            train_final_classes += 1

    # ------------------------------------------------------------
    # 3. Copy public val
    # Output:
    # val_final/Event_Public/n000xxx/images
    # ------------------------------------------------------------
    public_val_person_dirs = [
        p for p in sorted(PUBLIC_VAL_DIR.iterdir())
        if p.is_dir()
    ]

    for person_dir in tqdm(public_val_person_dirs, desc="Copy public val"):
        output_person_dir = VAL_FINAL_DIR / "Event_Public" / person_dir.name

        count = copy_folder_images(person_dir, output_person_dir)

        if count > 0:
            val_final_images += count
            val_final_classes += 1

    print()
    print("=" * 80)
    print("HOÀN TẤT BUILD TRAIN_FINAL / VAL_FINAL")
    print("=" * 80)
    print(f"Train final classes: {train_final_classes}")
    print(f"Train final images : {train_final_images}")
    print(f"Val final classes  : {val_final_classes}")
    print(f"Val final images   : {val_final_images}")
    print(f"Train final dir    : {TRAIN_FINAL_DIR}")
    print(f"Val final dir      : {VAL_FINAL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()