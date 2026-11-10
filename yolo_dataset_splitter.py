"""
YOLO Dataset Splitter
--------------------------------
This script divides a dataset of YOLO-format images and labels
into training, validation, and test subsets. 
Images without labels are treated as negatives (empty .txt files are created).
"""

import random
from pathlib import Path
import shutil

# ==============================
# CONFIGURATION
# ==============================

IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
OUTPUT_DIR = Path("split_output")

# Split proportions (sum must equal 1.0)
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

# Allowed image formats
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Optional: for reproducibility
RANDOM_SEED = 42


# ==============================
# FUNCTIONS
# ==============================

def setup_split_dirs(base_dir: Path, splits=("train", "val", "test")) -> None:
    """Create output directory structure for dataset splits."""
    for split in splits:
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def list_files(directory: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    """Return {basename: full_path} for all matching files."""
    return {file.stem: file for file in directory.iterdir() if file.suffix.lower() in extensions}


def perform_split(items: list[str], ratios: dict[str, float]) -> dict[str, list[str]]:
    """Split items into train/val/test subsets based on ratios."""
    random.shuffle(items)
    total = len(items)

    n_train = int(total * ratios["train"])
    n_val = int(total * ratios["val"])

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }


def copy_data_to_split(
    splits: dict[str, list[str]], 
    img_dict: dict[str, Path], 
    lbl_dict: dict[str, Path], 
    output_root: Path
) -> None:
    """Copy corresponding image and label files to split directories."""
    for split, basenames in splits.items():
        for name in basenames:
            img_src = img_dict[name]
            img_dest = output_root / split / "images" / img_src.name
            shutil.copy(img_src, img_dest)

            # Handle label (create empty if missing)
            lbl_dest = output_root / split / "labels" / f"{name}.txt"
            if name in lbl_dict:
                shutil.copy(lbl_dict[name], lbl_dest)
            else:
                lbl_dest.touch()


def print_summary(splits: dict[str, list[str]], output_dir: Path) -> None:
    """Display final dataset split summary."""
    print("\n YOLO dataset split completed successfully!")
    print(f"Output directory: {output_dir.resolve()}")
    for k, v in splits.items():
        print(f"  • {k.capitalize():<9}: {len(v)} images")


# ==============================
# MAIN LOGIC
# ==============================

def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    setup_split_dirs(OUTPUT_DIR)

    images = list_files(IMAGES_DIR, IMAGE_EXTS)
    labels = list_files(LABELS_DIR, (".txt",))

    all_image_names = list(images.keys())

    splits = perform_split(all_image_names, SPLIT_RATIOS)
    copy_data_to_split(splits, images, labels, OUTPUT_DIR)
    print_summary(splits, OUTPUT_DIR)


if __name__ == "__main__":
    main()
