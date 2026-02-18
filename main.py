import shutil
from pathlib import Path

from src.preprocessing.pipeline import run_preprocessing
from src.utils.config import (
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    STEP4_DIR,
    STEP5_DIR,
    PSEUDO_MASK_DIR,
)


DATA_ROOT = Path("data/dar_hea")


# --------------------------------------------------
# Utility: Clear output directories
# --------------------------------------------------
def clear_directory(directory: Path):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def reset_output_folders():
    print("Resetting output folders...")

    for folder in [
        STEP1_DIR,
        STEP2_DIR,
        STEP3_DIR,
        STEP4_DIR,
        STEP5_DIR,
        PSEUDO_MASK_DIR,
    ]:
        clear_directory(folder)

    print("Output folders cleared.\n")


# --------------------------------------------------
# Phase 2: Preprocessing + Pseudo Mask Generation
# --------------------------------------------------
def run_phase2(max_records_per_folder=10):

    total_count = 0

    for folder in sorted(DATA_ROOT.iterdir()):

        if not folder.is_dir():
            continue

        print(f"Processing folder: {folder.name}")

        png_files = sorted(folder.glob("*-0.png"))
        png_files = png_files[:max_records_per_folder]

        for img_path in png_files:
            print(f"  Processing: {img_path.name}")

            # Pass full path to preprocessing
            run_preprocessing(img_path)

            total_count += 1

    print(f"\nProcessed {total_count} images successfully.")


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":

    print("====================================")
    print(" ECG Preprocessing + Mask Pipeline ")
    print("====================================\n")

    reset_output_folders()

    run_phase2(max_records_per_folder=10)

    print("\nPipeline finished successfully.")
