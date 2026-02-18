import cv2
import wfdb
import numpy as np
from pathlib import Path


DATA_ROOT = Path("data/dar_hea")
IMAGE_OUTPUT = Path("data/segmentation_images")
MASK_OUTPUT = Path("data/segmentation_masks")

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT.mkdir(parents=True, exist_ok=True)


def generate_mask_for_record(record_folder: Path, record_base: str):

    img_path = record_folder / f"{record_base}-0.png"

    if not img_path.exists():
        return

    # Load image
    image = cv2.imread(str(img_path))
    height, width = image.shape[:2]

    # Load WFDB signal
    record = wfdb.rdrecord(str(record_folder / record_base))
    signal = record.p_signal  # shape (5000, 12)

    num_samples, num_leads = signal.shape

    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Horizontal scaling (map samples → width)
    x_coords = np.linspace(0, width - 1, num_samples).astype(int)

    # Divide image vertically into equal bands (one per lead)
    band_height = height // num_leads

    for lead_idx in range(num_leads):

        lead_signal = signal[:, lead_idx]

        lead_min = np.min(lead_signal)
        lead_max = np.max(lead_signal)

        if lead_max - lead_min == 0:
            continue

        normalized = (lead_signal - lead_min) / (lead_max - lead_min)

        y_base = lead_idx * band_height
        y_coords = y_base + (1 - normalized) * (band_height - 1)
        y_coords = y_coords.astype(int)

        for i in range(len(x_coords) - 1):
            cv2.line(
                mask,
                (x_coords[i], y_coords[i]),
                (x_coords[i + 1], y_coords[i + 1]),
                255,
                1
            )

    # Save image + mask
    cv2.imwrite(str(IMAGE_OUTPUT / f"{record_base}.png"), image)
    cv2.imwrite(str(MASK_OUTPUT / f"{record_base}_mask.png"), mask)


def generate_all_masks(max_records_per_folder=10):
    """
    Generate masks for limited number of records per folder (testing mode).
    """

    for folder in sorted(DATA_ROOT.iterdir()):

        if not folder.is_dir():
            continue

        print(f"Processing folder: {folder.name}")

        # Collect unique record bases inside this folder
        record_bases = sorted(
            {f.stem.replace("-0", "") for f in folder.glob("*-0.png")}
        )

        # Limit number for testing
        record_bases = record_bases[:max_records_per_folder]

        for record_base in record_bases:
            print(f"  Generating mask for: {record_base}")
            generate_mask_for_record(folder, record_base)

    print("Mask generation completed.")
