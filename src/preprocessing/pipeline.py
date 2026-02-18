import cv2
from pathlib import Path

from src.utils.io_utils import load_image, save_image
from src.utils.config import (
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    STEP4_DIR,
    STEP5_DIR,
    PSEUDO_MASK_DIR,
)

from .illumination import correct_illumination
from .grid_removal import remove_grid
from .contrast import apply_clahe
from .thresholding import adaptive_threshold
from .morphology import clean

from src.segmentation.mask_refiner import refine_mask


def run_preprocessing(image_path: Path):
    """
    Complete preprocessing pipeline for one ECG image.

    Steps:
    1. Illumination correction
    2. Red grid removal
    3. CLAHE contrast enhancement
    4. Adaptive thresholding
    5. Morphological cleaning
    6. Refined pseudo ground truth mask generation

    Parameters:
        image_path (Path): Full path to ECG image file
    """

    # ----------------------------
    # Load image
    # ----------------------------
    image = load_image(str(image_path))
    filename = image_path.name

    # ----------------------------
    # Step 1: Illumination Correction (grayscale branch)
    # ----------------------------
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    illum = correct_illumination(gray_original)
    save_image(illum, str(STEP1_DIR / filename))

    # ----------------------------
    # Step 2: Red Grid Removal (color branch)
    # ----------------------------
    grid_removed = remove_grid(image)
    save_image(grid_removed, str(STEP2_DIR / filename))

    # ----------------------------
    # Convert to grayscale for next stages
    # ----------------------------
    gray = cv2.cvtColor(grid_removed, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # Step 3: CLAHE Contrast Enhancement
    # ----------------------------
    clahe = apply_clahe(gray)
    save_image(clahe, str(STEP3_DIR / filename))

    # ----------------------------
    # Step 4: Adaptive Thresholding
    # ----------------------------
    binary = adaptive_threshold(clahe)
    save_image(binary, str(STEP4_DIR / filename))

    # ----------------------------
    # Step 5: Morphological Cleaning
    # ----------------------------
    morph = clean(binary)
    save_image(morph, str(STEP5_DIR / filename))

    # ----------------------------
    # Step 6: Refined Pseudo Ground Truth Mask
    # ----------------------------
    refined_mask = refine_mask(morph)
    save_image(refined_mask, str(PSEUDO_MASK_DIR / filename))

    return refined_mask
