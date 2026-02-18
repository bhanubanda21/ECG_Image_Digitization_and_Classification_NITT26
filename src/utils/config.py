from pathlib import Path

BASE_DIR = Path("data")

RAW_DIR = BASE_DIR / "raw"

STEP1_DIR = BASE_DIR / "step1"
STEP2_DIR = BASE_DIR / "step2"
STEP3_DIR = BASE_DIR / "step3"
STEP4_DIR = BASE_DIR / "step4"
STEP5_DIR = BASE_DIR / "step5"

PSEUDO_MASK_DIR = BASE_DIR / "pseudo_masks"

# Create directories
for d in [
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    STEP4_DIR,
    STEP5_DIR,
    PSEUDO_MASK_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
