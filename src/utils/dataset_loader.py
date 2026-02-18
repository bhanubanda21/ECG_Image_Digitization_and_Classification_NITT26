import shutil
from pathlib import Path
from .config import TEST_SOURCE_DIR, RAW_DIR


def load_first_n_images(n=10):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(TEST_SOURCE_DIR.glob("*.*"))[:n]

    for img_path in images:
        destination = RAW_DIR / img_path.name
        shutil.copy(img_path, destination)

    print(f"{len(images)} images copied to raw folder.")
