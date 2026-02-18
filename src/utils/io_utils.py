import cv2
import os
from pathlib import Path


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def save_image(image, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)


def clear_directory(path):
    path = Path(path)
    if path.exists():
        for file in path.glob("*"):
            file.unlink()
    else:
        path.mkdir(parents=True, exist_ok=True)
