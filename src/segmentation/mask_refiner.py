import cv2
import numpy as np


def remove_small_noise(binary, min_area=100):
    """
    Remove very tiny blobs only.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def remove_left_calibration(binary, left_ratio=0.06):
    """
    Remove calibration pulse from left margin only.
    """
    h, w = binary.shape
    left_width = int(w * left_ratio)

    binary[:, :left_width] = 0

    return binary


def remove_bottom_text(binary, bottom_ratio=0.08):
    """
    Remove bottom printed text (25mm/s, 10mm/mV).
    """
    h, w = binary.shape
    bottom_height = int(h * bottom_ratio)

    binary[h - bottom_height:, :] = 0

    return binary


def refine_mask(morph_image):
    """
    Minimal refinement:
    - Keep waveform intact
    - Remove only obvious layout artifacts
    """

    _, binary = cv2.threshold(morph_image, 127, 255, cv2.THRESH_BINARY)

    cleaned = remove_small_noise(binary, min_area=100)

    cleaned = remove_left_calibration(cleaned, left_ratio=0.06)

    cleaned = remove_bottom_text(cleaned, bottom_ratio=0.08)

    return cleaned
