import numpy as np


def extract_waveform(binary_image):
    """
    Extract 1D ECG signal from a binary row image.

    Returns:
        signal (1D array)
        left_offset (int)
        cropped_image (2D array)
    """

    if binary_image is None or binary_image.size == 0:
        return np.array([]), 0, None

    # Detect active columns
    col_sum = np.sum(binary_image == 255, axis=0)
    active_cols = np.where(col_sum > 10)[0]

    if len(active_cols) == 0:
        return np.array([]), 0, None

    left = active_cols[0]
    right = active_cols[-1]

    cropped = binary_image[:, left:right]

    height, width = cropped.shape
    signal = np.full(width, np.nan)

    for col in range(width):
        ys = np.where(cropped[:, col] == 255)[0]
        if len(ys) > 0:
            signal[col] = np.median(ys)

    return signal, left, cropped
