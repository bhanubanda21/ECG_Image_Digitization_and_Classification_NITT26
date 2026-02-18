import numpy as np


def split_into_4_rows(binary_image):
    """
    Stable 4-row ECG splitter.
    """

    row_sum = np.sum(binary_image == 255, axis=1)
    valid_rows = np.where(row_sum > 50)[0]

    if len(valid_rows) == 0:
        return []

    top = valid_rows[0]
    bottom = valid_rows[-1]

    cropped = binary_image[top:bottom, :]

    height = cropped.shape[0]
    row_height = height // 4

    rows = []

    for i in range(4):
        start = i * row_height
        end = (i + 1) * row_height
        rows.append(cropped[start:end, :])

    return rows
