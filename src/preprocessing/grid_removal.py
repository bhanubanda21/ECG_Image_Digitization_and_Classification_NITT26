import cv2
import numpy as np


def remove_grid(image):
    """
    Suppress red ECG grid using HSV masking.
    Converts detected red grid pixels to white.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 | mask2

    # Create a copy to modify
    output = image.copy()

    # Set red grid pixels to white
    output[red_mask > 0] = [255, 255, 255]

    return output
