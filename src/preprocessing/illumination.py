import cv2
import numpy as np


def correct_illumination(gray):
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    corrected = cv2.divide(gray, blur, scale=255)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected
