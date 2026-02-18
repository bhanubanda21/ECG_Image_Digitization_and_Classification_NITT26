import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def interpolate_signal(signal):
    nans = np.isnan(signal)
    not_nans = ~nans

    indices = np.arange(len(signal))
    signal[nans] = np.interp(indices[nans], indices[not_nans], signal[not_nans])

    return signal


def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val - min_val == 0:
        return signal

    return (signal - min_val) / (max_val - min_val)


def save_signal(signal, filename):
    output_dir = Path("data/step6_digitized_signal")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{filename}.npy", signal)
    np.savetxt(output_dir / f"{filename}.csv", signal, delimiter=",")


def plot_signal(signal, filename):
    output_dir = Path("data/step6_digitized_signal")

    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.gca().invert_yaxis()
    plt.title("Digitized ECG Signal")
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_plot.png")
    plt.close()


def overlay_signal_on_image(cropped_image, signal, filename):
    """
    Overlay extracted signal on binary image.
    """

    if cropped_image is None or len(signal) == 0:
        return

    overlay = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

    for x in range(len(signal)):
        y = int(signal[x])
        if not np.isnan(y):
            cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    output_dir = Path("data/step6_digitized_signal")
    cv2.imwrite(str(output_dir / f"{filename}_overlay.png"), overlay)


def compare_signals(digitized, ground_truth, filename):
    """
    Overlay digitized and ground truth signals in same plot.
    """

    if len(digitized) == 0 or len(ground_truth) == 0:
        return

    digitized = (digitized - np.min(digitized)) / (np.max(digitized) - np.min(digitized))
    ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))

    gt_resampled = np.interp(
        np.linspace(0, len(ground_truth) - 1, len(digitized)),
        np.arange(len(ground_truth)),
        ground_truth
    )

    plt.figure(figsize=(12, 4))
    plt.plot(gt_resampled, label="Ground Truth", linewidth=2)
    plt.plot(digitized, label="Digitized", alpha=0.7)
    plt.legend()
    plt.title("Digitized vs Ground Truth")
    plt.tight_layout()

    output_dir = Path("data/step6_digitized_signal")
    plt.savefig(output_dir / f"{filename}_comparison.png")
    plt.close()
