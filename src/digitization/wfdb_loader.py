import wfdb
import matplotlib.pyplot as plt
from pathlib import Path


def load_wfdb_signal(record_path):
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.fs


def plot_wfdb_signal(signal, fs, filename):
    plt.figure(figsize=(12, 4))
    plt.plot(signal[:, 0])
    plt.title(f"Ground Truth ECG (fs={fs} Hz)")
    plt.tight_layout()

    output_dir = Path("data/step6_digitized_signal")
    plt.savefig(output_dir / f"{filename}_groundtruth.png")
    plt.close()
