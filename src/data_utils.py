import os
import numpy as np
from scipy.signal import butter, filtfilt
import pyedflib  # can switch to mne.io.read_raw_edf if preferred

PROJECT_ROOT = os.path.expanduser("~/projects/seizure_project")
RAW_ROOT = os.path.join(PROJECT_ROOT, "data/chbmit/raw")
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "data/chbmit/processed")

def butter_bandpass_filter(data,
                           lowcut: float = 0.5,
                           highcut: float = 30.0,
                           fs: float = 256.0,
                           order: int = 4):
    """
    Apply Butterworth band-pass filter (0.5–30 Hz) channel-wise.
    data: (channels, time)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)

def load_edf_file(edf_path: str) -> np.ndarray:
    """
    Load one CHB-MIT EDF file and return data as (channels, time).
    """
    with pyedflib.EdfReader(edf_path) as f:
        n_channels = f.signals_in_file
        sigs = [f.readSignal(i) for i in range(n_channels)]
    data = np.asarray(sigs)
    return data

def segment_windows(data: np.ndarray,
                    fs: int = 256,
                    win_sec: int = 5) -> np.ndarray:
    """
    Segment continuous EEG into non-overlapping windows.
    data: (channels, time)
    Returns: (n_windows, channels, win_len)
    """
    win_len = fs * win_sec
    n_samples = data.shape[-1]
    n_windows = n_samples // win_len
    trimmed = data[:, :n_windows * win_len]
    windows = trimmed.reshape(data.shape[0], n_windows, win_len)
    windows = np.transpose(windows, (1, 0, 2))  # (n_windows, channels, win_len)
    return windows

def get_preictal_mask_for_record(win_start_times, seizure_intervals, preictal_len=1800.0):
    """
    win_start_times: array of window start times (seconds)
    seizure_intervals: list of (start, end) in seconds for this EDF
    Returns: labels array (0=interictal, 1=preictal)
    """
    labels = np.zeros_like(win_start_times, dtype=np.int64)
    for (sz_start, sz_end) in seizure_intervals:
        preictal_start = max(0.0, sz_start - preictal_len)
        preictal_end = sz_start
        in_preictal = (win_start_times >= preictal_start) & (win_start_times < preictal_end)
        labels[in_preictal] = 1
    return labels

import re

def parse_summary_file(summary_path: str):
    """
    Parse chbXX-summary.txt into:
    { "chb01_03.edf": [(start1, end1), ...], ... }
    Times in seconds from EDF start.
    """
    record_seizures = {}
    current_record = None
    sz_start = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            # Match "File Name: chb01_03.edf"
            if line.startswith("File Name:"):
                m = re.search(r"([a-zA-Z0-9_]+\.edf)", line)
                if m:
                    current_record = m.group(1)
                    if current_record not in record_seizures:
                        record_seizures[current_record] = []
                    sz_start = None

            # Match "Seizure Start Time:  2996 seconds"
            elif line.startswith("Seizure Start Time"):
                nums = re.findall(r"\d+", line)
                if nums:
                    sz_start = float(nums[0])

            # Match "Seizure End Time:  3036 seconds"
            elif line.startswith("Seizure End Time"):
                nums = re.findall(r"\d+", line)
                if nums and current_record is not None and sz_start is not None:
                    sz_end = float(nums[0])
                    record_seizures[current_record].append((sz_start, sz_end))
                    sz_start = None

    return record_seizures

import numpy as np

def get_window_start_times(n_windows: int, win_sec: float = 5.0):
    """
    Returns array of window start times (seconds) for non-overlapping windows.
    win k starts at k * win_sec.
    """
    return np.arange(n_windows, dtype=np.float32) * float(win_sec)

def label_windows_preictal_interictal(win_start_times: np.ndarray,
                                      seizure_intervals,
                                      preictal_len: float = 1800.0):
    """
    win_start_times: 1D array of window start times (seconds).
    seizure_intervals: list of (seizure_start, seizure_end) in seconds for this EDF.
    preictal_len: length of preictal period before seizure (seconds), e.g. 1800 = 30 min.

    Returns: labels (0 = interictal, 1 = preictal)
    """
    labels = np.zeros_like(win_start_times, dtype=np.int64)

    for (sz_start, sz_end) in seizure_intervals:
        preictal_start = max(0.0, sz_start - preictal_len)
        preictal_end = sz_start
        in_preictal = (win_start_times >= preictal_start) & (win_start_times < preictal_end)
        labels[in_preictal] = 1

        # Optionally, you can later exclude windows inside seizures if you don't want them at all.
        # in_seizure = (win_start_times >= sz_start) & (win_start_times < sz_end)
        # labels[in_seizure] = -1  # or mark to drop

    return labels

def process_patient(patient: str,
                    fs: float = 256.0,
                    win_sec: float = 5.0,
                    preictal_len: float = 1800.0):
    """
    Process all EDFs for one patient into windows, labels, and patient_ids.
    Returns:
      windows: (N, C, T)
      labels:  (N,)
      patient_ids: (N,) integer id for this patient
    """
    patient_dir = os.path.join(RAW_ROOT, patient)
    summary_path = os.path.join(patient_dir, f"{patient}-summary.txt")
    record_seizures = parse_summary_file(summary_path)

    all_windows = []
    all_labels = []

    # Map patient string to an integer ID (e.g., chb01 -> 0)
    patient_idx = int(patient.replace("chb", "")) - 1

    for fname in sorted(os.listdir(patient_dir)):
        if not fname.endswith(".edf"):
            continue
        edf_path = os.path.join(patient_dir, fname)
        seizure_intervals = record_seizures.get(fname, [])

        data = load_edf_file(edf_path)
        if data.shape[0] != 23:
            print("Skipping", fname, "channels:", data.shape[0])
            continue
        data_filt = butter_bandpass_filter(data, fs=fs)
        windows = segment_windows(data_filt, fs=int(fs), win_sec=int(win_sec))
        n_windows = windows.shape[0]
        if n_windows == 0:
            continue

        win_start_times = get_window_start_times(n_windows, win_sec=win_sec)
        labels = label_windows_preictal_interictal(
            win_start_times, seizure_intervals, preictal_len=preictal_len
        )

        all_windows.append(windows)
        all_labels.append(labels)

    if not all_windows:
        return None, None, None

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    patient_ids = np.full_like(labels, fill_value=patient_idx, dtype=np.int64)

    return windows, labels, patient_ids
