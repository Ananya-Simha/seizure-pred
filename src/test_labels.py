import os
import numpy as np
from data_utils import (
    RAW_ROOT,
    load_edf_file,
    butter_bandpass_filter,
    segment_windows,
    parse_summary_file,
    get_window_start_times,
    label_windows_preictal_interictal,
)

patient = "chb01"
summary_path = os.path.join(RAW_ROOT, patient, f"{patient}-summary.txt")
record_seizures = parse_summary_file(summary_path)

record = "chb01_03.edf"  # choose one that has a seizure in your summary
edf_path = os.path.join(RAW_ROOT, patient, record)

# Load and preprocess
data = load_edf_file(edf_path)                        # (C, T)
data_filt = butter_bandpass_filter(data, fs=256.0)
windows = segment_windows(data_filt, fs=256, win_sec=5)
n_windows = windows.shape[0]
print("Windows shape:", windows.shape)

# Window times and labels
win_start_times = get_window_start_times(n_windows, win_sec=5.0)
seizure_intervals = record_seizures.get(record, [])
print("Seizure intervals:", seizure_intervals)

labels = label_windows_preictal_interictal(win_start_times, seizure_intervals, preictal_len=1800.0)
print("Labels shape:", labels.shape)
print("Num preictal windows:", np.sum(labels == 1))
print("Num interictal windows:", np.sum(labels == 0))
