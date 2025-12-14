import os
import numpy as np
from data_utils import RAW_ROOT, load_edf_file, butter_bandpass_filter, segment_windows

patient = "chb01"
fname = "chb01_01.edf"
edf_path = os.path.join(RAW_ROOT, patient, fname)

data = load_edf_file(edf_path)          # (C, T)
print("Raw shape:", data.shape)

data_filt = butter_bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=256.0)
windows = segment_windows(data_filt, fs=256, win_sec=5)
print("Windows shape:", windows.shape)
