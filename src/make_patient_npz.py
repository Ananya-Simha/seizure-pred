# in a small script, e.g. src/make_many_npz.py
from data_utils import process_patient, PROCESSED_ROOT
import numpy as np
import os

patients = [f"chb{str(i).zfill(2)}" for i in range(1, 11)]  # chb01–chb10

for patient in patients:
    windows, labels, patient_ids = process_patient(patient)
    if windows is None:
        print("No data for", patient)
        continue
    out_path = os.path.join(PROCESSED_ROOT, f"{patient}.npz")
    np.savez_compressed(out_path, windows=windows, labels=labels, patient_ids=patient_ids)
    print("Saved", patient, "->", out_path, windows.shape, labels.shape)
