import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CHBMITWindowsDataset(Dataset):
    """
    Loads preprocessed windows from multiple patient .npz files.
    Each .npz should contain: windows, labels, patient_ids.
    """
    def __init__(self, processed_root: str, patients: list[str]):
        self.windows = []
        self.labels = []
        self.patient_ids = []

        for patient in patients:
            path = os.path.join(processed_root, f"{patient}.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path)
            windows = data["windows"].astype("float32")
            self.windows.append(data["windows"])      # (N, C, T)
            self.labels.append(data["labels"])        # (N,)
            self.patient_ids.append(data["patient_ids"])  # (N,)

        self.windows = np.concatenate(self.windows, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.patient_ids = np.concatenate(self.patient_ids, axis=0)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        x = self.windows[idx]       # (C, T)
        y = self.labels[idx]        # scalar {0,1}
        d = self.patient_ids[idx]   # scalar int

        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        d = torch.tensor(d, dtype=torch.long)
        return x, y, d
