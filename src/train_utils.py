import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import CHBMITWindowsDataset

def make_lopo_loaders(processed_root, all_patients, test_patient,
                      batch_size=64, num_workers=0):
    train_patients = [p for p in all_patients if p != test_patient]

    full_train = CHBMITWindowsDataset(processed_root, train_patients)
    test_dataset = CHBMITWindowsDataset(processed_root, [test_patient])

    # Stratified split for val
    labels = np.array([full_train[i][1] for i in range(len(full_train))])
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_pos_val = max(1, int(0.1 * len(idx_pos)))
    n_neg_val = max(1, int(0.1 * len(idx_neg)))

    val_idx = np.concatenate([idx_pos[:n_pos_val], idx_neg[:n_neg_val]])
    train_idx = np.concatenate([idx_pos[n_pos_val:], idx_neg[n_neg_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    from torch.utils.data import Subset
    train_subset = Subset(full_train, train_idx.tolist())
    val_subset = Subset(full_train, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


   