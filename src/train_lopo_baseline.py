import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

from data_utils import PROCESSED_ROOT
from models import FeatureExtractor1D, SeizureClassifier
from train_utils import make_lopo_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model_feat, model_clf, loader, optimizer, criterion):
    model_feat.train()
    model_clf.train()
    total_loss = 0.0

    for x, y, d in loader:
        x = x.to(device)  # (B, C, T)
        y = y.float().to(device)  # BCEWithLogits -> float

        optimizer.zero_grad()
        z = model_feat(x)
        logits = model_clf(z).squeeze(-1)  # (B,)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model_feat, model_clf, loader):
    model_feat.eval()
    model_clf.eval()
    all_y = []
    all_prob = []

    with torch.no_grad():
        for x, y, d in loader:
            x = x.to(device)
            y = y.to(device)
            z = model_feat(x)
            logits = model_clf(z).squeeze(-1)
            prob = torch.sigmoid(logits)
            all_y.append(y.cpu())
            all_prob.append(prob.cpu())

    y_true = torch.cat(all_y).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)  # sensitivity
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return acc, sens, auc

def main():
    # Patients you processed, e.g. chb01–chb10
    all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 11)]
    test_patient = "chb01"  # start with one; later loop over all

    train_loader, val_loader, test_loader = make_lopo_loaders(
        PROCESSED_ROOT, all_patients, test_patient, batch_size=64, num_workers=0
    )

    # Infer input channels from one batch
    x_sample, _, _ = next(iter(train_loader))
    in_channels = x_sample.shape[1]

    model_feat = FeatureExtractor1D(in_channels=in_channels).to(device)
    feat_dim = model_feat(torch.randn(1, in_channels, x_sample.shape[2]).to(device)).shape[1]
    model_clf = SeizureClassifier(in_features=feat_dim).to(device)

    import numpy as np
    # collect train labels once
    all_train_labels = []
    for _, y, _ in train_loader:
        all_train_labels.append(y.numpy())
    all_train_labels = np.concatenate(all_train_labels)
    num_pos = (all_train_labels == 1).sum()
    num_neg = (all_train_labels == 0).sum()

    pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float32).to(device)
    print("Train pos:", num_pos, "neg:", num_neg, "pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(model_feat.parameters()) + list(model_clf.parameters()), lr=1e-3)

    for epoch in range(1, 6):  # few epochs for now
        train_loss = train_one_epoch(model_feat, model_clf, train_loader, optimizer, criterion)
        val_acc, val_sens, val_auc = evaluate(model_feat, model_clf, val_loader)
        print(f"Epoch {epoch} | loss {train_loss:.4f} | val acc {val_acc:.3f} sens {val_sens:.3f} auc {val_auc:.3f}")

    test_acc, test_sens, test_auc = evaluate(model_feat, model_clf, test_loader)
    print(f"Test ({test_patient}) | acc {test_acc:.3f} sens {test_sens:.3f} auc {test_auc:.3f}")

if __name__ == "__main__":
    main()
