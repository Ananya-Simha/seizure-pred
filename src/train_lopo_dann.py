import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

from data_utils import PROCESSED_ROOT
from models import FeatureExtractor1D, SeizureClassifier, DomainDiscriminator
from grl import GradientReversalLayer
from train_utils import make_lopo_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train_one_epoch_dann(model_feat, model_clf, model_domain, grl,
                         loader, optimizer, task_criterion, dom_criterion,
                         lambda_domain: float, id_map):
    model_feat.train()
    model_clf.train()
    model_domain.train()
    total_loss = 0.0

    for x, y, d in loader:
        x = x.to(device, non_blocking=True)                # (B, C, T)
        y = y.float().to(device)        # task labels {0,1}
        d = d.to(device)                # domain labels: patient IDs (global)
        d_mapped = torch.tensor([id_map[int(di)] for di in d.cpu().numpy()], dtype=torch.long, device=device)

        optimizer.zero_grad()

        # Map global patient IDs to 0..num_domains-1 for train patients
        # Assume all_patients and test_patient known at outer scope; pass a mapping if needed
        # For now: assume dataset patient_ids are already 0..num_domains-1 for train patients.

        z = model_feat(x)

        # Task head
        logits_task = model_clf(z).squeeze(-1)
        loss_task = task_criterion(logits_task, y)

        # Domain head with GRL
        z_rev = grl(z)
        logits_dom = model_domain(z_rev)      # (B, num_domains)
        loss_dom = dom_criterion(logits_dom, d_mapped)

        loss = loss_task + lambda_domain * loss_dom
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
            x = x.to(device, non_blocking=True)
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
    # Patients you have processed
    all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 11)]  # chb01–chb03, extend later

    results = []

    for test_patient in all_patients:
        print(f"\n=== Test patient: {test_patient} ===")

        train_loader, val_loader, test_loader = make_lopo_loaders(
            PROCESSED_ROOT, all_patients, test_patient,
            batch_size=64, num_workers=0
        )
        # Collect all domain labels (patient_ids) in train split
        all_train_domains = []
        for _, _, d in train_loader:
            all_train_domains.append(d.numpy())
        import numpy as np
        all_train_domains = np.concatenate(all_train_domains)

        # Build mapping old_id -> 0..K-1
        unique_domains = sorted(set(all_train_domains.tolist()))
        id_map = {old: new for new, old in enumerate(unique_domains)}
        num_domains = len(unique_domains)
        print("Domain id_map:", id_map)

        # Infer input shape
        x_sample, _, _ = next(iter(train_loader))
        in_channels = x_sample.shape[1]
        seq_len = x_sample.shape[2]

        model_feat = FeatureExtractor1D(in_channels=in_channels).to(device)
        feat_dim = model_feat(torch.randn(1, in_channels, seq_len).to(device)).shape[1]
        model_clf = SeizureClassifier(in_features=feat_dim).to(device)

        num_domains = len(all_patients)-1
        model_domain = DomainDiscriminator(in_features=feat_dim, num_domains=num_domains).to(device)
        grl = GradientReversalLayer(lambda_=0.1).to(device)

        # Compute class weighting for this LOPO split
        import numpy as np 
        all_train_labels = []
        for _, y, _ in train_loader:
            all_train_labels.append(y.numpy())
        all_train_labels = np.concatenate(all_train_labels)
        num_pos = (all_train_labels == 1).sum()
        num_neg = (all_train_labels == 0).sum()
        pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float32).to(device)
        print("Train pos:", num_pos, "neg:", num_neg, "pos_weight:", pos_weight.item())

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            list(model_feat.parameters()) + list(model_clf.parameters()) + list(model_domain.parameters()),
            lr=1e-3
        )
        dom_criterion = nn.CrossEntropyLoss()

        # Train a few epochs
        lambda_domain = 0.1  # or schedule this per epoch

        for epoch in range(1, 6):
            train_loss = train_one_epoch_dann(
                model_feat, model_clf, model_domain, grl,
                train_loader, optimizer,
                task_criterion=criterion,
                dom_criterion=dom_criterion,
                lambda_domain=lambda_domain, 
                id_map=id_map
            )
            val_acc, val_sens, val_auc = evaluate(model_feat, model_clf, val_loader)
            print(f"Epoch {epoch} | loss {train_loss:.4f} | "
                  f"val acc {val_acc:.3f} sens {val_sens:.3f} auc {val_auc:.3f}")
            test_acc, test_sens, test_auc = evaluate(model_feat, model_clf, test_loader) 
            print(f"Test ({test_patient}) | acc {test_acc:.3f} sens {test_sens:.3f} auc {test_auc:.3f}")

        results.append((test_patient, test_acc, test_sens, test_auc))

    # Save results to CSV
    import csv, os
    csv_path = os.path.join(os.path.dirname(__file__), "dann_lopo_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "acc", "sens", "auc"])
        writer.writerows(results)
    print("Saved baseline LOPO results to:", csv_path)



if __name__ == "__main__":
    main()
