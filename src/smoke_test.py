import torch
from models import FeatureExtractor1D, SeizureClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy = torch.randn(8, 23, 1280).to(device)  # 8 windows, 23 channels, 5s at 256Hz

feat = FeatureExtractor1D(in_channels=23).to(device)
z = feat(dummy)

clf = SeizureClassifier(in_features=z.shape[1]).to(device)
logits = clf(z)

print("CUDA available:", torch.cuda.is_available())
print("Feature shape:", z.shape)
print("Logits shape:", logits.shape)
