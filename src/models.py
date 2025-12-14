import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor1D(nn.Module):
    """
    1D-CNN feature extractor G_f for EEG windows.
    Input:  (B, C, T)
    Output: (B, feat_dim)
    """
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)

        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)

        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)      # (B, channels, 1)
        x = x.squeeze(-1)     # (B, channels)
        return x              # feature vector z

class SeizureClassifier(nn.Module):
    """
    Task classifier G_y: preictal vs interictal.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, z):
        return self.fc(z)     # logits

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator G_d: predicts patient ID.
    """
    def __init__(self, in_features: int, num_domains: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, num_domains)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x)    # logits over domains
