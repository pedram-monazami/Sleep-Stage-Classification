import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    def __init__(self, in_channels=2):
        super(CNNBranch, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )

    def forward(self, x):
        return self.encoder(x)  # (B, 128, 16, 16)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (B, seq_len, input_dim)
        weights = self.attention(x)  # (B, seq_len, 1)
        weights = torch.softmax(weights, dim=1)  # Softmax over seq_len
        weighted_sum = torch.sum(weights * x, dim=1)  # (B, input_dim)
        return weighted_sum


class LSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.pooling = AttentionPooling(hidden_dim * 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, T, D)
        out, _ = self.lstm(x)
        pooled = self.pooling(out)
        return self.classifier(pooled)


class TransformerClassifierWithAttention(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, num_layers=2, ff_dim=256, num_classes=5, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooling = AttentionPooling(input_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, T, D)
        x = self.transformer(x)  # (B, seq_len, input_dim)
        pooled = self.pooling(x)  # (B, input_dim)
        return self.classifier(pooled)


class DualCNNLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(DualCNNLSTM, self).__init__()
        self.cwt_cnn = CNNBranch()
        self.wsst_cnn = CNNBranch()
        self.lstm = LSTMClassifierWithAttention(input_dim=256, num_classes=num_classes)  # 128 + 128 channels → 256 dims

    def forward(self, cwt, wsst):
        c1 = self.cwt_cnn(cwt)     # (B, 128, 16, 16)
        c2 = self.wsst_cnn(wsst)   # (B, 128, 16, 16)
        fused = torch.cat([c1, c2], dim=1)  # (B, 256, 16, 16)
        B, C, H, W = fused.shape
        seq = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, 256)
        return self.lstm(seq)


class DualCNNTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(DualCNNTransformer, self).__init__()
        self.cwt_cnn = CNNBranch()
        self.wsst_cnn = CNNBranch()
        self.transformer = TransformerClassifierWithAttention(input_dim=256, num_classes=num_classes)

    def forward(self, cwt, wsst):
        c1 = self.cwt_cnn(cwt)     # (B, 128, 16, 16)
        c2 = self.wsst_cnn(wsst)   # (B, 128, 16, 16)
        fused = torch.cat([c1, c2], dim=1)  # (B, 256, 16, 16)
        B, C, H, W = fused.shape
        seq = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, 256)
        return self.transformer(seq)


class SingleCNNLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super(SingleCNNLSTM, self).__init__()
        self.cnn = CNNBranch()
        self.lstm = LSTMClassifierWithAttention(input_dim=128, num_classes=num_classes)

    def forward(self, cwt, wsst):
        c = self.cnn(cwt)     # (B, 128, 16, 16)
        B, C, H, W = c.shape
        seq = c.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, 128)
        return self.lstm(seq)


class SingleCNNTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(SingleCNNTransformer, self).__init__()
        self.cnn = CNNBranch()
        self.transformer = TransformerClassifierWithAttention(input_dim=128, num_classes=num_classes)

    def forward(self, cwt, wsst):
        c = self.cnn(cwt)     # (B, 128, 16, 16)
        B, C, H, W = c.shape
        seq = c.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 256, 128)
        return self.transformer(seq)
