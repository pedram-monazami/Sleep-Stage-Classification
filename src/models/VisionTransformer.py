import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=2):
        super(CNNBlock, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pooling),
        )
    def forward(self, x):
        x = self.cnn_block(x)
        return x

class CNNBranch(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, num_layers=4, kernel_size=3, stride=1, padding=1, pooling=2, same_channel_last_layer=True):
        super(CNNBranch, self).__init__()
        layers = []
        out_ch = base_channels
        if same_channel_last_layer: num_layers -= 1
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2 ** i)
            layers.append(CNNBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, pooling=pooling))

        if same_channel_last_layer:
            layers.append(CNNBlock(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, pooling=pooling))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
        # [B, base_channels * (2 ^ (num_layers - 1)), img_size % (2 ^ num_layers)] if same_channel_last_layer=False else
        # [B, base_channels * (2 ^ (num_layers - 2)), img_size % (2 ^ num_layers)]


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=2, patch_size=16, emb_size=384, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)  # [B, emb, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1, expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class ViTBranch(nn.Module):
    def __init__(self, in_channels=4, img_size=256, patch_size=16, emb_size=384, depth=6, num_heads=6, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(emb_size, num_heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])  # use CLS token output
        return x  # [B, emb_size]


class ResidualViTBranch(nn.Module):
    def __init__(self, in_channels=(128, 2), img_size=(32, 256), patch_size=(2, 16), emb_size=192, depth=4, num_heads=6, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels[0], patch_size[0], emb_size, img_size[0])
        self.cwt_patch_embed = PatchEmbedding(in_channels[1], patch_size[1], emb_size, img_size[1])
        self.wsst_patch_embed = PatchEmbedding(in_channels[1], patch_size[1], emb_size, img_size[1])
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, 3 * num_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(emb_size, num_heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x, x1_raw, x2_raw):
        B = x.shape[0]
        x = self.patch_embed(x)
        x1 = self.cwt_patch_embed(x1_raw)
        x2 = self.wsst_patch_embed(x2_raw)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x, x1, x2), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])  # use CLS token output
        return x  # [B, emb_size]


class DualViT(nn.Module):
    def __init__(self, in_channels=2, emb_size=384, num_classes=5):
        super().__init__()
        self.cwt_branch = ViTBranch(in_channels=in_channels, emb_size=emb_size)
        self.wsst_branch = ViTBranch(in_channels=in_channels, emb_size=emb_size)

        self.classifier = nn.Sequential(
            nn.Linear(emb_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, cwt_input, wsst_input):
        cwt_feat = self.cwt_branch(cwt_input)    # [B, 384]
        wsst_feat = self.wsst_branch(wsst_input)  # [B, 384]
        combined = torch.cat((cwt_feat, wsst_feat), dim=1)  # [B, 768]
        return self.classifier(combined)  # [B, num_classes]


class SingleViT(nn.Module):
    def __init__(self, in_channels=4, emb_size=384, num_classes=5):
        super().__init__()
        self.vit_branch = ViTBranch(in_channels=in_channels, emb_size=emb_size)

        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        vit = self.vit_branch(data)    # [B, 384]
        return self.classifier(vit)  # [B, num_classes]


class DualCNNViT(nn.Module):
    def __init__(self, num_classes=5):
        super(DualCNNViT, self).__init__()
        self.cwt_cnn = CNNBranch()
        self.wsst_cnn = CNNBranch()
        self.vit = ViTBranch(in_channels=256, patch_size=1, emb_size=256, img_size=16, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, cwt, wsst):
        cwt_feat = self.cwt_cnn(cwt)     # (B, 128, 16, 16)
        wsst_feat = self.wsst_cnn(wsst)   # (B, 128, 16, 16)
        combined = torch.cat((cwt_feat, wsst_feat), dim=1)  # [B, 256, 16, 16]
        vit_out = self.vit(combined)
        return self.classifier(vit_out)  # [B, num_classes]


class DualResidualCNNViT(nn.Module):
    def __init__(self, num_classes=5):
        super(DualResidualCNNViT, self).__init__()
        self.cwt_cnn = CNNBranch(in_channels=2, base_channels=32, num_layers=3)
        self.wsst_cnn = CNNBranch(in_channels=2, base_channels=32, num_layers=3)
        self.vit = ResidualViTBranch()
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, cwt, wsst):
        cwt_feat = self.cwt_cnn(cwt)     # (B, 64, 32, 32)
        wsst_feat = self.wsst_cnn(wsst)   # (B, 64, 32, 32)
        combined = torch.cat((cwt_feat, wsst_feat), dim=1)  # [B, 128, 32, 32]
        vit_out = self.vit(combined, cwt, wsst)
        return self.classifier(vit_out)  # [B, num_classes]