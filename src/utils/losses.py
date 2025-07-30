import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # Copy your FocalLoss class code here exactly as you provided it.
    # No changes are needed.
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        super(FocalLoss, self).__init__()
        # ... rest of your FocalLoss code