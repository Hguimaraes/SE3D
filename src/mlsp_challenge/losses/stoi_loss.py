import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F

class STOILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stoi = sb.nnet.stoi_loss.stoi_loss

    def forward(self, y_hat, y, lens=None):
        y_hat, y = y_hat.transpose(1, 2), y.transpose(1, 2)
        return self.stoi(y_hat, y, lens) + F.l1_loss(y_hat.transpose(1, 2), y.transpose(1, 2))