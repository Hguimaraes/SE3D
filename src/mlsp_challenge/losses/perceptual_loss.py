import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2VecModel
from torch_stoi import NegSTOILoss


# Perceptual Loss
# Paper: https://arxiv.org/pdf/2010.15174v3.pdf
# https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str, alpha:float=10):
        super().__init__()
        ckpt = torch.load(PRETRAINED_MODEL_PATH)
        self.alpha = alpha
        self.stoi = NegSTOILoss(sample_rate=16000)

        self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.feature_extractor
        self.model.eval()

    def forward(self, y_hat, y, lens=None):
        stoi_loss = self.stoi(y_hat, y)
        rep_y_hat, rep_y = map(self.model, [y_hat.squeeze(1), y.squeeze(1)])
        return self.alpha*torch.abs(rep_y_hat - rep_y).mean() + stoi_loss.mean()