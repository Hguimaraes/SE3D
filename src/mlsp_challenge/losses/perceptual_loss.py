import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2VecModel


# Perceptual Loss
# Paper: https://arxiv.org/pdf/2010.15174v3.pdf
# https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
class PerceptualLoss(nn.Module):
    def __init__(self, PRETRAINED_MODEL_PATH:str):
        super().__init__()
        ckpt = torch.load(PRETRAINED_MODEL_PATH)

        self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.feature_extractor
        self.model.eval()

    def forward(self, y_hat, y):
        y_hat, y = y_hat.squeeze(1), y.squeeze(1)
        y_hat, y = map(self.model, [y_hat, y])
        return torch.abs(y_hat - y).mean()