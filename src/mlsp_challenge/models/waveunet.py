import torch
import torch.nn as nn
import speechbrain as sb

class WaveUNet(nn.Module):
    def __init__(self):
        super(WaveUNet, self).__init__()
        kernel_size=79
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=8, 
                out_channels=1, 
                kernel_size=kernel_size,
                padding=kernel_size//2
            ),
            nn.Tanh()
        )
    
    def forward(self, x):
        rep = self.compute_features(x)
        return self.model(rep)
    
    def compute_features(self, batch):
        return batch