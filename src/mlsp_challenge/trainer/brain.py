import torch
import speechbrain as sb

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        return self.modules.model(batch["input"])

    
    def compute_objectives(self, predictions, batch, stage):
        return torch.nn.functional.l1_loss(predictions, batch["target"])