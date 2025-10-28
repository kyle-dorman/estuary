import torch
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return sigmoid_focal_loss(
            inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
