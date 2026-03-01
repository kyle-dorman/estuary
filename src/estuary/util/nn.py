from collections.abc import Sequence

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Multiclass Focal Loss (softmax variant).

    Expected inputs:
      - logits: (N, C, ...) float tensor (unnormalized scores)
      - targets: (N, ...) int64 tensor with class indices in [0, C-1]

    Notes:
      - This is NOT the sigmoid focal loss (binary/multilabel).
      - alpha can be:
          * None (no class weighting)
          * float (global scalar multiplier)
          * sequence/tensor of shape (C,) for per-class weights
    """

    def __init__(
        self,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.detach().float()
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            # float scalar
            self.alpha = float(alpha)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype != torch.long:
            targets = targets.long()

        # Flatten spatial dims into batch if needed
        if inputs.ndim > 2:
            # (N, C, H, W, ...) -> (N*H*W*..., C)
            c = inputs.shape[1]
            inputs_2d = inputs.permute(0, *range(2, inputs.ndim), 1).contiguous().view(-1, c)
            targets_1d = targets.contiguous().view(-1)
        else:
            inputs_2d = inputs
            targets_1d = targets

        # Mask ignore_index
        if self.ignore_index is not None:
            valid = targets_1d != self.ignore_index
            inputs_2d = inputs_2d[valid]
            targets_1d = targets_1d[valid]

        if inputs_2d.numel() == 0:
            # No valid elements
            return inputs.sum() * 0.0

        log_probs = F.log_softmax(inputs_2d, dim=1)  # (M, C)
        log_pt = log_probs.gather(1, targets_1d.unsqueeze(1)).squeeze(1)  # (M,)
        pt = log_pt.exp()

        focal_factor = (1.0 - pt).clamp(min=0.0).pow(self.gamma)
        loss = -focal_factor * log_pt  # (M,)

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                loss = loss * self.alpha
            else:
                alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
                if alpha.ndim != 1:
                    raise ValueError("alpha tensor must be 1D with shape (C,)")
                if alpha.numel() != inputs_2d.shape[1]:
                    raise ValueError(
                        f"alpha must have length C={inputs_2d.shape[1]}, got {alpha.numel()}"
                    )
                loss = loss * alpha.gather(0, targets_1d)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"Invalid reduction: {self.reduction}")
