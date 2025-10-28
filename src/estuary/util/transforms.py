from typing import Any

import torch
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR


class PowerTransformTorch(IntensityAugmentationBase2D):
    def __init__(self, lambdas: list[float], max_val: int):
        super().__init__(p=1)

        tlambdas = torch.tensor(lambdas, dtype=torch.float32)
        # Build a (1, C, 1, 1) tensor filled with max_val to compute per-channel scale
        x = torch.full((1, len(lambdas), 1, 1), float(max_val), dtype=torch.float32)
        scale = self._yeo_johnson(x, tlambdas)  # shape (1, C, 1, 1)
        self.flags = {
            "lambdas": tlambdas,
            "scale": scale,
        }

    @staticmethod
    def _yeo_johnson(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Yeo–Johnson transform.
        x: (B,C,H,W) or (C,H,W) assumed channel-first with C == lam.numel().
        lam: (C,) tensor of per-channel lambdas.
        Uses torch.where for safe broadcasting (avoids boolean indexing shape errors).
        """
        # Ensure float32 for numerical stability/consistency
        x = x.to(torch.float32)
        lam = lam.to(torch.float32)

        # Broadcast lambdas along the channel axis
        # If input is (B,C,H,W, ...), channel dim = 1; if (C,H,W, ...), channel dim = 0
        if x.ndim >= 4:
            chan_dim = 1
        else:
            chan_dim = 0
        lam_shape = [1] * x.ndim
        lam_shape[chan_dim] = -1
        lam_b = lam.view(*lam_shape)

        eps = 1e-12

        # Conditions
        pos = x >= 0
        lam_is_zero = lam_b.abs() < eps
        lam_is_two = (lam_b - 2.0).abs() < eps

        # Positive branch for all elements (we'll mask with pos later)
        y_pos = torch.where(
            lam_is_zero,
            torch.log1p(x),
            ((x + 1.0).pow(lam_b) - 1.0) / lam_b,
        )

        # Negative branch for all elements (we'll mask with ~pos later)
        y_neg = torch.where(
            lam_is_two,
            -torch.log1p(-x),
            -(((1.0 - x).pow(2.0 - lam_b) - 1.0) / (2.0 - lam_b)),
        )

        # Select based on sign of x
        out = torch.where(pos, y_pos, y_neg)
        return out

    def apply_transform(
        self,
        inp: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        lambdas: Tensor = flags["lambdas"]
        scale = flags["scale"]

        # Apply per-channel YJ
        x_t = self._yeo_johnson(inp, lambdas)

        # Scale based on max pixel value. Values are now between 0 and 1.
        x_t = x_t / scale

        return x_t


def shift_channels(image: torch.Tensor, shifts: list[torch.Tensor]) -> torch.Tensor:
    """
    Shift all channels.

    Shift each image's channel by either shifts.
    """
    KORNIA_CHECK_IS_TENSOR(image)

    shifted = (image + torch.stack(shifts, dim=1).view(-1, len(shifts), 1, 1).to(image)).clamp_(
        min=0, max=1
    )

    return shifted


class RandomChannelShift(IntensityAugmentationBase2D):
    """Randomly shift each channel of an image.

    Args:
        shift_limit: maximum value up to which the shift value can be generated for each channel;
          recommended interval - [0, 1], should always be positive
        num_channels: The number of channels in the image.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Note:
        Input tensor must be float and normalized into [0, 1].
    """

    def __init__(
        self,
        shift_limit: float = 0.5,
        num_channels: int = 3,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        params = [
            (shift_limit, f"{i}_shift_limit", 0, (-shift_limit, shift_limit))
            for i in range(num_channels)
        ]
        self.num_channels = num_channels
        self.shift_limit = shift_limit
        self._param_generator = rg.PlainUniformGenerator(*params)  # type: ignore

    def apply_transform(
        self,
        inp: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Tensor | None = None,
    ) -> Tensor:
        channel_params = [params[f"{i}_shift_limit"] for i in range(self.num_channels)]
        return shift_channels(inp, channel_params)


class ScaleNormalization(IntensityAugmentationBase2D):
    """Normalize channels to the range [0, 1] by a fixed value"""

    def __init__(self, max_val: Tensor | float | int) -> None:
        super().__init__(p=1)
        max_val = torch.tensor(max_val, dtype=torch.float32)
        self.flags = {"max_val": max_val.view(1, -1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Tensor | None = None,
    ) -> Tensor:
        return input / flags["max_val"]


def contrast_stretch_torch(
    imgs: torch.Tensor, p_low: float = 2.0, p_high: float = 98.0
) -> torch.Tensor:
    """
    Perform contrast stretching on a batched torch tensor.

    Args:
        imgs: Tensor of shape (B, C, H, W) or (C, H, W).
        p_low: Low percentile (0-100).
        p_high: High percentile (0-100).

    Returns:
        Tensor with values stretched to [0, 1].
    """
    added_batch = False
    if imgs.ndim == 3:  # (C,H,W)
        imgs = imgs.unsqueeze(0)  # (1,C,H,W)
        added_batch = True

    B, C, H, W = imgs.shape
    flat = imgs.view(B, C, -1)

    # Mask out very small values
    threshold = 1e-6
    flat_masked = flat.clone()
    flat_masked[flat_masked < threshold] = float("nan")

    # Compute percentiles ignoring NaNs
    v_min = torch.nanquantile(flat_masked, q=p_low / 100.0, dim=-1, keepdim=True)
    v_max = torch.nanquantile(flat_masked, q=p_high / 100.0, dim=-1, keepdim=True)

    stretched = (flat - v_min) / (v_max - v_min + 1e-8)
    stretched = stretched.clamp(0.0, 1.0).view(B, C, H, W)

    if added_batch:
        stretched = stretched.squeeze(0)

    return stretched
