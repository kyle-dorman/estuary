from typing import Any

import kornia.augmentation as K
import torch
import torch.nn.functional as F
from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.contrib import diamond_square
from kornia.core import Tensor, as_tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR
from kornia.enhance import normalize_min_max
from kornia.utils.helpers import _extract_device_dtype
from torchvision.transforms import InterpolationMode


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


class RandomPlasmaFog(IntensityAugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    This is based on the original paper: TorMentor: Deterministic dynamic-path, data augmentations
    with fractals.
    See: :cite:`tormentor` for more details.

    .. note::
        This function internally uses :func:`kornia.contrib.diamond_square`.

    Args:
        roughness: value to scale during the recursion in the generation of the fractal map.
        fog_intensity: value that scales the intensity values of the generated maps.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 3, 4)
        >>> RandomPlasmaFog(roughness=(0.1, 0.7), p=1.)(img)
        tensor([[[[0.7682, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000]]]])

    """

    def __init__(
        self,
        roughness: tuple[float, float] = (0.1, 0.7),
        fog_intensity: tuple[float, float] = (0, 1.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (roughness, "roughness", None, None),
            (fog_intensity, "fog_intensity", None, None),
        )

    def apply_transform(
        self,
        image: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        B, C, H, W = image.shape
        assert C == 3
        roughness = params["roughness"].to(image)
        fog_intensity = params["fog_intensity"].to(image).view(-1, 1, 1, 1)
        fog_map = diamond_square(
            (B, 1, H + 20, W + 20), roughness, device=image.device, dtype=image.dtype
        )[:, :, 10:-10, 10:-10]
        fog_map = normalize_min_max(fog_map.contiguous(), 0.0, 1.0)
        fog_map = torch.exp(-2.0 * fog_map) / 4.0

        A = (1.0, 1.0, 1.0)
        At = torch.tensor(A, dtype=torch.float32, device=image.device).view(C, 1, 1)
        At = At.expand(B, C, 1, 1)

        hazy = image * fog_map + At * (1 - fog_map)
        out = (1 - fog_intensity) * image + fog_intensity * hazy

        return out


def _randn_like(input: Tensor, mean: float, std: float) -> Tensor:
    x = torch.randn_like(input)  # Generating on GPU is fastest with `torch.randn_like(...)`
    if std != 1.0:  # `if` is cheaper than multiplication
        x *= std
    if mean != 0.0:  # `if` is cheaper than addition
        x += mean
    return x


class ScaledRandomGaussianNoise(K.RandomGaussianNoise):
    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        if "gaussian_noise" in params:
            gaussian_noise = params["gaussian_noise"]
        else:
            gaussian_noise = _randn_like(input, mean=flags["mean"], std=flags["std"])
            self._params["gaussian_noise"] = gaussian_noise
        out = input + gaussian_noise

        out -= torch.amin(out, dim=(2, 3), keepdim=True)
        out /= torch.amax(out, dim=(2, 3), keepdim=True)

        mins = torch.amin(input, dim=(2, 3), keepdim=True)
        maxs = torch.amax(input, dim=(2, 3), keepdim=True)

        out = out * (maxs - mins) + mins

        return out


def _build_affine_pixel_matrix(
    center: tuple[float, float],
    angle_deg: torch.Tensor,
    edge_shift: torch.Tensor,
    offset_dx: torch.Tensor,
    offset_dy: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Build a batched 2x3 affine matrix in pixel coordinates for the described transformation.
    Supports scalar inputs or 1D tensors (B,). Returns (B,2,3).
    """
    cx, cy = center

    # Determine batch, device, dtype
    B = int(angle_deg.shape[0])
    device = angle_deg.device
    dtype = (
        torch.float32
        if angle_deg.dtype not in (torch.float16, torch.float32, torch.float64)
        else angle_deg.dtype
    )

    theta = torch.deg2rad(angle_deg)
    Ht = torch.tensor(H, dtype=dtype, device=device)
    Wt = torch.tensor(H, dtype=dtype, device=device)
    HWt = torch.tensor(max(H, W), dtype=dtype, device=device)

    # Effective length for shear (broadcasted)
    L_eff = torch.abs(Ht * torch.cos(theta)) + torch.abs(Wt * torch.sin(theta))
    L_eff = torch.where(L_eff < 1e-6, HWt, L_eff)
    k = edge_shift / L_eff

    # Prepare batched identities
    I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)  # noqa: E741

    # T(-cx,-cy)
    T1 = I.clone()
    T1[:, 0, 2] = -cx
    T1[:, 1, 2] = -cy

    # R(-θ)
    c = torch.cos(-theta)
    s = torch.sin(-theta)
    R1 = I.clone()
    R1[:, 0, 0] = c
    R1[:, 0, 1] = -s
    R1[:, 1, 0] = s
    R1[:, 1, 1] = c

    # Sx(k)
    S = I.clone()
    S[:, 0, 1] = k

    # R(θ)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R2 = I.clone()
    R2[:, 0, 0] = c
    R2[:, 0, 1] = -s
    R2[:, 1, 0] = s
    R2[:, 1, 1] = c

    # T(cx,cy)
    T2 = I.clone()
    T2[:, 0, 2] = cx
    T2[:, 1, 2] = cy

    # T(dx,dy)
    T3 = I.clone()
    T3[:, 0, 2] = offset_dx
    T3[:, 1, 2] = offset_dy

    # Compose batched: T3 @ T2 @ R2 @ S @ R1 @ T1
    M = T3 @ T2 @ R2 @ S @ R1 @ T1
    return M[:, :2, :]


def _pixel_to_norm_affine_matrix(
    H: int, W: int, device=None, dtype=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the 3x3 matrix that transforms normalized grid coordinates [-1,1] to pixel
    coordinates [0,W-1], [0,H-1] and its inverse. Used to convert pixel affine to normalized affine.
    """
    # norm_xy = (2*x/(W-1)-1, 2*y/(H-1)-1)
    # pixel_xy = ((norm_x+1)*(W-1)/2, (norm_y+1)*(H-1)/2)
    to_pixel = torch.eye(3, device=device, dtype=dtype)
    to_pixel[0, 0] = (W - 1) / 2.0
    to_pixel[0, 2] = (W - 1) / 2.0
    to_pixel[1, 1] = (H - 1) / 2.0
    to_pixel[1, 2] = (H - 1) / 2.0
    to_norm = torch.eye(3, device=device, dtype=dtype)
    to_norm[0, 0] = 2.0 / (W - 1)
    to_norm[0, 2] = -1.0
    to_norm[1, 1] = 2.0 / (H - 1)
    to_norm[1, 2] = -1.0
    return to_pixel, to_norm


def misregister_angled(
    image: torch.Tensor,
    params_per_band: list[dict],
    fill_value: float = 0.0,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
):
    # Handle input dims
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    elif image.ndim != 4:
        raise ValueError("image must be (C,H,W) or (B,C,H,W)")

    B, C, H, W = image.shape
    dtype = image.dtype

    # Center of rotation/shear in pixel coords
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # Build per-(B,C) pixel affine matrices => (B,C,2,3)
    affines_px = []
    for c in range(C):
        p = params_per_band[c]
        M_bc = _build_affine_pixel_matrix(
            (cx, cy), p["angle_deg"], p["edge_shift"], p["offset_dx"], p["offset_dy"], H, W
        )  # (B,2,3)
        affines_px.append(M_bc)

    affines_px = torch.stack(affines_px, dim=1).to(dtype=torch.float32)  # (B,C,2,3)

    # Convert pixel affine to normalized affine using the inverse pixel matrix
    to_pixel, to_norm = _pixel_to_norm_affine_matrix(H, W, device=image.device, dtype=torch.float32)

    # Build 3x3 from 2x3 and invert per (B,C)
    Ap = torch.eye(3, dtype=torch.float32, device=image.device).view(1, 1, 3, 3).repeat(B, C, 1, 1)
    Ap[:, :, :2, :] = affines_px
    Ap_inv = torch.inverse(Ap)

    # Broadcast multiply: (3,3) @ (B,C,3,3) @ (3,3) -> (B,C,3,3)
    Anorm = torch.einsum("ij,bcjk,kl->bcil", to_norm, Ap_inv, to_pixel)
    affines_norm = Anorm[:, :, :2, :].contiguous()  # (B,C,2,3)

    # Prepare input for grid_sample: (B*C,1,H,W)
    x = image.reshape(B * C, 1, H, W)

    # Grid for each (B*C)
    grid = F.affine_grid(affines_norm.reshape(B * C, 2, 3), [B * C, 1, H, W], align_corners=False)

    # Sample image
    mode = "bilinear" if interpolation == InterpolationMode.BILINEAR else "nearest"
    sampled = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Out-of-bounds mask to blend fill_value
    ones = torch.ones((B * C, 1, H, W), device=image.device, dtype=sampled.dtype)
    mask = F.grid_sample(ones, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    mask = (mask > 0.5).to(sampled.dtype)
    out = sampled + fill_value * (1 - mask)

    # Reshape back to (B,C,H,W), clamp, cast
    out = out.reshape(B, C, H, W)
    if dtype.is_floating_point:
        out = out.clamp(0, 1)
    out = out.to(dtype)
    return out.squeeze(0) if squeeze else out


class MisalignedImageGenerator(RandomGeneratorBase):
    def __init__(
        self,
        angle_deg: tuple[float, float],
        edge_shift: tuple[float, float],
        offset: tuple[float, float],
    ) -> None:
        super().__init__()
        self.angle_deg = angle_deg
        self.edge_shift = edge_shift
        self.offset = offset

    def __repr__(self) -> str:
        repr = f"angle_deg={self.angle_deg}, edge_shift={self.edge_shift}, offset={self.offset}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        angle_deg = as_tensor(self.angle_deg, device=device, dtype=dtype)
        self.angle_deg_sampler = UniformDistribution(
            angle_deg[0], angle_deg[1], validate_args=False
        )

        edge_shift = as_tensor(self.edge_shift, device=device, dtype=dtype)
        self.edge_shift_sampler = UniformDistribution(
            edge_shift[0], edge_shift[1], validate_args=False
        )

        offset = as_tensor(self.offset, device=device, dtype=dtype)
        self.offset_sampler = UniformDistribution(offset[0], offset[1], validate_args=False)

        # Randomly choose one channel to flip sign
        self.channel_sampler = UniformDistribution(1, 3, validate_args=False)

        # Randomly choose one axis to flip sign (0 for dx, 1 for dy)
        self.axis_sampler = UniformDistribution(0, 2, validate_args=False)

        # Randomly choose one axis to flip sign (0 for dx, 1 for dy)
        self.direction_sampler = UniformDistribution(0, 2, validate_args=False)

    def forward(
        self, batch_shape: tuple[int, ...], same_on_batch: bool = False
    ) -> dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.offset, self.angle_deg, self.edge_shift])
        # self.ksize_factor.expand((batch_size, -1))
        angle_deg = _adapted_rsampling((batch_size, 2), self.angle_deg_sampler).to(
            device=_device, dtype=_dtype
        )
        edge_shift = _adapted_rsampling((batch_size, 2), self.edge_shift_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        offset = _adapted_rsampling((batch_size, 2, 2), self.offset_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        return {
            "angle_deg": angle_deg,
            "edge_shift": edge_shift,
            "offset": offset,
        }


class MisalignedImage(IntensityAugmentationBase2D):
    def __init__(
        self,
        p: float,
        angle_deg: tuple[float, float],
        edge_shift: tuple[float, float],
        offset: tuple[float, float],
        border_crop: int,
    ):
        super().__init__(p=p)

        self._param_generator = MisalignedImageGenerator(
            angle_deg=angle_deg, edge_shift=edge_shift, offset=offset
        )

        self.flags = {
            "border_crop": torch.tensor(border_crop, dtype=torch.uint32),
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        B, _, _, _ = input.shape

        params_per_band = []
        # Reference band: no-op
        params_per_band.append(
            {
                "angle_deg": torch.zeros(B, device=input.device, dtype=input.dtype),
                "edge_shift": torch.zeros(B, device=input.device, dtype=input.dtype),
                "offset_dx": torch.zeros(B, device=input.device, dtype=input.dtype),
                "offset_dy": torch.zeros(B, device=input.device, dtype=input.dtype),
            }
        )

        # For each band to shift
        for index in range(2):
            offset = params["offset"][:, index]

            params_per_band.append(
                {
                    "angle_deg": params["angle_deg"][:, index],
                    "edge_shift": params["edge_shift"][:, index],
                    "offset_dx": offset[:, 0],
                    "offset_dy": offset[:, 1],
                }
            )

        data = misregister_angled(
            input,
            params_per_band=params_per_band,
        )

        crop = flags["border_crop"].item()
        if crop > 0:
            data = data[:, :, crop:-crop, crop:-crop]

        return data


if __name__ == "__main__":
    base = torch.randn((5, 3, 10, 10))
    aug = MisalignedImage(
        p=1.0, angle_deg=(5.0, 35.0), edge_shift=(-20, 20), offset=(-20, 20), border_crop=30
    )

    aug(base)
