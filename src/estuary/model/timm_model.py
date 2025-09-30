import math
from typing import Any

import timm
import torch
import torch.nn as nn
from timm.layers import SelectAdaptivePool2d

from estuary.model.config import EstuaryConfig, ModelType


def named_modules(
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
):
    if not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        yield from named_modules(
            module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        yield name, module


def _init_weight_goog(m, n="", fix_group_fanout=True):
    """Weight initialization as per Tensorflow official implementations.

    Args:
        m (nn.Module): module to init
        n (str): module name
        fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/
            group convs

    Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
    * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if "routing_fn" in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)


def efficientnet_init_weights(model: nn.Module, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_modules():
        init_fn(m, n)

    # iterate and call any module.init_weights() fn, children first
    for _, m in named_modules(model):
        if hasattr(m, "init_weights"):
            m.init_weights()  # type: ignore


class CatAvgMaxMobileNetv4Head(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        head_hidden_size: int,
        pad_type: str,
        drop_rate: float,
        act_layer: Any,
        norm_layer: Any,
    ) -> None:
        super().__init__()

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type="catavgmax")
        num_pooled_chs = num_features * self.global_pool.feat_mult()
        # mobilenet-v4 post-pooling PW conv is followed by a norm+act layer
        self.conv_head = timm.layers.create_conv2d(
            num_pooled_chs, head_hidden_size, 1, padding=pad_type
        )  # never bias
        norm_act_layer = timm.layers.get_norm_act_layer(norm_layer, act_layer)
        assert norm_act_layer is not None
        self.norm_head = norm_act_layer(head_hidden_size)
        self.act2 = nn.Identity()
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = timm.layers.Linear(head_hidden_size, num_classes)

        efficientnet_init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Input features.
            pre_logits: Return features before final linear layer.

        Returns:
            Classification logits or features.
        """
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)


class LSEPool(nn.Module):
    """Log-Sum-Exp pooling (smooth max). beta→∞ ≈ max, beta→0 ≈ mean."""

    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,H,W)
        b = torch.clamp(self.beta, 1.0, 100.0)
        x_flat = (x * b).flatten(2)  # (B,C,HW)
        x_out = torch.logsumexp(x_flat, dim=-1) / b  # (B,C)
        return x_out.unsqueeze(-1).unsqueeze(-1)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps

    def forward(self, x):
        p = self.p if isinstance(self.p, float) else torch.clamp(self.p, min=1e-1, max=6)  # type: ignore
        return (
            torch.mean(x.clamp(min=self.eps).pow(p), dim=(-2, -1))
            .pow(1.0 / p)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )


class AttnPool(nn.Module):
    """Tiny spatial attention pooling: softmax over HxW, weighted sum over features."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.attn = nn.Conv2d(in_ch, 1, kernel_size=1)
        efficientnet_init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,H,W)
        a = self.attn(x)  # (B,1,H,W)
        a = torch.softmax(a.flatten(2), dim=-1).view_as(a)
        # (B, C, 1, 1)
        return (x * a).flatten(2).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)


class TimmModel(nn.Module):
    # catavgmax gem lse attn
    def __init__(self, conf: EstuaryConfig, num_classes: int) -> None:
        super().__init__()

        assert conf.model_type == ModelType.TIMM
        in_chans = conf.bands.num_channels()

        if conf.pretrained:
            assert in_chans == 3, conf.bands

        _timm_pools = {"avg", "max", "avgmax", "catavgmax"}

        self.head = None

        if conf.global_pool == "catavgmax" and "v4" in conf.model_name:
            kwargs = dict(
                model_name=conf.model_name,
                pretrained=conf.pretrained,
                in_chans=in_chans,
                drop_path_rate=conf.drop_path,
                num_classes=1,
            )
            self.model = timm.create_model(**kwargs)  # type: ignore
            assert isinstance(self.model, timm.models.MobileNetV3)
            assert self.model.num_features is not None
            self.head = CatAvgMaxMobileNetv4Head(
                num_features=self.model.num_features,
                num_classes=num_classes,
                head_hidden_size=self.model.head_hidden_size,
                drop_rate=conf.dropout,
                act_layer=type(self.model.bn1.act),
                pad_type="",
                norm_layer=nn.BatchNorm2d,
            )
        elif conf.global_pool in _timm_pools:
            # Let timm configure pooling & classifier in one go
            kwargs = dict(
                model_name=conf.model_name,
                pretrained=conf.pretrained,
                in_chans=in_chans,
                num_classes=num_classes,
                drop_rate=conf.dropout,
                drop_path_rate=conf.drop_path,
                global_pool=conf.global_pool,
            )
            if conf.model_name.startswith("efficientvit"):
                kwargs.pop("drop_path_rate")
            self.model = timm.create_model(**kwargs)  # type: ignore
        else:
            # Build model first, then attach a custom pooling module
            kwargs = dict(
                model_name=conf.model_name,
                pretrained=conf.pretrained,
                in_chans=in_chans,
                num_classes=num_classes,
                drop_rate=conf.dropout,
                drop_path_rate=conf.drop_path,
            )
            if conf.model_name.startswith("efficientvit"):
                kwargs.pop("drop_path_rate")
            self.model = timm.create_model(**kwargs)  # type: ignore

            if conf.global_pool == "lse":
                self.model.global_pool = LSEPool(beta=conf.lse_beta)
            elif conf.global_pool == "gem":
                self.model.global_pool = GeM()
            elif conf.global_pool == "attn":
                assert isinstance(self.model, timm.models.MobileNetV3)
                num_features = self.model.num_features
                self.model.global_pool = AttnPool(num_features)  # type: ignore
            else:
                raise RuntimeError(f"Unexpected global_pool {conf.global_pool}")

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.head is not None:
            x = self.model.forward_features(data["image"])  # type: ignore
            return self.head.forward(x)

        return self.model.forward(data["image"].contiguous())
