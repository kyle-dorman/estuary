import timm
import torch
import torch.nn as nn

from estuary.low_quality.config import QualityConfig


class TimmModel(nn.Module):
    def __init__(self, conf: QualityConfig, num_classes: int) -> None:
        super().__init__()

        in_chans = conf.bands.num_channels()

        if conf.pretrained:
            assert in_chans == 3, conf.bands

        _timm_pools = {"avg", "max", "avgmax", "catavgmax"}
        assert conf.global_pool in _timm_pools

        self.head = None
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

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.head is not None:
            x = self.model.forward_features(data["image"])  # type: ignore
            return self.head.forward(x)

        return self.model.forward(data["image"].contiguous())
