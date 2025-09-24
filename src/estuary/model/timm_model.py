import timm
import torch
import torch.nn as nn

from estuary.model.config import EstuaryConfig


class TimmModel(nn.Module):
    def __init__(self, conf: EstuaryConfig) -> None:
        assert conf.model_type == "timm"
        in_chans = conf.bands.num_channels()

        if conf.pretrained:
            assert in_chans == 3, conf.bands

        self.model = timm.create_model(
            conf.model_name,
            pretrained=conf.pretrained,
            in_chans=in_chans,
            num_classes=len(conf.classes),
            drop_rate=conf.dropout,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model.forward(data)
