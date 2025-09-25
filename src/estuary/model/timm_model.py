import timm
import torch
import torch.nn as nn

from estuary.model.config import EstuaryConfig, ModelType


class TimmModel(nn.Module):
    def __init__(self, conf: EstuaryConfig, num_classes: int) -> None:
        super().__init__()

        assert conf.model_type == ModelType.TIMM
        in_chans = conf.bands.num_channels()

        if conf.pretrained:
            assert in_chans == 3, conf.bands

        self.model = timm.create_model(
            conf.model_name,
            pretrained=conf.pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=conf.dropout,
            drop_path_rate=conf.drop_path,
        )

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.forward(data["image"])
