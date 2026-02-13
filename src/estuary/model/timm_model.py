import timm
import torch
import torch.nn as nn

from estuary.model.config import EstuaryConfig


class TimmModel(nn.Module):
    def __init__(self, conf: EstuaryConfig, num_classes: int) -> None:
        super().__init__()

        in_chans = conf.bands.num_channels()

        if conf.pretrained:
            assert in_chans == 3, conf.bands

        self.head = None
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

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.head is not None:
            x = self.model.forward_features(data["image"])  # type: ignore
            return self.head.forward(x)

        return self.model.forward(data["image"].contiguous())
