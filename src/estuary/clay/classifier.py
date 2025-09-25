import torch
import torch.nn as nn
from claymodel.backbone import Transformer
from claymodel.model import Encoder

from estuary.model.config import EstuaryConfig


class ClayConvDecoder(nn.Module):
    """
    A lightweight CNN head that maps ViT patch embeddings to class logits.

    Args:
        conf:           # config
        num_classes:    # output categories
    Input:
        x: Tensor[B, L, encoder_dim]  # L = H_patches*W_patches
    Output:
        Tensor[B, num_classes]
    """

    def __init__(
        self,
        conf: EstuaryConfig,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(conf.encoder_dim, conf.decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(conf.decoder_dim),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(conf.decoder_depth):
            blocks += [
                nn.Conv2d(conf.decoder_dim, conf.decoder_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conf.decoder_dim),
                nn.ReLU(inplace=True),
            ]
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conf.decoder_dim, num_classes)
        self.dropout = nn.Dropout(p=conf.dropout)

        # ---------------------
        # Weight initialisation
        # ---------------------
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = x[:, 1:]  # [B, L, D]

        # Reshape embs for conv projection
        B, L, _ = embs.shape
        H = W = int(L**0.5)
        assert H * W == L
        # [B, D, H, W]
        embs = embs.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.proj(embs)  # [B, D', H, W]
        x = self.conv_blocks(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Kaiming‑normal for convs, constant‑one for norm layers, truncated normal for FC."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ClayTransformerDecoder(nn.Module):
    def __init__(
        self,
        conf: EstuaryConfig,
        num_classes: int,
    ):
        super().__init__()
        self.encoder_dim = conf.encoder_dim
        self.dim = conf.decoder_dim

        self.enc_to_dec = (
            nn.Linear(conf.encoder_dim, conf.decoder_dim, bias=False)
            if conf.encoder_dim != conf.decoder_dim
            else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, conf.decoder_dim) * 0.02)
        self.classifier = nn.Linear(conf.decoder_dim, num_classes)
        self.dropout = nn.Dropout(p=conf.dropout)
        self.transformer = Transformer(
            dim=conf.decoder_dim,
            depth=conf.decoder_depth,
            heads=conf.decoder_heads,
            dim_head=conf.decoder_dim_head,
            mlp_dim=int(conf.decoder_dim * conf.decoder_mlp_ratio),
            fused_attn=True,
        )

        self.apply(self._init_weights)

    def forward(
        self,
        x,  # B, L + 1, D
    ) -> torch.Tensor:
        embs = x[:, 1:]  # [B, L, D]
        B, _, _ = embs.shape
        # Change the embedding dimension from encoder to decoder
        x0 = self.enc_to_dec(x)  # => B, L, D'

        # Add cls tokens
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D')
        x1 = torch.cat((cls_token, x0), dim=1)  # (B, L + 1, D')

        # Transform cls token
        x2 = self.transformer(x1)

        cls_token = x2[:, 0]
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)

        return logits

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm | nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ClayClassifier(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: ClayTransformerDecoder | ClayConvDecoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, datacube: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        datacube: dict containing the following keys:
            - pixels: [B C H W]
            - time: [B 4] # week hour
            - latlon: [B 4] # lat lon
            - waves: [4]
            - gsd: [1]
        """
        if len(datacube["waves"].shape) == 2:
            datacube["waves"] = datacube["waves"][0]
        if len(datacube["gsd"].shape) == 2:
            datacube["gsd"] = datacube["gsd"][0]
        x, _, _, _ = self.encoder(datacube)
        x = self.decoder(x)

        return x
