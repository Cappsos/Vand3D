from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_to_feature_volume(tokens: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Convert patch tokens (B, L, C) → 3-D feature cube (B, C, D, H, W)."""
    B, L, C = tokens.shape
    D_p, H_p, W_p = grid_shape
    assert L == D_p * H_p * W_p, "Token length does not match patch grid dimensions"
    return tokens.view(B, D_p, H_p, W_p, C).permute(0, 4, 1, 2, 3).contiguous()


class FiLMUNet3D(nn.Module):
    """Lightweight 3-D U-Net decoder with a single FiLM gate for text conditioning."""

    def __init__(
        self,
        n_layers: int,
        in_token_dim: int,
        t_dim: int = 768,
        base_c: int = 64,
        deep_supervision: bool = True,
        voxel_size: tuple[int,int,int] | None = None
    ) -> None:
        super().__init__()

        self.deep_supervision = deep_supervision
        self.voxel_size = voxel_size

        # Layer-wise 1×1×1 projections (each ViT layer → base_c channels)
        self.layer_proj = nn.ModuleList(
            [nn.Conv3d(in_token_dim, base_c, kernel_size=1) for _ in range(n_layers)]
        )

        # FiLM text modulation at the bottleneck
        self.gamma = nn.Linear(t_dim, base_c * 8)
        self.beta = nn.Linear(t_dim, base_c * 8)

        # --- Encoder path (very shallow) ---
        self.down1 = nn.Sequential(
            nn.Conv3d(base_c * n_layers, base_c * 2, 3, padding=1), nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2), nn.Conv3d(base_c * 2, base_c * 4, 3, padding=1), nn.GELU()
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2), nn.Conv3d(base_c * 4, base_c * 8, 3, padding=1), nn.GELU()
        )

        # --- Decoder path ---
        self.up2 = nn.ConvTranspose3d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv3d(base_c * 8, base_c * 4, 3, padding=1), nn.GELU())

        self.up1 = nn.ConvTranspose3d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(base_c * 4, base_c * 2, 3, padding=1), nn.GELU())

        self.out_conv = nn.Conv3d(base_c * 2, 1, 1)

        if deep_supervision:
            self.aux2 = nn.Conv3d(base_c * 4, 1, 1)  
            self.aux3 = nn.Conv3d(base_c * 8, 1, 1)  

    def forward(
        self,
        token_list: List[torch.Tensor],
        txt: torch.Tensor,
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            token_list: list of L tensors with shape (B, P, C).
            txt: text embedding (B, t_dim).
            grid_shape: (
                D_p, H_p, W_p
            ) patch grid shape used to reshape tokens to volumes.
        Returns:
            full-res mask, half-scale aux mask, quarter-scale aux mask (aux may be None).
        """
        # 1) reshape each layer to volume and project to base_c
        vols = [proj(tokens_to_feature_volume(tk, grid_shape)) for tk, proj in zip(token_list, self.layer_proj)]
        x = torch.cat(vols, dim=1)  # (B, base_c * n_layers, D, H, W)

        # 2) encoder
        d1 = self.down1(x)  # (B, 2C, …)
        d2 = self.down2(d1)  # (B, 4C, …)
        d3 = self.down3(d2)  # (B, 8C, …)

        # FiLM modulation at bottleneck
        g, b = self.gamma(txt)[:, :, None, None, None], self.beta(txt)[:, :, None, None, None]
        d3 = d3 * (1 + g) + b

        # 3) decoder
        u2 = self.up2(d3)
        cat2 = torch.cat([u2, d2], dim=1)
        dec2 = self.dec2(cat2)

        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, d1], dim=1)
        dec1 = self.dec1(cat1)

        out_full = torch.sigmoid(self.out_conv(dec1))  # (B,1,D,H,W)
        if self.voxel_size is not None:
            out_full = F.interpolate(
                out_full, size=self.voxel_size, mode="trilinear", align_corners=False
            )
        
        
    
            
        return out_full
