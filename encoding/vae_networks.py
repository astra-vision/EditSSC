import torch
import torch.nn as nn
from diffusers.models import VQModel



class SDVQVAEScene3D(nn.Module):
    """
    Adapts the Stable Diffusion VQ-VAE (diffusers.VQModel) for 3D LiDAR scenes.

    The Z (height) axis is folded into the channel dimension so the VQ-VAE
    processes a 3D voxel grid as a multi-channel 2D BEV image, then the SD
    decoder reconstructs it symmetrically:

        vol [B, H, W, Z]  →  embed     →  [B, C, H, W, Z]
                          →  fold Z    →  [B, C×Z, H, W]
                          →  VQ enc    →  latent  [B, L, H/f, W/f]
                          →  codebook  →  quant   [B, L, H/f, W/f]  +  vq_loss
                          →  VQ dec    →  [B, C×Z, H, W]
                          →  unfold Z  →  [B, C, H, W, Z]
                          →  class head (Conv3d 1×1×1)  →  [B, num_class, H, W, Z]

    Relevant args:
        sd_latent_channels   (int,  default 3)             : latent depth L
        sd_n_embed           (int,  default 8192)          : codebook size
        sd_vq_embed_dim      (int,  default None → =L)     : codebook vector dim
        sd_block_out_channels(list, default [128,256,512,512]): encoder widths
        sd_layers_per_block  (int,  default 2)             : ResNet layers/block
        sd_norm_num_groups   (int,  default 32)            : GroupNorm groups
                                                             (must divide block_out_channels[0])
    """

    def __init__(self, args):
        super().__init__()

        num_class = args.num_class
        H_grid, W_grid, Z_grid = args.grid_size
        self.Z = Z_grid
        self.num_class = num_class
        self.C = args.geo_feat_channels
        self.embedding = nn.Embedding(num_class, self.C)

        sd_latent_channels    = getattr(args, 'sd_latent_channels', 16)
        sd_n_embed            = getattr(args, 'sd_n_embed', 512)
        sd_vq_embed_dim       = getattr(args, 'sd_vq_embed_dim', None)
        sd_block_out_channels = tuple(getattr(args, 'sd_block_out_channels', [128, 256, 512]))
        sd_layers_per_block   = getattr(args, 'sd_layers_per_block', 2)
        sd_norm_num_groups    = getattr(args, 'sd_norm_num_groups', 32)

        in_channels_2d = self.C * Z_grid  # height folded into channels

        print(
            f'Building SDVQVAEScene3D: in_channels={in_channels_2d} (C={self.C} × Z={Z_grid}), '
            f'latent={sd_latent_channels}, codebook={sd_n_embed}, '
            f'blocks={sd_block_out_channels}'
        )

        self.vqvae = VQModel(
            in_channels=in_channels_2d,
            out_channels=in_channels_2d,
            down_block_types=("DownEncoderBlock2D",) * len(sd_block_out_channels),
            up_block_types=("UpDecoderBlock2D",) * len(sd_block_out_channels),
            block_out_channels=sd_block_out_channels,
            layers_per_block=sd_layers_per_block,
            latent_channels=sd_latent_channels,
            num_vq_embeddings=sd_n_embed,
            vq_embed_dim=sd_vq_embed_dim,
            norm_num_groups=sd_norm_num_groups,
        )

        # 1×1×1 conv to project C features → num_class logits at each voxel
        self.class_head = nn.Conv3d(self.C, num_class, kernel_size=1)

    # ------------------------------------------------------------------
    # Volume ↔ 2D helpers
    # ------------------------------------------------------------------

    def _vol_to_2d(self, vol: torch.Tensor) -> torch.Tensor:
        """vol [B, H, W, Z] (int labels)  →  [B, C*Z, H, W] (float features)"""
        x = vol.detach().clone()
        x[x == 255] = 0
        x = self.embedding(x)                           # [B, H, W, Z, C]
        
        x = x.permute(0, 4, 1, 2, 3)                   # [B, C, H, W, Z]
        B, C, H, W, Z = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()      # [B, C, Z, H, W]
        x = x.view(B, C * Z, H, W)                     # [B, C*Z, H, W]
        return x

    def _2d_to_vol(self, x_2d: torch.Tensor) -> torch.Tensor:
        """[B, C*Z, H, W]  →  [B, C, H, W, Z]"""
        B, _CZ, H, W = x_2d.shape
        x = x_2d.view(B, self.C, self.Z, H, W)         # [B, C, Z, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()      # [B, C, H, W, Z]
        return x

    # ------------------------------------------------------------------
    # Encode / quantize / decode / forward
    # ------------------------------------------------------------------

    def encode(self, vol: torch.Tensor) -> torch.Tensor:
        """vol [B, H, W, Z]  →  pre-quant latent [B, L, H/f, W/f]"""
        return self.vqvae.encode(self._vol_to_2d(vol)).latents

    def enc_and_quantize(self, vol: torch.Tensor):
        """vol [B, H, W, Z]  →  pre-quant latent [B, L, H/f, W/f]"""
        z = self.vqvae.encode(self._vol_to_2d(vol)).latents
        quant, vq_loss, (_, _, indices) = self.vqvae.quantize(z)
        return z, quant, vq_loss, indices

    def quantize(self, z: torch.Tensor):
        """
        Returns:
            quant   : [B, L, H/f, W/f]  nearest codebook entry
            vq_loss : commitment loss scalar
            indices : [B*H/f*W/f]  codebook indices
        """
        quant, vq_loss, (_, _, indices) = self.vqvae.quantize(z)
        return quant, vq_loss, indices

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """quant [B, L, H/f, W/f]  →  logits [B, num_class, H, W, Z]"""
        x_2d = self.vqvae.decode(quant).sample          # [B, C*Z, H, W]
        x_3d = self._2d_to_vol(x_2d)                    # [B, C, H, W, Z]
        return self.class_head(x_3d)                     # [B, num_class, H, W, Z]

    def forward(self, vol: torch.Tensor, return_vq_loss: bool = False):
        """
        Args:
            vol          : [B, H, W, Z]  integer class labels
            return_vq_loss: if True, also return the VQ commitment loss

        Returns:
            logits       : [B, num_class, H, W, Z]
            vq_loss (opt): scalar commitment loss
        """
        z = self.encode(vol)
        quant, vq_loss, _indices = self.quantize(z)
        logits = self.decode(quant)

        if return_vq_loss:
            return logits, vq_loss
        return logits

    def geo_parameters(self):
        return list(self.embedding.parameters()) + list(self.vqvae.parameters()) + list(self.class_head.parameters())

def create_autoencoder(args):
    """
    Factory function to create autoencoder based on architecture type.

    Args:
        args: Configuration object with architecture selection

    Returns:
        Autoencoder model instance
    """
    
    return SDVQVAEScene3D(args)
    
   