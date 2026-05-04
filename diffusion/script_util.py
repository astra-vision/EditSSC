import os
import yaml
import torch
import torch.nn as nn
# from utils_c import dist_util

from diffusion.unet_stable_diffusion import UNetModel

from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion import gaussian_diffusion as gd


class StableDiffusionBEVWrapper(nn.Module):
    """
    Wrapper for UNetModel (Stable Diffusion UNet) adapted for triplane inputs in BEV format.
    The model extracts only the xy_part from triplanes, processes it, and recomposes at the end.

    This matches the pattern used in BEVUNetModel where only xy_part is processed.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Extract parameters from args with defaults
        image_size = getattr(args, 'image_size', 128)  # Default image size
        in_channels = getattr(args, 'geo_feat_channels', 16)  # Default from config
        model_channels = getattr(args, 'model_channels', 64)  # Default from config
        out_channels = getattr(args, 'geo_feat_channels', 16)  # Default from config
        num_res_blocks = getattr(args, 'num_res_blocks', 2)  # Default
        attention_resolutions = getattr(args, 'attention_resolutions', [4, 2, 1])  # Default
        dropout = getattr(args, 'dropout', 0.0)
        channel_mult = getattr(args, 'channel_mult', [1, 2, 4, 8])  # Default
        conv_resample = getattr(args, 'conv_resample', True)
        use_checkpoint = getattr(args, 'use_checkpoint', False)
        use_fp16 = getattr(args, 'use_fp16', False)
        num_heads = getattr(args, 'num_heads', -1)
        num_head_channels = getattr(args, 'num_head_channels', -1)
        num_heads_upsample = getattr(args, 'num_heads_upsample', -1)
        use_scale_shift_norm = getattr(args, 'use_scale_shift_norm', False)
        resblock_updown = getattr(args, 'resblock_updown', False)
        use_new_attention_order = getattr(args, 'use_new_attention_order', False)
        use_spatial_transformer = getattr(args, 'use_spatial_transformer', False)
        transformer_depth = getattr(args, 'transformer_depth', 1)
        context_dim = getattr(args, 'context_dim', None)
        n_embed = getattr(args, 'n_embed', None)
        legacy = getattr(args, 'legacy', True)
        num_classes = getattr(args, 'num_classes', None)
        training_sd_latent = getattr(args, 'training_sd_latent', False)
        condition_latent_image = getattr(args, 'condition_latent_image', False)
        # Create the UNetModel
        self.model = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=2,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
            training_sd_latent=training_sd_latent,
            condition_latent_image=condition_latent_image,
            args=args,
        )

    def forward(self, x, timesteps, H=128, W=128, D=16, y=None):
        """
        Forward pass matching TriplaneUNetModel interface.
        This is called by _WrappedModel in respace.py with signature: (x, ts, H, W, D, y, data)

        The UNetModel forward method now handles triplane decomposition internally.
        It extracts only the xy_part, processes it, and recomposes at the end.

        Args:
            x: Input tensor [B, C, H+W, W+D] (triplane in composed format)
            timesteps: Timestep tensor
            H, W, D: Spatial dimensions for triplane
            y: Conditional input (optional)
            data: Additional data dict (optional)
            context: Context for cross-attention (optional)

        Returns:
            Output tensor in same format as input (recomposed triplane)
        """
        # UNetModel forward now handles triplane decomposition internally
        # It extracts xy_part, processes it, and recomposes with original xz, yz
        return self.model(x, timesteps=timesteps, context=context, y=y, H=H, W=W, D=D)




def create_model_and_diffusion_from_args(args):
    """
    Create model and diffusion based on architecture type.
     Diffusion BEV UNet (use_stable_diffusion_bev=True)
       -> StableDiffusionBEVWrapper (Stable Diffusion UNet with xy_part extraction)
    """
    diffusion = create_gaussian_diffusion(args)
    print("=" * 80)
    print("Creating StableDiffusionBEVWrapper...")
    model = StableDiffusionBEVWrapper(args)
    
    
    

    return model, diffusion

def create_gaussian_diffusion(args):
    steps = args.steps
    predict_xstart = args.predict_xstart
    learn_sigma = args.learn_sigma
    timestep_respacing= args.timestep_respacing

    sigma_small=False
    noise_schedule="linear"  # schedule du noise ajouté
    use_kl=False
    rescale_timesteps=False
    rescale_learned_sigmas=False

    betas = gd.get_named_beta_schedule(noise_schedule, steps)  # les betas du forward process
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        args=args,
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
