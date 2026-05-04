import yaml
from argparse import Namespace
from utils.parser_util import add_diffusion_training_options, add_encoding_training_options


def dict_to_namespace(d):
    """Convert a nested dictionary to a Namespace object (supports dot notation)."""
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d
from dataset.tri_dataset_builder import TriplaneDataset
from diffusion.script_util import create_model_and_diffusion_from_args
from diffusion.resample import create_named_schedule_sampler
from diffusion.train_util import TrainLoop
from diffusion import logger
from utils import dist_util
from dataset.path_manager import *
from utils.utils import cycle
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch
import argparse
import os

def train_diffusion(args):
    """
    Train with basic train_util.py (original training loop).
    """
    # Save checkpoints in a subdirectory to keep the experiment folder organized
    log_dir = os.path.join(args.save_path, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)

    ds = TriplaneDataset(args, 'train')
    val_ds = TriplaneDataset(args, 'val')

    # Limit dataset to one sample for sanity check if specified
    max_train_samples = getattr(args, 'max_train_samples', None)
    if max_train_samples is not None and max_train_samples > 0:
        print(f"⚠️  SANITY CHECK MODE: Limiting training dataset to {max_train_samples} sample(s)")
        ds = Subset(ds, list(range(min(max_train_samples, len(ds)))))

    collate_fn = None

    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = True, pin_memory = True, collate_fn=collate_fn)
    dl = cycle(dl)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False, pin_memory = True, collate_fn=collate_fn)
    val_dl = cycle(val_dl)

    model, diffusion = create_model_and_diffusion_from_args(args)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    TrainLoop(
        # diffusion_net = args.diff_net_type,
        triplane_loss_type = args.triplane_loss_type,
        timestep_respacing = args.timestep_respacing,
        training_step = args.steps,
        model=model,
        diffusion=diffusion,
        data=dl,
        val_data=val_dl,
        ssc_refine = args.ssc_refine,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=args.diff_lr,
        ema_rate=args.ema_rate,
        use_ema=getattr(args, 'use_ema', True),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.diff_n_iters,
        use_cosine_lr=getattr(args, 'use_cosine_lr', False),
        use_constant_lr=getattr(args, 'use_constant_lr', False),
        warmup_steps=getattr(args, 'warmup_steps', 0),
        grad_clip=getattr(args, 'grad_clip', 0.0),
        args=args,
    ).run_loop()



if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path

    # Two modes of operation:
    # Mode 1: Original mode with just --config (for backward compatibility)
    # Mode 2: New mode with --config, --work-dir, and --cfg-options (for automation)

    parser = argparse.ArgumentParser(
        description='Train diffusion model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original usage (backward compatible)
  python train_diffusion_main.py --config path/to/config.yaml

  # New usage with overrides (for automation)
  python train_diffusion_main.py \\
      --config common_diffusion_base.yaml \\
      --work-dir experiments/exp_001 \\
      --cfg-options "batch_size=10 diff_lr=0.001"
        """
    )

    parser.add_argument('--config', type=str, required=True, help='Path to base config YAML')
    parser.add_argument('--work-dir', type=str, default=None,
                       help='Working directory for outputs (optional, enables override mode)')
    parser.add_argument('--cfg-options', type=str, default="",
                       help='Config overrides in format "key1=value1 key2=value2"')
    parser.add_argument('--use-lightning', action='store_true',
                       help='Use PyTorch Lightning training (stable-diffusion style) instead of basic train_util.py')

    args_cli = parser.parse_args()

    # Check which mode to use
    if args_cli.work_dir is not None:
        # NEW MODE: Use config_loader for overrides
        print("\n" + "="*60)
        print("🔧 Loading Configuration with Overrides")
        print("="*60)

        # Import config_loader
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from exp_final.configs.diffusion.config_loader import (  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!A corriger!!!!!!!!!!!!
            load_yaml_config, override_config, save_config
        )

        # Load base config
        config_dict = load_yaml_config(args_cli.config)

        # Apply overrides if provided
        if args_cli.cfg_options:
            print("\n📝 Applying overrides:")
            config_dict = override_config(config_dict, args_cli.cfg_options)

        # Update save_path to work_dir
        config_dict['save_path'] = args_cli.work_dir

        # Create work directory
        os.makedirs(args_cli.work_dir, exist_ok=True)
        print(f"\n✓ Work directory: {args_cli.work_dir}")

        # Save final config for reproducibility
        config_save_path = os.path.join(args_cli.work_dir, "config_used.yaml")
        save_config(config_dict, config_save_path)

        # Convert dict to Namespace for training (supports dot notation like args.batch_size)
        cfg = dict_to_namespace(config_dict)

        print("="*60)
        print("✅ Configuration ready!\n")

    else:
        # ORIGINAL MODE: Direct YAML loading (backward compatible)
        print(f"Loading config from: {args_cli.config}")
        with open(args_cli.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        cfg = dict_to_namespace(config_dict)

        # Extract config name (without .yaml) and create save path
        config_name = os.path.splitext(os.path.basename(args_cli.config))[0]
        cfg.save_path = os.path.join(cfg.save_path, config_name)
        os.makedirs(cfg.save_path, exist_ok=True)

        print(f"Save path: {cfg.save_path}")

    # Setup and run training
    dist_util.setup_dist(cfg.gpu_id)
    train_diffusion(cfg)
