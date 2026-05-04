import argparse
from encoding_c.train_ae import Trainer
from dataset_c.path_manager import *
from omegaconf import OmegaConf
import sys
import os


def main():
    # Two modes of operation:
    # Mode 1: Original mode with just --config (for backward compatibility)
    # Mode 2: New mode with --config, --work-dir, and --cfg-options (for automation)

    parser = argparse.ArgumentParser(
        description='Train autoencoder model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original usage (backward compatible)
  python train_ae_main.py --config path/to/config.yaml

  # New usage with overrides (for automation)
  python train_ae_main.py \\
      --config common_ae_base.yaml \\
      --work-dir experiments/exp_001 \\
      --cfg-options "bs=8 lr=0.0001"
        """
    )

    parser.add_argument('--config', type=str, required=True, help='Path to base config YAML')
    parser.add_argument('--work-dir', type=str, default=None,
                       help='Working directory for outputs (optional, enables override mode)')
    parser.add_argument('--cfg-options', type=str, default="",
                       help='Config overrides in format "key1=value1 key2=value2"')

    args_cli = parser.parse_args()

    # Debug: Print parsed arguments
    print(f"DEBUG: work_dir = {args_cli.work_dir}")
    print(f"DEBUG: cfg_options = {args_cli.cfg_options}")
    print(f"DEBUG: config = {args_cli.config}")

    # Check which mode to use
    if args_cli.work_dir is not None:
        # NEW MODE: Use config_loader for overrides
        print("\n" + "="*60)
        print("🔧 Loading Configuration with Overrides")
        print("="*60)

        # Import config_loader
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from exp_final.configs.diffusion.config_loader import (
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

        # Convert to OmegaConf for training
        cfg = OmegaConf.create(config_dict)

        print("="*60)
        print("✅ Configuration ready!\n")

    else:
        # ORIGINAL MODE: Direct YAML loading (backward compatible)
        print(f"Loading config from: {args_cli.config}")
        cfg = OmegaConf.load(args_cli.config)

        # Extract config name (without .yaml) and create save path
        config_name = os.path.splitext(os.path.basename(args_cli.config))[0]
        cfg.save_path = os.path.join(cfg.save_path, config_name)
        os.makedirs(cfg.save_path, exist_ok=True)

        print(f"Save path: {cfg.save_path}")

    # Setup and run training
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
