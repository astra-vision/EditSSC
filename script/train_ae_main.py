import argparse
import os

from encoding.train_ae import Trainer
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args_cli = parser.parse_args()

    print(f"Loading config from: {args_cli.config}")
    cfg = OmegaConf.load(args_cli.config)

    config_name = os.path.splitext(os.path.basename(args_cli.config))[0]
    cfg.save_path = os.path.join(cfg.save_path, config_name)
    os.makedirs(cfg.save_path, exist_ok=True)
    print(f"Save path: {cfg.save_path}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
