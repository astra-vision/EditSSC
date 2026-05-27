from argparse import Namespace
from omegaconf import OmegaConf
from dataset.tri_dataset_builder import TriplaneDataset
from diffusion.script_util import create_model_and_diffusion_from_args
from diffusion.resample import create_named_schedule_sampler
from diffusion.train_util import TrainLoop
from diffusion import logger
from utils import dist_util
from utils.utils import cycle
from torch.utils.data import DataLoader, Subset
import argparse
import os


def dict_to_namespace(d):
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    return d


def train_diffusion(args):
    log_dir = os.path.join(args.save_path, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)

    ds = TriplaneDataset(args, 'train')
    val_ds = TriplaneDataset(args, 'val')

    max_train_samples = getattr(args, 'max_train_samples', None)
    if max_train_samples is not None and max_train_samples > 0:
        print(f"Limiting training dataset to {max_train_samples} sample(s)")
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
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args_cli = parser.parse_args()

    print(f"Loading config from: {args_cli.config}")
    config_dict = OmegaConf.to_container(OmegaConf.load(args_cli.config), resolve=True)
    cfg = dict_to_namespace(config_dict)

    config_name = os.path.splitext(os.path.basename(args_cli.config))[0]
    cfg.save_path = os.path.join(cfg.save_path, config_name)
    os.makedirs(cfg.save_path, exist_ok=True)
    print(f"Save path: {cfg.save_path}")

    dist_util.setup_dist(cfg.gpu_id)
    train_diffusion(cfg)
