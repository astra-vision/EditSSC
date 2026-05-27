import torch
import numpy as np
import argparse
import os
import traceback
from encoding.vae_networks import create_autoencoder
from diffusion.triplane_util import compose_featmaps
from tqdm.auto import tqdm
from dataset.kitti_dataset import SemKITTI
from pathlib import Path
from omegaconf import OmegaConf




@torch.no_grad()
def save(args):
    output_root = getattr(args, "triplane_save_path", None) or args.save_path

    if args.dataset == 'kitti':
        dataset = SemKITTI(args, 'train', get_query=True, folder=args.data_name)
        val_dataset = SemKITTI(args, 'val', get_query=True, folder=args.data_name)
        tri_size = args.tri_size

    
    num_workers = 0
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=None if num_workers == 0 else 2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=None if num_workers == 0 else 2
    )

    checkpoint_path = args.resume
    if checkpoint_path is None:
        raise ValueError("No checkpoint provided. Set `resume` in config or pass --resume_checkpoint.")

    print(f'The number of voxel labels is {len(dataset)}.')
    print(f'Load autoencoder model from "{checkpoint_path}"')

    model = create_autoencoder(args)
    model = model.cuda()
    print("resume", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for loader in [val_dataloader, dataloader]:
        try:
            for data in tqdm(loader):
                vox = data["voxel_label"].type(torch.LongTensor).cuda()
                path = data["path"]

                z, quant, vq_loss, indices = model.enc_and_quantize(vox)
                H_tri, W_tri, D_tri = tri_size
                feat_xy = quant
                feat_xz = torch.randn(z.shape[0], z.shape[1], H_tri, D_tri, device=z.device, dtype=z.dtype)
                feat_yz = torch.randn(z.shape[0], z.shape[1], W_tri, D_tri, device=z.device, dtype=z.dtype)
                triplane, _ = compose_featmaps(
                    feat_xy.squeeze(0), feat_xz.squeeze(0), feat_yz.squeeze(0), tri_size
                )

                file_idx = str(Path(path[0]).stem.split('_')[0])
                folder_idx = str(Path(path[0]).parent.parent.stem)
                save_folder_path = os.path.join(output_root, folder_idx, args.save_name)
                os.makedirs(save_folder_path, exist_ok=True)
                if os.path.exists(os.path.join(save_folder_path, file_idx +args.save_tail)):
                    print(f"File already exists: {os.path.join(save_folder_path, file_idx +args.save_tail)}")
                    continue

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


                print(f"Saving to {os.path.join(save_folder_path, file_idx +args.save_tail)}")
                triplane_cpu = triplane.cpu().numpy()
                del triplane
                np.save(os.path.join(save_folder_path, file_idx +args.save_tail), triplane_cpu)

        except RuntimeError as e:
            if "DataLoader worker" in str(e) and "killed" in str(e):
                print(f"\n⚠️  ERROR: DataLoader worker was killed (likely out of memory).")
                print(f"   This usually happens when num_workers > 0 and the system runs out of memory.")
                print(f"   Solution: Set num_workers=0 in your config or reduce memory usage.")
                print(f"   Full error: {e}")
                raise
            else:
                raise
        except Exception as e:
            print(f"\n❌ Unexpected error in DataLoader: {e}")
            traceback.print_exc()
            raise
     


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Optional path to checkpoint')
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    if args_cli.resume_checkpoint is not None:
        cfg.resume = args_cli.resume_checkpoint
    output_root = getattr(cfg, "triplane_save_path", None) or cfg.save_path
    os.makedirs(output_root, exist_ok=True)


    save(cfg)



if __name__ == '__main__':
    main()


