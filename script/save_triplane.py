import torch
import numpy as np
import argparse
from encoding.networks import create_autoencoder
from diffusion.triplane_util import compose_featmaps, decompose_featmaps
from tqdm.auto import tqdm
import os
from dataset.kitti_dataset import SemKITTI
from dataset.carla_dataset import CarlaDataset
from dataset.path_manager import *
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from utils.utils import make_query

import torch.nn.functional as F
from omegaconf import OmegaConf
import gc


def apply_vox_transformations(vox):
    """
    Apply 6 physically meaningful transformations to a 3D voxel scene (PyTorch version).
    
    Designed for 3D driving scenes where:
    - X-axis: left-right (lateral)
    - Y-axis: front-back (longitudinal)
    - Z-axis: up-down (height)
    
    Args:
        vox: Tensor of shape [batch_size, X, Y, Z] representing a 3D scene
        
    Returns:
        List of 6 transformed voxel tensors:
        [h_flip, rot_90, rot_180, rot_270, h_flip+rot_90, h_flip+rot_270]
    """
    transformations = []
    
    # 1. Horizontal flip (mirror left-right, like opposite side of road)
    vox_h_flip = torch.flip(vox, dims=[1])
    transformations.append(vox_h_flip)
    
    # 2. Rotation 90° counterclockwise around Z-axis (vertical)
    vox_rot_90 = torch.rot90(vox, k=1, dims=[1, 2])
    transformations.append(vox_rot_90)
    
    # 3. Rotation 180° around Z-axis
    vox_rot_180 = torch.rot90(vox, k=2, dims=[1, 2])
    transformations.append(vox_rot_180)
    
    # 4. Rotation 270° counterclockwise / 90° clockwise around Z-axis
    vox_rot_270 = torch.rot90(vox, k=3, dims=[1, 2])
    transformations.append(vox_rot_270)
    
    # 5. Horizontal flip + 90° rotation
    vox_h_flip_rot_90 = torch.rot90(vox_h_flip, k=1, dims=[1, 2])
    transformations.append(vox_h_flip_rot_90)
    
    # 6. Horizontal flip + 270° rotation
    vox_h_flip_rot_270 = torch.rot90(vox_h_flip, k=3, dims=[1, 2])
    transformations.append(vox_h_flip_rot_270)
    
    return transformations


@torch.no_grad()
def save(args):
    if args.dataset == 'kitti':
        dataset = SemKITTI(args, 'train', get_query=True, folder=args.data_name)
        val_dataset = SemKITTI(args, 'val', get_query=True, folder=args.data_name)
        
        tri_size=args.tri_size

    
    num_workers = 0  # Always use 0 for inference to avoid OOm
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,  
        prefetch_factor=None if num_workers == 0 else 2  # No prefetching for single-threaded
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,  # Disable pin_memory to reduce memory pressure
        prefetch_factor=None if num_workers == 0 else 2  # No prefetching for single-threaded
    )

    print(f'The number of voxel labels is {len(dataset)}.')
    print(f'Load autoencoder model from "{args.resume}"')

    if args.semantic_training_or_generation :
        model = create_autoencoder(args)
        model = model.cuda()
        print("resume",args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    


    coord,query=make_query(tuple(args.grid_size))
    coord=coord.type(torch.LongTensor).cuda()
    query=query.type(torch.FloatTensor).cuda()
    for loader in [val_dataloader, dataloader]: # val_dataloader,
        try:
            for data in tqdm(loader):
                # Extract needed data immediately and delete large unused items from data dict
                vox = data["voxel_label"].type(torch.LongTensor).cuda()
                invalid = data["invalid"].type(torch.LongTensor).cuda()
                path = data["path"]
                
       
                z, quant, vq_loss, indices = model.enc_and_quantize(vox)
                H_tri, W_tri, D_tri = tri_size
                feat_xy = quant
                feat_xz = torch.randn(z.shape[0], z.shape[1], H_tri, D_tri, device=z.device, dtype=z.dtype)
                feat_yz = torch.randn(z.shape[0], z.shape[1], W_tri, D_tri, device=z.device, dtype=z.dtype)
                triplane, _ = compose_featmaps(
                    feat_xy.squeeze(0), feat_xz.squeeze(0), feat_yz.squeeze(0), tri_size
                )
                

                # break
                file_idx = str(Path(path[0]).stem.split('_')[0])  # e.g., 002165
                folder_idx = str(Path(path[0]).parent.parent.stem)  # e.g., 00
                save_folder_path = os.path.join(args.save_path, folder_idx, args.save_name)  # e.g., /home/sebin/dataset/sequence/00/tri_1enc_1dec_0pad
                os.makedirs(save_folder_path, exist_ok=True)
                if os.path.exists(os.path.join(save_folder_path, file_idx +args.save_tail)):
                    print(f"File already exists: {os.path.join(save_folder_path, file_idx +args.save_tail)}")
                    continue

                # Free intermediate tensors (keep vox for other branches)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


                print(f"Saving to {os.path.join(save_folder_path, file_idx +args.save_tail)}")
                # Move to CPU and convert to numpy before saving to free GPU memory immediately
                triplane_cpu = triplane.cpu().numpy()
                del triplane  # Explicitly delete GPU tensor
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
            import traceback
            traceback.print_exc()
            raise
     


def main():
    import sys
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    #parser.add_argument('--resume_checkpoint', type=str, default=None, help='Optional path to checkpoint')
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    os.makedirs(cfg.save_path, exist_ok=True)


    save(cfg)



if __name__ == '__main__':
    main()


