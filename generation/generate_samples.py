
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # ou "osmesa" si EGL ne marche pas

from utils.utils import save_remap_lut, point2voxel
from encoding.train_ae import get_pred_mask
from diffusion.triplane_util import build_sampling_model, decompose_featmaps
from utils.utils import save_remap_lut, point2voxel, load_config_from_yaml
from utils.render_utils import save_generated_scene, save_lidar_comparison
from encoding.ssc_metrics import SSCMetrics

import torch
import argparse
import numpy as np
from dataset.tri_dataset_builder import TriplaneDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

import yaml
from PIL import Image
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from pathlib import Path



def infer_from_samples(samples, coords, ae, args, shape_3d):
    if args.scale_triplanes:
        samples=samples*args.triplane_scale_factor
    xy_feat, xz_feat, yz_feat = decompose_featmaps(samples, shape_3d)
    model_output = ae.decode(xy_feat) 
    pred_prob = torch.softmax(model_output, dim=1)
    output = pred_prob.argmax(dim=1).float()  # [B, N]
    output_np = output.cpu().numpy()
    return output_np


def _compute_iou_miou(pred, gt, n_classes):
    """Per-sample binary IoU + semantic mIoU using SSCMetrics.one_stats.

    ``pred`` and ``gt`` can be float or int arrays of class ids in
    ``[0, n_classes)`` and any shape; they are flattened internally. Returns
    ``(iou_pct, miou_pct)`` (both in %).
    """
    pred_int = np.asarray(pred).astype(np.int64).reshape(-1)
    gt_int = np.asarray(gt).astype(np.int64).reshape(-1)
    # SSCMetrics indexes a (n_classes, n_classes) confusion matrix, so any
    # out-of-range label (e.g. 255 for 'invalid') would crash np.add.at.
    valid = (pred_int >= 0) & (pred_int < n_classes) & (gt_int >= 0) & (gt_int < n_classes)
    if not np.any(valid):
        return 0.0, 0.0
    pred_int = pred_int[valid]
    gt_int = gt_int[valid]
    metrics = SSCMetrics(n_classes=n_classes, ignore=[])
    iou, miou, _ = metrics.one_stats(pred_int, gt_int)
    return float(iou), float(miou)


def sample(args):
    args.batch_size_eval=1
    args.batch_size=1

    ds = TriplaneDataset(args, 'train')
    val_ds = TriplaneDataset(args, 'val')

    collate_fn = None

    dl = DataLoader(ds, batch_size = args.batch_size_eval, shuffle = False, pin_memory = False, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size = args.batch_size_eval, shuffle = False,num_workers=10, pin_memory = True, collate_fn=collate_fn)
   
    
    if hasattr(args, 'valdl') and args.valdl:
        datal = val_dl
        dataset_name = "validationset"
    else :
        datal = dl
        dataset_name = "trainset" 

    model, ae, sample_fn, coords, query, out_shape, _, learning_map_inv, H, W, D, grid_size, _, args,_ = build_sampling_model(args)
    args.grid_size = grid_size

    vis_enabled = bool(getattr(args, 'vis', False))
    load_lidar = bool(getattr(args, 'load_lidar', False))
    learning_map_inv_vis = None
    color_map_vis = None
    vis_dir = None
    if vis_enabled:
        with open(args.yaml_path, 'r') as stream:
            scene_yaml = yaml.safe_load(stream)
        learning_map_inv_vis = scene_yaml.get('learning_map_inv', None)
        color_map_vis = scene_yaml.get('color_map', {})
        vis_dir = os.path.join(args.save_path, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        if load_lidar:
            print(f"[vis] Visualization enabled (lidar + recon + gt). Saving PNGs to: {vis_dir}")
        else:
            print(f"[vis] Visualization enabled. Saving PNGs to: {vis_dir}")

    with torch.no_grad():
        failed_files = []
        total_files = 0
        for triplane, cond, data in tqdm(datal):
            total_files += 1
            file_name = "unknown"
            try:
                cond['y']=cond['y'].cuda()
                if data is not None:
                    cond['data'] = data
                num = os.path.splitext(os.path.basename(cond['path'][0]))[0]
                file_name = num
                save_path=os.path.join(args.save_path,f"{num}.npy")

                if vis_enabled:
                    vis_filename = f"{num}_comparison.png" if load_lidar else f"{num}_generated.png"
                    vis_path = os.path.join(vis_dir, vis_filename)
                else:
                    vis_path = None

                npy_done = os.path.exists(save_path)
                vis_done = vis_path is not None and os.path.exists(vis_path)

                if npy_done and (not vis_enabled or vis_done):
                    print(f"Skipping {save_path}, already exists.")
                    continue

                if npy_done:
                    # .npy already there but visualization is missing -> reload
                    output_np = np.load(save_path)
                else:
                    samples = sample_fn(model, out_shape, progress=False, model_kwargs=cond,clip_denoised=False)
                    output_np=infer_from_samples(samples, coords, ae, args, (H,W,D))
                    np.save(save_path, output_np)

                if vis_enabled and not vis_done:
                    scene_gen = output_np[0] if output_np.ndim == 4 else output_np

                    if load_lidar and data is not None and 'occupancy' in data:
                        # 3-panel comparison: LiDAR (input) | recon | GT
                        output_np_gt = infer_from_samples(
                            triplane.cuda(), coords, ae, args, (H, W, D)
                        )
                        scene_gt = output_np_gt[0] if output_np_gt.ndim == 4 else output_np_gt

                        occupancy = data['occupancy']
                        if isinstance(occupancy, torch.Tensor):
                            occupancy = occupancy.cpu().numpy()
                        if occupancy.ndim == 4:
                            occupancy = occupancy[0]

                        iou, miou = _compute_iou_miou(scene_gen, scene_gt, args.num_class)
                        print(f"[vis] {num}: IoU={iou:.2f}% | mIoU={miou:.2f}%")

                        save_lidar_comparison(
                            lidar_scene=occupancy,
                            generated_scene=scene_gen,
                            gt_scene=scene_gt,
                            sample_id=num,
                            learning_map_inv=learning_map_inv_vis,
                            color_map=color_map_vis,
                            folder_path=vis_dir,
                            original_shape=tuple(grid_size),
                            metrics={'iou': iou, 'miou': miou},
                        )
                    else:
                        save_generated_scene(
                            scene_gen,
                            sample_id=num,
                            learning_map_inv=learning_map_inv_vis,
                            color_map=color_map_vis,
                            folder_path=vis_dir,
                            original_shape=tuple(grid_size),
                            depth_color=False,
                        )

                # Free memory between iterations.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception as e:
                print(f"ERROR: Failed to generate file {file_name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                failed_files.append(file_name)

        if failed_files:
            successful = total_files - len(failed_files)
            print(f"\nWARNING: Failed to generate {len(failed_files)}/{total_files} files:")
            for f in failed_files[:20]:  # Print first 20 failed files
                print(f"  - {f}")
            if len(failed_files) > 20:
                print(f"  ... and {len(failed_files) - 20} more")
            print(f"Successfully generated: {successful}/{total_files} files")







if __name__ == '__main__':
    import sys
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Optional path to checkpoint')
    parser.add_argument(
        '--vis',
        action='store_true',
        help='Also render and save the generated scene as a PNG (only the '
             'generated voxel scene, no GT or conditioning).',
    )
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    with open(cfg.yaml_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)




    config_name = os.path.splitext(os.path.basename(args_cli.config))[0]
    stem = Path(cfg.diff_path).stem  # "ema_0.9999_145000"

    # Extract the number at the end
    print("stem",stem)
    num = int(stem.split('_')[-1]) // 1000

    # Determine save folder based on dataset type (4 options)

    if hasattr(cfg, 'valdl') and cfg.valdl:
        cfg.save_path = os.path.join(cfg.save_path, f"validation_{num}", config_name)
    else :
        cfg.save_path = os.path.join(cfg.save_path, f"training_{num}", config_name)
    
    os.makedirs(cfg.save_path, exist_ok=True)
    print(f"Saving to: {cfg.save_path}")

    if args_cli.vis:
        cfg.vis = True

    sample(cfg)
