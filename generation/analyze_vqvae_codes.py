"""
Analyze VQ-VAE codebook entry ↔ semantic class correspondences.

For each code, measure which semantic classes co-occur most often at matching
BEV locations (majority vote over ground-truth BEV labels).

Outputs (written next to the AE checkpoint directory):
  vqvae_codes_analysis.txt    – human-readable report
  vqvae_codes_99_coverage.json – ``codes`` (~99% coverage) plus ``recommended_code`` /
                                 ``recommended_purity`` (max purity among codes in that set;
                                 used by training_free_gen ``--signature-code-pick recommended``).

Usage:
    python generation/analyze_vqvae_codes.py \\
        --config configs/common_ae_base.yaml \\
        [--num_samples 10000] \\
        [--split train]
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataset.kitti_dataset import SemKITTI
from encoding.vae_networks import create_autoencoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]

# BEV projection priority (higher priority wins along Z).
# Small objects (poles, trunks) beat vegetation, which beats ground.
PRIORITY_ORDER = [0, 15, 17, 12, 9, 10, 11, 14, 13, 19, 18, 16, 5, 4, 3, 2, 1, 8, 7, 6]

COVERAGE_THRESHOLD = 0.99  # cumulative coverage target when selecting codes per class


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class _Tee:
    """Write to stdout and a file at the same time."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _bev_priority(voxel_label: torch.Tensor, priority_order: list) -> torch.Tensor:
    """Collapse [B, H, W, D] voxel labels to [B, H, W] BEV via priority voting."""
    B, H, W, D = voxel_label.shape
    device = voxel_label.device

    p_map = torch.tensor(
        [priority_order.index(c) for c in range(20)], device=device
    )
    bev = torch.zeros((B, H, W), dtype=torch.long, device=device)
    bev_prio = torch.full((B, H, W), -1, dtype=torch.long, device=device)

    for z in range(D):
        sl = voxel_label[:, :, :, z]
        valid = sl < 20
        prio = p_map[sl.clamp(0, 19)]
        update = (prio > bev_prio) & valid & (sl != 0)
        bev[update] = sl[update]
        bev_prio[update] = prio[update]

    return bev


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_codes(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    max_samples = getattr(cfg, 'num_samples', 10000)
    imageset    = getattr(cfg, 'imageset', 'train')

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"Building Dataset ({imageset})...")
    dataset = SemKITTI(cfg, imageset=imageset, get_query=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=getattr(cfg, 'num_workers', 4),
    )
    num_class = cfg.num_class

    # ── Model ─────────────────────────────────────────────────────────────
    print("Initializing model...")
    model = create_autoencoder(cfg).to(device)
    model.eval()

    if not getattr(cfg, 'resume', None):
        print("ERROR: 'resume' (checkpoint path) must be set in the config.")
        return
    print(f"Loading weights from {cfg.resume}")
    ckpt = torch.load(cfg.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

    # Codebook size
    n_codes   = getattr(cfg, 'sd_n_embed', getattr(cfg, 'vq_num_embeddings', 512))
    latent_h  = cfg.tri_size[0]
    latent_w  = cfg.tri_size[1]
    scale_h   = cfg.grid_size[0] // latent_h
    scale_w   = cfg.grid_size[1] // latent_w

    print(f"Codebook size : {n_codes}")
    print(f"Grid {cfg.grid_size}  →  Latent {cfg.tri_size}  (scale {scale_h}×{scale_w})")

    # ── Output files ───────────────────────────────────────────────────────
    save_dir = os.path.dirname(cfg.resume)
    txt_file  = os.path.join(save_dir, 'vqvae_codes_analysis.txt')
    json_file = os.path.join(save_dir, 'vqvae_codes_99_coverage.json')
    print(f"Results will be saved to : {txt_file}")

    f_out           = open(txt_file, 'w')
    original_stdout = sys.stdout
    sys.stdout      = _Tee(sys.stdout, f_out)

    # ── Accumulate code ↔ class counts ─────────────────────────────────────
    code_class_counts = np.zeros((n_codes, num_class), dtype=np.int64)
    current_samples   = 0

    print("\nAnalyzing samples...")
    max_batches = (max_samples + cfg.bs - 1) // cfg.bs

    with torch.no_grad():
        for data in tqdm(dataloader, total=min(len(dataloader), max_batches)):
            if current_samples >= max_samples:
                break
            current_samples += data['voxel_label'].shape[0]

            voxel_label = data['voxel_label'].long().to(device)
            while voxel_label.dim() > 4:
                voxel_label = voxel_label.squeeze(-1)   # [B, H, W, D]

            # Encode → codebook indices
            _, _, _, indices = model.enc_and_quantize(voxel_label)
            indices = indices.view(-1, latent_h, latent_w)  # [B, Lh, Lw]
            B = indices.shape[0]

            # Priority-based BEV [B, H_gt, W_gt]
            bev = _bev_priority(voxel_label, PRIORITY_ORDER)

            # Downsample BEV to latent resolution (majority vote)
            bev_blocks = bev.view(B, latent_h, scale_h, latent_w, scale_w)
            bev_blocks = bev_blocks.permute(0, 1, 3, 2, 4).flatten(start_dim=3)
            bev_majority, _ = bev_blocks.mode(dim=-1)   # [B, Lh, Lw]

            idx_flat = indices.flatten().cpu().numpy()
            lbl_flat = bev_majority.flatten().cpu().numpy()

            for code in np.unique(idx_flat):
                mask   = idx_flat == code
                labels = lbl_flat[mask]
                valid  = labels[labels != 255]
                counts = np.bincount(valid, minlength=num_class)[:num_class]
                code_class_counts[code] += counts

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    code_usage          = code_class_counts.sum(axis=1)
    active_codes        = np.where(code_usage > 0)[0]
    print(f"Active codes : {len(active_codes)} / {n_codes}")

    sorted_codes = np.argsort(code_usage)[::-1]
    print("\nTop 20 most-used codes (dominant semantic classes, bg excluded) :")
    for code in sorted_codes[:20]:
        total = code_usage[code]
        if total == 0:
            continue
        counts_no_bg     = code_class_counts[code].copy()
        counts_no_bg[0]  = 0
        total_no_bg      = counts_no_bg.sum()
        if total_no_bg > 0:
            top = np.argsort(counts_no_bg)[::-1][:3]
            cls_str = " | ".join(
                f"{CLASS_NAMES[c]}: {counts_no_bg[c]/total_no_bg*100:.1f}%"
                for c in top if counts_no_bg[c] > 0
            )
            bg_pct = code_class_counts[code][0] / total * 100
            print(f"  Code {code:4d} (n={total:8d}, bg={bg_pct:5.1f}%) : {cls_str}")
        else:
            print(f"  Code {code:4d} (n={total:8d}) : 100 % background")

    # ── Class → codes (coverage threshold) ───────────────────────────────
    json_mapping = {}

    print(f"\nClass → codes ({COVERAGE_THRESHOLD*100:.0f}% cumulative coverage) :")
    for class_id in range(1, num_class):
        class_name       = CLASS_NAMES[class_id].upper()
        total_class_vol  = code_class_counts[:, class_id].sum()
        if total_class_vol == 0:
            continue

        print(f"\n  {class_name} (total volume : {total_class_vol}) :")

        candidates = []
        for code in active_codes:
            counts      = code_class_counts[code]
            total_valid = counts[1:].sum()
            if total_valid == 0 or counts[class_id] == 0:
                continue
            candidates.append({
                'code'     : int(code),
                'purity'   : counts[class_id] / total_valid,
                'class_vol': counts[class_id],
            })

        if not candidates:
            print(f"    No codes found.")
            continue

        candidates.sort(key=lambda x: x['class_vol'], reverse=True)

        cumulative   = 0
        selected     = []
        for item in candidates:
            selected.append(item['code'])
            cumulative += item['class_vol']
            coverage    = cumulative / total_class_vol
            print(
                f"    Code {item['code']:4d}: {item['class_vol']:7d} px "
                f"(coverage {coverage*100:5.2f}%, purity {item['purity']*100:5.1f}%)"
            )
            if coverage >= COVERAGE_THRESHOLD:
                break

        entry = {
            'codes'         : selected,
            'final_coverage': float(cumulative / total_class_vol),
        }
        # Same criterion as reading the .txt “purity” column: best code for conditioning
        # among those kept for coverage (not necessarily codes[0] by volume).
        sel_set = set(selected)
        sel_cand = [c for c in candidates if c['code'] in sel_set]
        if sel_cand:
            best = max(sel_cand, key=lambda x: x['purity'])
            entry['recommended_code'] = best['code']
            entry['recommended_purity'] = round(float(best['purity']), 6)
            print(
                f"    → recommended_code (max purity in coverage set): {best['code']:4d} "
                f"(purity {best['purity']*100:5.1f}%)"
            )
        json_mapping[class_name] = entry

    # Class 0 — empty / background
    total_bg_vol = code_class_counts[:, 0].sum()
    bg_candidates = [
        {'code': int(c), 'purity': code_class_counts[c][0] / code_class_counts[c].sum(),
         'bg_vol': code_class_counts[c][0]}
        for c in active_codes
        if code_class_counts[c].sum() > 0
        and code_class_counts[c][0] / code_class_counts[c].sum() >= 0.95
    ]
    if bg_candidates:
        bg_candidates.sort(key=lambda x: x['bg_vol'], reverse=True)
        print("\n  EMPTY SPACE (class 0, purity ≥ 95 %) :")
        cumulative = 0
        selected   = []
        for item in bg_candidates:
            selected.append(item['code'])
            cumulative += item['bg_vol']
            coverage    = cumulative / total_bg_vol
            print(
                f"    Code {item['code']:4d}: {item['bg_vol']:8d} px "
                f"(coverage {coverage*100:5.2f}%, purity {item['purity']*100:.2f}%)"
            )
            if coverage >= COVERAGE_THRESHOLD:
                break
        es_entry = {
            'codes'         : selected,
            'final_coverage': float(cumulative / total_bg_vol),
        }
        sel_bg = [x for x in bg_candidates if x['code'] in set(selected)]
        if sel_bg:
            best = max(sel_bg, key=lambda x: x['purity'])
            es_entry['recommended_code'] = best['code']
            es_entry['recommended_purity'] = round(float(best['purity']), 6)
            print(
                f"    → recommended_code (max bg purity in coverage set): {best['code']:4d} "
                f"(purity {best['purity']*100:5.2f}%)"
            )
        json_mapping["EMPTY_SPACE"] = es_entry

    # ── Write JSON ─────────────────────────────────────────────────────────
    sys.stdout = original_stdout
    f_out.close()

    with open(json_file, 'w') as f:
        json.dump(json_mapping, f, indent=4)

    print(f"\nAnalysis complete on {current_samples} samples.")
    print(f"  Text report : {txt_file}")
    print(f"  JSON mapping: {json_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze VQ-VAE code ↔ semantic class correspondences."
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the AE config YAML (e.g. configs/common_ae_base.yaml)')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Maximum number of samples to analyze (default: 10000)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to use (default: train)')
    args_cli = parser.parse_args()

    cfg             = OmegaConf.load(args_cli.config)
    cfg.num_samples = args_cli.num_samples
    cfg.imageset    = args_cli.split

    analyze_codes(cfg)
