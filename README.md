# EditSSC

Official code release for our CVPR Workshop paper.

## Project Page

- [Webpage](https://astra-vision.github.io/EditSSC/)

## Setup

### Conda (recommended)

```bash
cd /path/to/EditSSC
conda env create -f environment.yml
conda activate editssc
```

## Autoencoder Training

Run:

```bash
python script/train_ae_main.py --config configs/common_ae_base.yaml
```


## Save Triplanes

Run triplane extraction:

```bash
python script/save_triplane.py --config configs/common_ae_base.yaml
```


## Diffusion Training

Train an unconditional diffusion model on the saved triplanes:

```bash
python script/train_diffusion_main.py --config configs/common_diffusion_base.yaml
```

Train a LiDAR-conditioned diffusion model :

```bash
python script/train_diffusion_main.py --config configs/common_diffusion_cond_lidar.yaml
```


## Sample Generation

Generate scenes from a trained diffusion checkpoint. 

```bash
python generation/generate_samples.py --config configs/common_diffusion_base.yaml
```

## Sketch guided generation

Training-free scene generation from a BEV sketch (canvas). The pipeline has two steps:
first build a **class ↔ VQ-VAE code** mapping, then run inpainting with a sketch layout.

### Step 1 — Analyze VQ-VAE code ↔ class correspondences

Run this on your trained autoencoder. It measures which VQ-VAE codebook entries
correspond to each semantic class and writes two files :

- `vqvae_codes_analysis.txt` — human-readable report
- `vqvae_codes_99_coverage.json` — mapping used by sketch generation (~99% class coverage)

```bash
python generation/analyze_vqvae_codes.py --config configs/common_ae_base.yaml
```


### Step 2 — Generate scenes from a sketch (`training_free_gen.py`)

Pass the JSON from step 1 to map each canvas pixel (semantic class id) to a VQ-VAE
code, then inpaint with the diffusion model.

List built-in canvas layouts:
```bash
python generation/training_free_gen.py --list
```

Run with one or several predefined layouts (roundabout, S-road, cross-road, etc.):

```bash
python generation/training_free_gen.py \
  --config configs/common_diffusion_base.yaml \
  --codes-json models/semantic_ae/common_ae_base/vqvae_codes_99_coverage.json \
  --canvas roundabout
```

### Custom canvas
You can also use your own BEV sketches by adding a builder in
`generation/training_free_gen.py`:

1. Define a function that returns a `(128, 128)` NumPy array of **SemanticKITTI train
   ids** (e.g. `0` empty, `9` road, `15` vegetation, `1` car — see canvas builders in
   the same file).
2. Register it in `CANVAS_REGISTRY` with a `build_fn` and `save_dir_suffix`.
3. Run with `--canvas your_layout_name` as above.

