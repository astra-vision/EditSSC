# EditSSC

Official code release for our CVPR Workshop paper.

## Setup

### Conda (recommended)

```bash
cd /home/fbalde/EditSSC
conda env create -f environment.yml
conda activate editssc
```

## Autoencoder Training

Training entrypoint:

- `script/train_ae_main.py`

Default config:

- `configs/common_ae_base.yaml`

Run:

```bash
cd /home/fbalde/EditSSC
conda activate editssc
python script/train_ae_main.py --config configs/common_ae_base.yaml
```

To run autoencoder training with a custom config:

```bash
python script/train_ae_main.py --config /path/to/your_config.yaml
```

Training artifacts are written under the `save_path` defined in the config.
