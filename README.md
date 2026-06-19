# Lensless Computational Imaging

Reconstruction of images from a lensless camera (DigiCam) using ADMM-based algorithms,
implemented from scratch following [Monakhova et al. 2019](https://arxiv.org/abs/1908.11502)
and [Bezzam et al. 2025](https://arxiv.org/abs/2502.01102).

---

### Implemented algorithms

| Model | Description | Learnable params |
|---|---|---|
| **ADMM-100** | Classical ADMM, 100 iterations, fixed hyperparams | 0 |
| **Le-ADMM-20** | Unrolled ADMM, 20 iterations, per-iter μ/τ learned | 80 |
| **pre+U5+post** | 4M DRUNet pre + 5-iter ADMM + 4M DRUNet post | ~8M |
| **pre+U5** | 8M DRUNet pre + 5-iter ADMM | ~8M |
| **U5+post** | 5-iter ADMM + 8M DRUNet post | ~8M |

All ADMM variants share one implementation (`src/model/admm.py`). The modular wrappers
live in `src/model/modular.py`; DRUNet in `src/model/drunet.py`.

---

## Setup

```bash
conda create -n lensless python=3.11
conda activate lensless
pip install -r requirements.txt
```

---

## Data

The project uses [bezzam/DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
(HuggingFace). The `lensless_helpers/` module handles dataset loading and PSF simulation.

The `CustomDirDataset` expects this layout:

```
data_dir/
  lensless/   <ImageID>.png
  masks/      <ImageID>.npy
  lensed/     <ImageID>.png   # optional
```

---

## Training

```bash
# ADMM-100 (no training, fixed hyperparams — runs as a single "epoch")
python train.py model=admm100

# Le-ADMM-20 (unrolled, learns μ/τ per iteration)
python train.py model=admm

# Modular: pre(4M) + U5 + post(4M)
python train.py model=pre4_u5_post4

# Modular: pre(8M) + U5
python train.py model=pre8_u5

# Modular: U5 + post(8M)
python train.py model=u5_post8
```

All configs are Hydra-based (`src/configs/`). Key overrides:

```bash
python train.py model=pre4_u5_post4 trainer.n_epochs=50 trainer.device=cuda writer=wandb
```

Checkpoints are saved to `saved/` (best val PSNR). Training logs go to CometML (default)
or W&B (`writer=wandb`).

---

## Inference

```bash
python inference.py \
  inferencer.from_pretrained=daminovkamil/lensless-reconstruction \
  datasets.test.data_dir=demo_sample
```

`inferencer.from_pretrained` accepts a HuggingFace repo id, a local run directory, or a
`.pth` file — the architecture is read from the checkpoint, so no `model=` is needed.
Reconstructions are saved under `data/saved/reconstructions/<partition>/<ImageID>.png`
(the partition is the dataset split name, e.g. `test/` for `CustomDirDataset`).

---

## Metrics

```bash
python calculate_metrics.py \
  --gt_dir demo_sample/lensed \
  --pred_dir data/saved/reconstructions/test
```

Prints PSNR, SSIM, MSE, and LPIPS (VGG) computed on the ROI.

---

## Training settings

The learned models were trained on **RTX 5090**.

| Setting | Value |
|---|---|
| Dataset | DigiCam-Mirflickr-MultiMask-10K — `train[:90%]` / val `train[90%:]` / `test` |
| Optimizer | Adam, lr `1e-4`, betas `(0.9, 0.999)` |
| LR schedule | ConstantLR (no decay) |
| Seed | `1` |

ADMM-100 is parameter-free (fixed `μ=1e-4`, `τ=2e-4`, 100 iterations) — it is *evaluated*,
not trained, via `inference.py` (see below).

```bash
# evaluate the parameter-free ADMM-100 baseline on the test split
python inference.py model=admm100 datasets=digicam \
  '~datasets.train' '~datasets.val' \
  inferencer.from_pretrained=null 'inferencer.device_tensors=[lensless,psf,gt]'
```

---

## Results

All metrics on the **test split**, computed on the ROI.

| Model | Epochs | PSNR ↑ | SSIM ↑ | MSE ↓ | LPIPS ↓ |
|---|---|---|---|---|---|
| ADMM-100 | — | 6.69 | 0.292 | 0.2155 | 0.807 |
| Le-ADMM-20 | 50 | 12.11 | 0.366 | 0.0629 | 0.777 |
| pre+U5 | 21 | 14.29 | 0.279 | 0.0381 | 0.652 |
| U5+post | 48 | 15.30 | 0.385 | 0.0304 | 0.581 |
| **pre+U5+post** | **37** | **16.47** | **0.463** | **0.0233** | **0.537** |

---
