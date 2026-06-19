import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import lpips
import torch
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_tensor

from lensless_helpers.preprocessor import ALIGNMENT, convert_image_to_float, force_rgb
from lensless_helpers.utils import resize

TOP, LEFT = ALIGNMENT["top_left"]
H, W = ALIGNMENT["height"], ALIGNMENT["width"]


def load_gt(path):
    img = convert_image_to_float(force_rgb(np.array(Image.open(path))))
    img = resize(img, shape=(H, W, 3), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def load_pred(path):
    img = to_tensor(Image.open(path).convert("RGB"))
    return img[:, TOP : TOP + H, LEFT : LEFT + W].contiguous()


def compute(pred, gt, metric_fns, lpips_fn):
    values = {}
    for name, metric in metric_fns.items():
        metric.reset()
        values[name] = metric(pred, gt).item()
    values["LPIPS"] = lpips_fn(2 * pred - 1, 2 * gt - 1).item()
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Compute lensless reconstruction metrics.")
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    metric_fns = {
        "PSNR": PeakSignalNoiseRatio(data_range=1.0).to(args.device),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device),
        "MSE": MeanSquaredError().to(args.device),
    }
    lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()

    results = []
    for pred_path in sorted(args.pred_dir.glob("*.png")):
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [skip] no GT for {pred_path.name}")
            continue
        pred = load_pred(pred_path)[None].to(args.device)
        gt = load_gt(gt_path)[None].to(args.device)
        results.append(compute(pred, gt, metric_fns, lpips_fn))

    if not results:
        print("No matching image pairs found.")
        return

    print(f"\nMetrics over {len(results)} image(s) — ROI {H}×{W}")
    for name in results[0]:
        print(f"  {name:5s}: {sum(r[name] for r in results) / len(results):.4f}")


if __name__ == "__main__":
    main()
