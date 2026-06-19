import argparse
from pathlib import Path

from PIL import Image
import lpips
import torch
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from lensless_helpers.preprocessor import ALIGNMENT

TOP, LEFT = ALIGNMENT["top_left"]
H, W = ALIGNMENT["height"], ALIGNMENT["width"]


def load_gt(path):
    img = Image.open(path).convert("RGB")
    img = resize(img, [H, W], interpolation=InterpolationMode.NEAREST)
    return to_tensor(img)


def load_pred(path):
    img = to_tensor(Image.open(path).convert("RGB"))
    return img[:, TOP : TOP + H, LEFT : LEFT + W].contiguous()


def compute(pred, gt, lpips_fn):
    return {
        "PSNR": peak_signal_noise_ratio(pred, gt, data_range=1.0).item(),
        "SSIM": structural_similarity_index_measure(pred, gt, data_range=1.0).item(),
        "MSE": mean_squared_error(pred, gt).item(),
        "LPIPS": lpips_fn(2 * pred - 1, 2 * gt - 1).item(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute lensless reconstruction metrics.")
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()

    results = []
    for pred_path in sorted(args.pred_dir.glob("*.png")):
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [skip] no GT for {pred_path.name}")
            continue
        pred = load_pred(pred_path)[None].to(args.device)
        gt = load_gt(gt_path)[None].to(args.device)
        results.append(compute(pred, gt, lpips_fn))

    if not results:
        print("No matching image pairs found.")
        return

    print(f"\nMetrics over {len(results)} image(s) — ROI {H}×{W}")
    for name in results[0]:
        print(f"  {name:5s}: {sum(r[name] for r in results) / len(results):.4f}")


if __name__ == "__main__":
    main()
