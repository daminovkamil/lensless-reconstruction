import lpips
import torch
from torch import nn

from lensless_helpers.preprocessor import ALIGNMENT


def crop_roi(x: torch.Tensor) -> torch.Tensor:
    # (B, H, W, C)
    top, left = ALIGNMENT["top_left"]
    height, width = ALIGNMENT["height"], ALIGNMENT["width"]
    return x[:, top : top + height, left : left + width, :]


class LenslessLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.mse = nn.MSELoss()

        self.lpips = lpips.LPIPS(net="vgg")
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, reconstructed, gt, **batch):
        # (B, H, W, 3) in [0, 1]
        pred = crop_roi(reconstructed).permute(0, 3, 1, 2)
        target = crop_roi(gt).permute(0, 3, 1, 2)

        mse = self.mse(pred, target)
        lpips_value = self.lpips(2 * pred - 1, 2 * target - 1).mean()

        total = self.mse_weight * mse + self.lpips_weight * lpips_value
        return {"loss": total, "mse_loss": mse, "lpips_loss": lpips_value}
