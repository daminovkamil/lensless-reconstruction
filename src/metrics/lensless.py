import torch

from src.loss.lensless import crop_roi
from src.metrics.base_metric import BaseMetric


class LenslessMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, reconstructed, gt, **kwargs):
        pred = crop_roi(reconstructed).permute(0, 3, 1, 2).contiguous()
        target = crop_roi(gt).permute(0, 3, 1, 2).contiguous()
        return self.metric(pred, target).item()
