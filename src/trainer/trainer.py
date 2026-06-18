from contextlib import nullcontext

import torch

from src.loss.lensless import crop_roi
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with self._autocast_context():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _autocast_context(self):
        amp_dtype = self.config.trainer.get("amp_dtype", "off")
        if (
            self.is_train
            and amp_dtype == "bf16"
            and str(self.device).startswith("cuda")
        ):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode != "train":
            self._log_reconstructions(batch, mode)

    def _log_reconstructions(self, batch, mode, examples=8):
        recon = crop_roi(batch["reconstructed"]).detach().cpu().clamp(0, 1)
        gt = crop_roi(batch["gt"]).detach().cpu().clamp(0, 1)
        n = min(examples, recon.shape[0])

        pairs = [torch.cat([recon[i], gt[i]], dim=1) for i in range(n)]
        grid = (torch.cat(pairs, dim=0) * 255).to(torch.uint8).numpy()

        self.writer.add_image(f"{mode}_reconstruction_vs_gt", grid)
