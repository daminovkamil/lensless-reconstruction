import os

from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch import nn


class BaseModel(nn.Module):
    """Base model: __str__ reports parameter counts (printed by train.py)."""

    @classmethod
    def from_pretrained(
        cls,
        source,
        filename="model_best.pth",
        config_filename="config.yaml",
        map_location="cpu",
    ):
        """Build a model from a saved config and load its weights.

        ``source`` may be a local checkpoint file, a local run directory, or a
        HuggingFace Hub repo id (e.g. ``"daminovkamil/lensless-reconstruction"``).
        The matching ``config.yaml`` saved next to the checkpoint at train time
        defines the architecture, so this works for every model variant.
        """
        ckpt_path, config_path = cls._resolve_pretrained(
            source, filename, config_filename
        )
        config = OmegaConf.load(config_path)
        model = instantiate(config.model)
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        return model.to(map_location)

    @staticmethod
    def _resolve_pretrained(source, filename, config_filename):
        if os.path.isfile(source):
            run_dir = os.path.dirname(source)
            return source, os.path.join(run_dir, config_filename)
        if os.path.isdir(source):
            return (
                os.path.join(source, filename),
                os.path.join(source, config_filename),
            )
        ckpt_path = hf_hub_download(repo_id=source, filename=filename)
        config_path = hf_hub_download(repo_id=source, filename=config_filename)
        return ckpt_path, config_path

    def __str__(self) -> str:
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
