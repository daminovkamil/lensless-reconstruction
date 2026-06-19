import os

import torch
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn


class BaseModel(nn.Module):
    """Base model: __str__ reports parameter counts (printed by train.py)."""

    @classmethod
    def from_pretrained(cls, source, filename="model_best.pth", map_location="cpu"):
        """Load from a local file/dir or HuggingFace repo id; architecture is read from the checkpoint config."""
        if os.path.isfile(source):
            ckpt_path = source
        elif os.path.isdir(source):
            ckpt_path = os.path.join(source, filename)
        else:
            ckpt_path = hf_hub_download(repo_id=source, filename=filename)

        checkpoint = torch.load(
            ckpt_path, map_location=map_location, weights_only=False
        )
        model = instantiate(OmegaConf.create(checkpoint["config"]).model)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location)

    def __str__(self) -> str:
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
