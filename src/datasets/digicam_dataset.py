import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from datasets import load_dataset
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset
from src.datasets.preprocessing import crop_gt, load_measurement


class DigiCamDataset(BaseDataset):
    def __init__(
        self,
        repo_id,
        split,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        self.repo_id = repo_id
        self.hf_token = self._get_hf_token()
        self.hf_dataset = load_dataset(repo_id, split=split, token=self.hf_token)
        self.mask_dir = self._download_masks()
        self.psf_cache_dir = self._get_psf_cache_dir()
        self._psf_cache = {}

        mask_labels = self.hf_dataset["mask_label"]
        index = [
            {"row": i, "mask_label": int(mask_labels[i]), "id": f"{split}_{i:06d}"}
            for i in range(len(mask_labels))
        ]

        super().__init__(index, limit, shuffle_index, instance_transforms)

    def __getitem__(self, ind):
        entry = self._index[ind]
        row = self.hf_dataset[entry["row"]]

        lensless = load_measurement(row["lensless"])
        instance_data = {
            "lensless": lensless.float(),
            "psf": self._get_psf(entry["mask_label"]),
            "gt": crop_gt(row["lensed"], lensless).float(),
            "img_id": entry["id"],
        }
        return self.preprocess_data(instance_data)

    def _get_psf(self, mask_label):
        if mask_label not in self._psf_cache:
            cache_path = self.psf_cache_dir / f"mask_{mask_label}.pt"
            if cache_path.exists():
                self._psf_cache[mask_label] = torch.load(cache_path, map_location="cpu")
            else:
                mask_path = self.mask_dir / f"mask_{mask_label}.npy"
                if not mask_path.exists():
                    mask_path = Path(
                        hf_hub_download(
                            self.repo_id,
                            f"masks/mask_{mask_label}.npy",
                            repo_type="dataset",
                            token=self.hf_token,
                        )
                    )
                psf = simulate_psf_from_mask(np.load(mask_path))[0].float()
                self._save_psf(cache_path, psf)
                self._psf_cache[mask_label] = psf
        return self._psf_cache[mask_label]

    def _download_masks(self):
        snapshot_dir = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            allow_patterns=["masks/*"],
            token=self.hf_token,
        )
        return Path(snapshot_dir) / "masks"

    @staticmethod
    def _get_hf_token():
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

    @staticmethod
    def _get_psf_cache_dir():
        cache_dir = Path(
            os.environ.get(
                "DIGICAM_PSF_CACHE_DIR",
                Path.home() / ".cache" / "digicam_psfs",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def _save_psf(cache_path, psf):
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save(psf, tmp_path)
            tmp_path.replace(cache_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "row" in entry, "Each item must include 'row' (HF row index)."
            assert "mask_label" in entry, "Each item must include 'mask_label'."
            assert "id" in entry, "Each item must include 'id'."
