from pathlib import Path

import numpy as np
from PIL import Image

from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset
from src.datasets.preprocessing import crop_gt, load_measurement


class CustomDirDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        data_dir = Path(data_dir)
        self.lensless_dir = data_dir / "lensless"
        self.masks_dir = data_dir / "masks"
        self.lensed_dir = data_dir / "lensed"
        self.has_gt = self.lensed_dir.is_dir()

        assert self.lensless_dir.is_dir(), f"Missing lensless/ in {data_dir}"
        assert self.masks_dir.is_dir(), f"Missing masks/ in {data_dir}"

        ids = sorted(p.stem for p in self.lensless_dir.glob("*.png"))
        assert len(ids) > 0, f"No .png measurements found in {self.lensless_dir}"
        index = [{"id": i} for i in ids]

        super().__init__(index, limit, shuffle_index, instance_transforms)

    def __getitem__(self, ind):
        img_id = self._index[ind]["id"]

        lensless = load_measurement(Image.open(self.lensless_dir / f"{img_id}.png"))
        psf = simulate_psf_from_mask(np.load(self.masks_dir / f"{img_id}.npy"))

        instance_data = {
            "lensless": lensless.float(),
            "psf": psf[0].float(),
            "img_id": img_id,
        }
        if self.has_gt:
            gt = crop_gt(Image.open(self.lensed_dir / f"{img_id}.png"), lensless)
            instance_data["gt"] = gt.float()

        return self.preprocess_data(instance_data)

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "id" in entry, "Each item must include 'id' (ImageID / file stem)."
