from src.model.base_model import BaseModel
from src.model.utils import to_channel_last, to_nchw


class ModularReconstruction(BaseModel):
    def __init__(self, camera_inverter, pre_processor=None, post_processor=None):
        super().__init__()
        self.pre_processor = pre_processor
        self.camera_inverter = camera_inverter
        self.post_processor = post_processor

    @staticmethod
    def _refine(processor, x):
        nchw = to_nchw(x)
        return to_channel_last(nchw + processor(nchw))

    def forward(self, lensless, psf, **batch):
        measurement = lensless
        if self.pre_processor is not None:
            measurement = self._refine(self.pre_processor, measurement)

        reconstructed = self.camera_inverter(measurement, psf, **batch)["reconstructed"]

        if self.post_processor is not None:
            reconstructed = self._refine(self.post_processor, reconstructed)

        return {"reconstructed": reconstructed.clamp(0, 1)}
