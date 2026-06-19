import math

import torch
import torch.nn.functional as F
from torch import nn

from src.model.admm import LenslessOperators
from src.model.base_model import BaseModel
from src.model.utils import to_channel_last, to_nchw


class FISTAReconstruction(BaseModel):
    def __init__(
        self,
        n_iters: int,
        tv_weight: float,
        prox_iters: int,
        step_scale: float,
        eps: float,
        learnable: bool = False,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.prox_iters = prox_iters
        self.eps = eps
        self.learnable = learnable

        if learnable:
            self.raw_tv_weight = nn.Parameter(
                torch.full((n_iters,), math.log(math.expm1(tv_weight)))
            )
            self.raw_step_scale = nn.Parameter(
                torch.full((n_iters,), math.log(step_scale / (1.0 - step_scale)))
            )
        else:
            self.tv_weight = tv_weight
            self.step_scale = step_scale

    def forward(self, lensless, psf, **batch):
        ops = LenslessOperators(to_nchw(psf))
        measurement = ops.pad(to_nchw(lensless))

        lipschitz = ops.h_abs2.amax(dim=(-2, -1), keepdim=True)

        image = torch.zeros_like(measurement)
        extrapolated = image
        momentum = 1.0

        for idx in range(self.n_iters):
            image_prev = image
            step_scale, tv_weight = self._params(idx)
            step = step_scale / lipschitz.clamp_min(self.eps)

            residual = ops.ctc * (ops.H(extrapolated) - measurement)
            gradient = torch.fft.irfft2(
                ops.h_fft.conj() * torch.fft.rfft2(residual), s=ops.pad_shape
            )
            image = self._prox_tv_nonnegative(
                ops,
                extrapolated - step * gradient,
                weight=step * tv_weight,
            )

            next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
            extrapolated = image + (momentum - 1.0) / next_momentum * (image - image_prev)
            momentum = next_momentum

        reconstructed = ops.crop(image).clamp(0.0, 1.0)
        return {"reconstructed": to_channel_last(reconstructed)}

    def _params(self, idx):
        if self.learnable:
            return (
                torch.sigmoid(self.raw_step_scale[idx]),
                F.softplus(self.raw_tv_weight[idx]),
            )
        return self.step_scale, self.tv_weight

    def _prox_tv_nonnegative(
        self,
        ops: LenslessOperators,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        primal = target.clamp_min(0.0)
        primal_bar = primal
        dual = torch.zeros(
            *primal.shape[:2],
            2,
            *primal.shape[2:],
            device=primal.device,
            dtype=primal.dtype,
        )

        tau = 0.25
        sigma = 0.25
        theta = 1.0
        radius = weight.unsqueeze(2)

        for _ in range(self.prox_iters):
            dual = torch.clamp(dual + sigma * ops.Psi(primal_bar), -radius, radius)

            primal_prev = primal
            primal_step = primal - tau * ops.Psi_adj(dual)
            primal = ((primal_step + tau * target) / (1.0 + tau)).clamp_min(0.0)
            primal_bar = primal + theta * (primal - primal_prev)

        return primal
