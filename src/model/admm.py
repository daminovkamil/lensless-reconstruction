import math

import torch
import torch.nn.functional as F
from torch import nn

from src.model.base_model import BaseModel
from src.model.utils import to_channel_last, to_nchw


def soft_threshold(x, thresh):
    return x.sign() * (x.abs() - thresh).clamp_min(0.0)


class LenslessOperators:
    def __init__(self, psf):
        h, w = psf.shape[-2:]

        self.in_shape = (h, w)
        self.pad_shape = (2 * h, 2 * w)
        self.top, self.left = h // 2, w // 2

        ph, pw = self.pad_shape
        device, dtype = psf.device, psf.dtype

        psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        self.h_fft = torch.fft.rfft2(torch.fft.ifftshift(self.pad(psf), dim=(-2, -1)))
        self.h_abs2 = self.h_fft.real**2 + self.h_fft.imag**2

        wy = 2 * torch.pi * torch.fft.fftfreq(ph, device=device, dtype=dtype)
        wx = 2 * torch.pi * torch.fft.rfftfreq(pw, device=device, dtype=dtype)
        self.psi_abs2 = (2 - 2 * torch.cos(wy)).view(ph, 1) + (
            2 - 2 * torch.cos(wx)
        ).view(1, -1)

        self.ctc = torch.zeros(1, 1, ph, pw, device=device, dtype=dtype)
        self.ctc[..., self.top : self.top + h, self.left : self.left + w] = 1.0

    def H(self, x):
        return torch.fft.irfft2(self.h_fft * torch.fft.rfft2(x), s=self.pad_shape)

    def Psi(self, x):
        dv = x - torch.roll(x, -1, dims=-2)
        dh = x - torch.roll(x, -1, dims=-1)
        return torch.stack([dv, dh], dim=2)

    def Psi_adj(self, u):
        dv, dh = u[:, :, 0], u[:, :, 1]
        return (dv - torch.roll(dv, 1, dims=-2)) + (dh - torch.roll(dh, 1, dims=-1))

    def update_x(self, base, v_term, mu1, mu2, mu3):
        rhs = torch.fft.rfft2(base) + self.h_fft.conj() * torch.fft.rfft2(v_term)
        denom = mu1 * self.h_abs2 + mu2 * self.psi_abs2 + mu3
        return torch.fft.irfft2(rhs / denom, s=self.pad_shape)

    def crop(self, x):
        h, w = self.in_shape
        return x[..., self.top : self.top + h, self.left : self.left + w]

    def pad(self, b):
        h, w = self.in_shape
        out = b.new_zeros(*b.shape[:-2], *self.pad_shape)
        out[..., self.top : self.top + h, self.left : self.left + w] = b
        return out


class UnrolledADMM(BaseModel):
    def __init__(self, n_iters, learnable, mu_init, tau_init):
        super().__init__()
        self.n_iters = n_iters
        self.learnable = learnable

        if learnable:
            raw_mu_init = math.log(math.expm1(mu_init))
            raw_tau_init = math.log(math.expm1(tau_init))

            self.raw_mu1 = nn.Parameter(torch.full((n_iters,), raw_mu_init))
            self.raw_mu2 = nn.Parameter(torch.full((n_iters,), raw_mu_init))
            self.raw_mu3 = nn.Parameter(torch.full((n_iters,), raw_mu_init))
            self.raw_tau = nn.Parameter(torch.full((n_iters,), raw_tau_init))
        else:
            self.mu_init, self.tau_init = mu_init, tau_init

    def _iteration_params(self, k):
        if self.learnable:
            return (
                F.softplus(self.raw_mu1[k]),
                F.softplus(self.raw_mu2[k]),
                F.softplus(self.raw_mu3[k]),
                F.softplus(self.raw_tau[k]),
            )
        return self.mu_init, self.mu_init, self.mu_init, self.tau_init

    def forward(self, lensless, psf, **batch):
        ops = LenslessOperators(to_nchw(psf))
        ctb = ops.pad(to_nchw(lensless))

        x = torch.zeros_like(ctb)
        Hx = torch.zeros_like(x)
        Psix = ops.Psi(x)
        a1 = torch.zeros_like(x)
        a2 = torch.zeros_like(Psix)
        a3 = torch.zeros_like(x)

        for k in range(self.n_iters):
            mu1, mu2, mu3, tau = self._iteration_params(k)

            u = soft_threshold(Psix + a2 / mu2, tau / mu2)
            v = (a1 + mu1 * Hx + ctb) / (ops.ctc + mu1)
            w = torch.relu(a3 / mu3 + x)

            base = (mu3 * w - a3) + ops.Psi_adj(mu2 * u - a2)
            x = ops.update_x(base, mu1 * v - a1, mu1, mu2, mu3)

            Hx = ops.H(x)
            Psix = ops.Psi(x)
            a1 = a1 + mu1 * (Hx - v)
            a2 = a2 + mu2 * (Psix - u)
            a3 = a3 + mu3 * (x - w)

        reconstructed = ops.crop(x).clamp(0.0, 1.0)
        return {"reconstructed": to_channel_last(reconstructed)}
