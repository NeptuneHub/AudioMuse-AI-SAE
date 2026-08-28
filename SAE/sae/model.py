# AudioMuse-AI-SAE - https://github.com/NeptuneHub/AudioMuse-AI-SAE
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI-SAE/blob/main/LICENSE>
#
# Method from "Steering dense music retrieval with open-vocabulary concept discovery"
# by Julien Guinot, Alain Riou, Elio Quinton and Gyorgy Fazekas
# <https://arxiv.org/abs/2608.08757>, used under CC BY 4.0
# <https://creativecommons.org/licenses/by/4.0/>

"""BatchTopK sparse autoencoder trained on DCLAP audio embeddings.

Implements the dictionary described in "Steering dense music retrieval with
open-vocabulary concept discovery" (arXiv:2608.08757): a linear encoder E, a
BatchTopK sparsifying map Pi and a linear decoder D whose weights are tied to
the encoder, optimised with L(z, zhat, s) = ||z - zhat||^2 where sparsity comes
from Pi instead of an explicit penalty. train_sae.py owns the optimisation loop,
export_onnx.py wraps SAEInference for the ONNX graph.

Main Features:
* BatchTopK keeps the batch_size * k largest activations across the whole batch,
  and an EMA of the smallest kept activation becomes a fixed threshold so that
  inference is batch independent (a JumpReLU) and therefore ONNX exportable.
* Auxiliary dead latent loss (AuxK) reconstructs the residual from latents that
  have not fired for dead_steps, which keeps the dictionary from collapsing.
* The affine input normalisation is carried as buffers inside the module, so a
  saved checkpoint and an exported graph both consume raw 512 dim embeddings.
* Tied decoder by default (as in the paper); the untied path keeps unit norm
  decoder columns and strips the radial component of their gradient.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int = 512
    d_sae: int = 4096
    k: int = 20
    tied: bool = True
    aux_k: int = 512
    aux_alpha: float = 0.03125
    dead_steps: int = 500
    threshold_beta: float = 0.999

    @classmethod
    def from_dict(cls, payload):
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


class BatchTopKSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        weight = torch.empty(cfg.d_sae, cfg.d_in)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        weight = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_enc = nn.Parameter(weight)
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_pre = nn.Parameter(torch.zeros(cfg.d_in))
        if cfg.tied:
            self.register_parameter('W_dec', None)
        else:
            self.W_dec = nn.Parameter(weight.t().contiguous().clone())
        self.register_buffer('threshold', torch.zeros(()))
        self.register_buffer('steps_since_fired', torch.zeros(cfg.d_sae))
        self.register_buffer('fired_count', torch.zeros(cfg.d_sae))
        self.register_buffer('norm_mean', torch.zeros(cfg.d_in))
        self.register_buffer('norm_scale', torch.ones(()))

    @property
    def decoder_weight(self):
        return self.W_enc.t() if self.cfg.tied else self.W_dec

    def set_normalization(self, mean, scale):
        self.norm_mean.copy_(torch.as_tensor(mean, dtype=self.norm_mean.dtype))
        self.norm_scale.fill_(float(scale))

    def init_pre_bias(self, mean_vector):
        with torch.no_grad():
            self.b_pre.copy_(torch.as_tensor(mean_vector, dtype=self.b_pre.dtype))

    @torch.no_grad()
    def init_from_data(self, samples, generator=None):
        index = torch.randint(
            0, samples.shape[0], (self.cfg.d_sae,), generator=generator, device=samples.device
        )
        directions = samples[index] - self.b_pre
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W_enc.copy_(directions)
        if not self.cfg.tied:
            self.W_dec.copy_(directions.t().contiguous())

    def normalize(self, x):
        return (x - self.norm_mean) * self.norm_scale

    def denormalize(self, x):
        return x / self.norm_scale + self.norm_mean

    def pre_activation(self, x):
        return F.relu(F.linear(x - self.b_pre, self.W_enc, self.b_enc))

    def batch_topk(self, pre_acts):
        batch, latents = pre_acts.shape
        keep = min(self.cfg.k * batch, batch * latents)
        flat = pre_acts.reshape(-1)
        values, indices = torch.topk(flat, keep, sorted=False)
        sparse = torch.zeros_like(flat).scatter_(0, indices, values)
        return sparse.view(batch, latents), values

    def encode(self, x):
        pre = self.pre_activation(x)
        return pre * (pre > self.threshold).to(pre.dtype)

    def decode(self, acts):
        return F.linear(acts, self.decoder_weight) + self.b_pre

    def forward(self, x):
        pre = self.pre_activation(x)
        acts, kept = self.batch_topk(pre)
        return self.decode(acts), acts, pre, kept

    def auxiliary_loss(self, x, recon, pre):
        dead = self.steps_since_fired > self.cfg.dead_steps
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return x.new_zeros(()), 0
        keep = min(self.cfg.aux_k, n_dead)
        masked = torch.where(dead, pre, torch.zeros_like(pre))
        values, indices = torch.topk(masked, keep, dim=-1)
        aux_acts = torch.zeros_like(pre).scatter_(-1, indices, values)
        residual = (x - recon).detach()
        aux_recon = F.linear(aux_acts, self.decoder_weight)
        return (aux_recon - residual).pow(2).sum(dim=-1).mean(), n_dead

    @torch.no_grad()
    def update_threshold(self, kept):
        positive = kept[kept > 0]
        if positive.numel() == 0:
            return
        batch_min = positive.min()
        if float(self.threshold) == 0.0:
            self.threshold.fill_(float(batch_min))
        else:
            self.threshold.mul_(self.cfg.threshold_beta)
            self.threshold.add_((1.0 - self.cfg.threshold_beta) * batch_min)

    @torch.no_grad()
    def update_firing_stats(self, acts):
        fired = (acts > 0).any(dim=0)
        self.steps_since_fired += 1.0
        self.steps_since_fired[fired] = 0.0
        self.fired_count += (acts > 0).sum(dim=0).to(self.fired_count.dtype)

    @torch.no_grad()
    def unit_norm_decoder(self):
        if self.cfg.tied:
            return
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_grad(self):
        if self.cfg.tied or self.W_dec.grad is None:
            return
        radial = (self.W_dec.grad * self.W_dec).sum(dim=0, keepdim=True) * self.W_dec
        self.W_dec.grad.sub_(radial)

    def dictionary(self):
        weight = self.decoder_weight
        return weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': asdict(self.cfg), 'state_dict': self.state_dict()}, path)
        path.with_suffix('.json').write_text(
            json.dumps(asdict(self.cfg), indent=2) + '\n', newline='\n'
        )

    @classmethod
    def load(cls, path, device='cpu'):
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(SAEConfig.from_dict(payload['config']))
        model.load_state_dict(payload['state_dict'])
        return model.to(device)


class SAEInference(nn.Module):
    def __init__(self, sae, part='full'):
        super().__init__()
        self.sae = sae
        self.part = part

    def forward(self, x):
        if self.part == 'decoder':
            return self.sae.denormalize(self.sae.decode(x))
        latents = self.sae.encode(self.sae.normalize(x))
        if self.part == 'encoder':
            return latents
        return latents, self.sae.denormalize(self.sae.decode(latents))
