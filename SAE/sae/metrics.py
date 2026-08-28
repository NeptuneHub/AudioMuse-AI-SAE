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

"""Reconstruction and sparsity metrics shared by training and evaluation.

train_sae.py calls these every log interval on a held out slice and
evaluate_sae.py calls them once over the full validation split, so both report
the same numbers: the achieved L0, the fraction of variance unexplained and the
cosine similarity that actually matters for a retrieval index.

Main Features:
* reconstruction_metrics: L0, FVU, explained variance, normalised MSE and mean
  cosine similarity between an embedding batch and its reconstruction.
* dead_latent_fraction: share of dictionary entries that never fired.
* fvu is computed against the variance of the evaluated batch, so a value of 1.0
  means the SAE does no better than predicting the batch mean.
"""

import torch


@torch.no_grad()
def reconstruction_metrics(x, recon, acts):
    residual = x - recon
    mse = residual.pow(2).sum(dim=-1)
    variance = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)
    energy = x.pow(2).sum(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(x, recon, dim=-1)
    fvu = (mse.sum() / variance.sum().clamp_min(1e-12)).item()
    return {
        'l0': (acts > 0).float().sum(dim=-1).mean().item(),
        'mse': mse.mean().item(),
        'normalized_mse': (mse.sum() / energy.sum().clamp_min(1e-12)).item(),
        'fvu': fvu,
        'explained_variance': 1.0 - fvu,
        'cosine': cosine.mean().item(),
    }


@torch.no_grad()
def dead_latent_fraction(fired_count):
    return (fired_count == 0).float().mean().item()
