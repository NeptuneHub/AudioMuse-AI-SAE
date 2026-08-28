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

"""Loader for the extracted DCLAP embedding matrix used to train the SAE.

extract_embeddings.py writes embeddings.npy (float32, one row per 10 second
segment) plus groups.npy holding the track index of every row. This module turns
that pair into train and validation tensors, splitting on the TRACK id so that
overlapping segments of one song never straddle the split, and derives the
affine normalisation that train_sae.py stores inside the checkpoint.

Main Features:
* Group aware train/validation split, seeded and reproducible.
* Optional row wise L2 normalisation, matching how AudioMuse stores CLAP vectors.
* Dataset scale factor chosen so that E[||x||] = sqrt(d_in), the usual
  conditioning trick for top-k dictionaries; mean centring is opt in because
  b_pre already absorbs the dataset mean.
* load_track_names resolves the display name of every row, preferring the
  track_names.json written by the extractor over the legacy meta.json list.
* Reads the matrix with one sequential load by default; --mmap keeps the memory
  mapped path for a corpus too large for RAM, which is far slower on 9p because
  the row index turns into one small read per row.
* Preloads the whole matrix onto the training device when it fits, which removes
  the host to device copy from the inner loop.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch


class EmbeddingStore:
    def __init__(
        self,
        embeddings_path,
        groups_path=None,
        l2_normalize=True,
        center=False,
        val_fraction=0.02,
        seed=0,
        device='cpu',
        preload=True,
        mmap=False,
    ):
        matrix = np.load(embeddings_path, mmap_mode='r' if mmap else None)
        if matrix.ndim != 2:
            raise ValueError(f'expected a 2D embedding matrix, got shape {matrix.shape}')
        self.n_rows, self.d_in = matrix.shape
        groups = None
        if groups_path is not None:
            groups = np.load(groups_path)
            if groups.shape[0] != self.n_rows:
                raise ValueError('groups.npy does not line up with embeddings.npy')

        rng = np.random.default_rng(seed)
        if groups is None:
            order = rng.permutation(self.n_rows)
            n_val = max(1, int(round(self.n_rows * val_fraction)))
            val_idx, train_idx = order[:n_val], order[n_val:]
        else:
            unique = np.unique(groups)
            shuffled = rng.permutation(unique)
            n_val_groups = max(1, int(round(unique.size * val_fraction)))
            val_groups = np.zeros(int(unique.max()) + 1, dtype=bool)
            val_groups[shuffled[:n_val_groups]] = True
            mask = val_groups[groups]
            val_idx = np.nonzero(mask)[0]
            train_idx = np.nonzero(~mask)[0]

        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        self.train = self._materialize(matrix, train_idx, l2_normalize)
        self.val = self._materialize(matrix, val_idx, l2_normalize)
        self.train_groups = None if groups is None else groups[train_idx]
        self.val_groups = None if groups is None else groups[val_idx]

        self.mean = self.train.mean(dim=0) if center else torch.zeros(self.d_in)
        norms = (self.train - self.mean).norm(dim=-1)
        self.scale = math.sqrt(self.d_in) / max(float(norms.mean()), 1e-8)

        self.device = torch.device(device)
        self.preloaded = bool(preload) and self._fits(self.device)
        if self.preloaded:
            self.train = self.train.to(self.device)
            self.val = self.val.to(self.device)
        self.mean = self.mean.to(self.train.device)

    @staticmethod
    def _materialize(matrix, index, l2_normalize):
        rows = torch.from_numpy(np.ascontiguousarray(matrix[index])).float()
        if l2_normalize:
            rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return rows

    def _fits(self, device):
        if device.type != 'cuda':
            return True
        needed = (self.train.numel() + self.val.numel()) * 4
        free, _total = torch.cuda.mem_get_info(device)
        return needed < free * 0.5

    def normalized(self, rows):
        return (rows - self.mean) * self.scale

    def train_batches(self, batch_size, steps, seed=0):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        n = self.train.shape[0]
        order = torch.randperm(n, generator=generator)
        cursor = 0
        for _ in range(steps):
            if cursor + batch_size > n:
                order = torch.randperm(n, generator=generator)
                cursor = 0
            index = order[cursor : cursor + batch_size]
            cursor += batch_size
            batch = self.train[index.to(self.train.device)]
            yield batch if self.preloaded else batch.to(self.device, non_blocking=True)

    def val_tensor(self, limit=None):
        rows = self.val if limit is None else self.val[:limit]
        return rows if self.preloaded else rows.to(self.device)

    def summary(self):
        return {
            'rows_total': int(self.n_rows),
            'rows_train': int(self.train.shape[0]),
            'rows_val': int(self.val.shape[0]),
            'd_in': int(self.d_in),
            'scale': float(self.scale),
            'preloaded': bool(self.preloaded),
        }


def load_track_names(data_dir):
    data_dir = Path(data_dir)
    names_path = data_dir / 'track_names.json'
    if names_path.exists():
        names = json.loads(names_path.read_text()).get('names')
        if names:
            return list(names)
    meta_path = data_dir / 'meta.json'
    if meta_path.exists():
        entries = json.loads(meta_path.read_text()).get('tracks') or []
        if entries:
            return [Path(entry['path']).name for entry in entries]
    return None
