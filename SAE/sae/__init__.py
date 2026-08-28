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

"""Package root for the DCLAP sparse autoencoder tooling.

Re-exports the model, dataset and metric helpers used by the top level scripts
(extract_embeddings.py, train_sae.py, evaluate_sae.py, export_onnx.py,
concept_attribution.py and steer.py).

Main Features:
* BatchTopKSAE and SAEConfig, the trainable dictionary over DCLAP embeddings.
* EmbeddingStore and load_track_names, the loaders for extracted corpora.
* reconstruction_metrics, the shared L0 / FVU / cosine reporting helper.
* Re-exports resolve lazily, so extract_embeddings.py can pull a corpus out of
  Postgres on a machine that has no torch installed.
"""

import importlib

_EXPORTS = {
    'BatchTopKSAE': 'sae.model',
    'SAEConfig': 'sae.model',
    'SAEInference': 'sae.model',
    'EmbeddingStore': 'sae.data',
    'load_track_names': 'sae.data',
    'reconstruction_metrics': 'sae.metrics',
}

__all__ = [
    'BatchTopKSAE',
    'EmbeddingStore',
    'SAEConfig',
    'SAEInference',
    'load_track_names',
    'reconstruction_metrics',
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
