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

"""Packs the validated concepts into the small catalogue the app ships.

validate_concepts.py writes a working file full of diagnostics: rejected terms,
per latent document frequencies and the example tracks used to eyeball whether a
concept means anything. None of that belongs in a shipped artifact. The example
tracks in particular are real titles out of whoever's library the dictionary was
fitted on, so they are deliberately not carried over.

What the serving path needs per concept is the latent support and the edit mask
over it: the inverted code weighted by each latent's inverse document frequency,
restricted to its largest coordinates and then L2 normalised, as the reference
implementation does. Normalising is what makes a strength setting mean the same
thing for every concept, because the mask is a fixed length step in latent space
rather than a per concept scale. Weighting by each latent's mean activation was
tried instead and scored better against the dictionary's own viola detector, but
that measurement is circular: judged by ear the lists it produced were string
quartet instrumentals that had lost the genre and the vocals entirely.

Main Features:
* Emits dclap_sae_concepts.json next to the exported ONNX graphs at the project
  root, grouped and ordered for the refinement UI.
* Carries the grounding score so the UI can show how well a term is evidenced.
* Never copies track names, paths, item ids or corpus statistics out of the
  training library: the catalogue describes the model, not whose music it saw.
"""

import argparse
import json
from pathlib import Path

import numpy as np

CATEGORY_LABELS = {
    'genre': 'Genre',
    'instrument': 'Instrument',
    'mood': 'Mood',
    'vocals': 'Vocals',
    'production': 'Production',
    'tempo': 'Tempo',
    'era': 'Era',
}
CATEGORY_ORDER = ['genre', 'instrument', 'vocals', 'mood', 'production', 'tempo', 'era']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Build the shipped concept catalogue')
    parser.add_argument('--validated', default='runs/concepts_validated.json')
    parser.add_argument('--stats', default='runs/dclap_sae_d1024/best_eval/latent_stats.npz')
    parser.add_argument('--support-size', type=int, default=12)
    parser.add_argument('--encoder', default='dclap_sae_k20_d1024_best_encoder.onnx')
    parser.add_argument('--decoder', default='dclap_sae_k20_d1024_best_decoder.onnx')
    parser.add_argument('--out', default='../dclap_sae_concepts.json')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = json.loads(Path(args.validated).read_text())
    stats = np.load(args.stats)
    idf = np.clip(stats['idf'], 0.0, None)

    concepts = []
    for category in CATEGORY_ORDER:
        for entry in source['categories'].get(category, []):
            support = np.asarray(entry['support'], dtype=np.int64)
            weights = np.asarray(entry['support_values'], dtype=np.float64) * idf[support]
            order = np.argsort(-np.abs(weights))[: args.support_size]
            support = support[order]
            weights = weights[order]
            norm = float(np.linalg.norm(weights))
            if norm <= 0.0:
                continue
            weights = weights / norm
            concepts.append(
                {
                    'term': entry['term'],
                    'category': category,
                    'label': CATEGORY_LABELS.get(category, category.title()),
                    'grounding': entry['grounding'],
                    'support': [int(i) for i in support],
                    'mask': [round(float(v), 6) for v in weights],
                }
            )

    bundle = {
        'encoder': args.encoder,
        'decoder': args.decoder,
        'd_sae': 1024,
        'category_order': CATEGORY_ORDER,
        'concepts': concepts,
    }
    out = Path(args.out)
    out.write_text(json.dumps(bundle, indent=2) + '\n', newline='\n')
    print(f'wrote {out} with {len(concepts)} concepts, {out.stat().st_size} bytes')


if __name__ == '__main__':
    main()
