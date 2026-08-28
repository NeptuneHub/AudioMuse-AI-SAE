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

"""Steers DCLAP retrieval by editing the sparse code of the query.

Closes the loop of arXiv:2608.08757: concept_attribution.py turns a phrase into
a support of latents, this script amplifies or suppresses that support inside a
query embedding and shows what the change does to the retrieved neighbours. The
numbers that matter are paired: how much the target concept moves, and how much
of the original result set survives (the edit versus preservation trade off).

Main Features:
* Two edit modes: code, which rewrites the query's sparse activations and
  decodes, and direction, which adds the decoded concept direction in raw
  embedding space for queries that are not audio (a text query).
* Reports delta CLAP against the concept text, neighbour preservation against
  the unsteered ranking and the Mahalanobis distance of the edited query, the
  audio-likeness proxy the paper uses to catch off manifold edits.
* Amplify with a positive alpha, suppress ("guitar-free") with a negative one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_attribution import (
    audio_distribution,
    decode_raw,
    invert_adam,
    mahalanobis,
    nearest_neighbour_code,
)
from sae.data import load_track_names
from sae.dclap import DclapTextEncoder
from sae.model import BatchTopKSAE


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def retrieve(corpus, query, top_n):
    scores = unit(corpus) @ unit(query).squeeze(0)
    values, indices = torch.topk(scores, min(top_n, scores.shape[0]))
    return indices.tolist(), values.tolist()


def edit_code(model, query, support, alpha, unit_activation):
    with torch.no_grad():
        code = model.encode(model.normalize(query)).clone()
        for latent in support:
            step = alpha * float(unit_activation[latent])
            code[0, latent] = max(0.0, float(code[0, latent]) + step)
        return decode_raw(model, code)


def edit_direction(model, query, code, alpha):
    with torch.no_grad():
        direction = decode_raw(model, code) - model.denormalize(model.b_pre.unsqueeze(0))
        return query + alpha * unit(direction) * query.norm()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Steer DCLAP retrieval with SAE latents')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data')
    parser.add_argument('--text-model', default=None, help='needed for text queries or concepts')
    parser.add_argument('--tokenizer', default='roberta-base')
    parser.add_argument('--concept', required=True)
    parser.add_argument('--concepts-json', default=None, help='reuse a concept_attribution.py run')
    parser.add_argument('--attribution-method', default='adam')
    parser.add_argument('--query-track', type=int, default=None)
    parser.add_argument('--query-text', default=None)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--top-n', type=int, default=20)
    parser.add_argument('--mode', choices=['auto', 'code', 'direction'], default='auto')
    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--adam-l1', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    if args.query_track is None and not args.query_text:
        parser.error('pass --query-track or --query-text')
    if args.query_text and not args.text_model:
        parser.error('--query-text needs --text-model')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    data_dir = Path(args.data)
    model = BatchTopKSAE.load(args.checkpoint, device=device).eval()

    corpus = torch.from_numpy(np.load(data_dir / 'track_embeddings.npy')).float().to(device)
    corpus = unit(corpus)
    names = load_track_names(data_dir)
    if not names:
        raise SystemExit(f'no track names under {data_dir}, rerun extract_embeddings.py')

    stats_path = Path(args.checkpoint).parent / 'latent_stats.npz'
    if not stats_path.exists():
        raise SystemExit(f'{stats_path} missing, run evaluate_sae.py first')
    unit_activation = np.load(stats_path)['mean_activation']

    text_encoder = None
    if args.text_model:
        text_encoder = DclapTextEncoder(args.text_model, tokenizer_name=args.tokenizer)

    support = None
    if args.concepts_json:
        payload = json.loads(Path(args.concepts_json).read_text())
        for entry in payload:
            if entry['concept'] == args.concept:
                support = entry['methods'][args.attribution_method]['support']
                break
    concept_vector = None
    if text_encoder is not None:
        concept_vector = torch.from_numpy(text_encoder.encode([args.concept])).float().to(device)
    if support is None:
        if concept_vector is None:
            raise SystemExit('need --text-model or --concepts-json to resolve the concept')
        mean, precision = audio_distribution(corpus)
        init, _neighbour = nearest_neighbour_code(model, corpus, concept_vector)
        code = invert_adam(
            model,
            concept_vector,
            init,
            model.cfg.k,
            mean,
            precision,
            args.gamma,
            args.steps,
            args.lr,
            args.adam_l1,
        )
        support = torch.nonzero(code.squeeze(0)).flatten().tolist()
    concept_code = torch.zeros(1, model.cfg.d_sae, device=device)
    for latent in support:
        concept_code[0, latent] = float(unit_activation[latent])

    if args.query_track is not None:
        query = corpus[args.query_track : args.query_track + 1]
        query_label = names[args.query_track]
        default_mode = 'code'
    else:
        query = torch.from_numpy(text_encoder.encode([args.query_text])).float().to(device)
        query_label = f'text: {args.query_text}'
        default_mode = 'direction'
    mode = default_mode if args.mode == 'auto' else args.mode

    if mode == 'code':
        steered = edit_code(model, query, support, args.alpha, unit_activation)
    else:
        steered = edit_direction(model, query, concept_code, args.alpha)

    base_idx, base_scores = retrieve(corpus, query, args.top_n)
    steered_idx, steered_scores = retrieve(corpus, steered, args.top_n)

    mean, precision = audio_distribution(corpus)
    result = {
        'query': query_label,
        'concept': args.concept,
        'mode': mode,
        'alpha': args.alpha,
        'support': support,
        'preserved': len(set(base_idx) & set(steered_idx)) / max(1, len(base_idx)),
        'mahalanobis_query': round(float(mahalanobis(query, mean, precision).mean()), 2),
        'mahalanobis_steered': round(float(mahalanobis(steered, mean, precision).mean()), 2),
        'baseline': [
            {'track': names[i], 'score': round(s, 4)} for i, s in zip(base_idx, base_scores)
        ],
        'steered': [
            {'track': names[i], 'score': round(s, 4)} for i, s in zip(steered_idx, steered_scores)
        ],
    }
    if concept_vector is not None:
        concept_unit = unit(concept_vector).squeeze(0)
        base_cos = float((corpus[base_idx] @ concept_unit).mean())
        steered_cos = float((corpus[steered_idx] @ concept_unit).mean())
        result['concept_cosine_baseline'] = round(base_cos, 5)
        result['concept_cosine_steered'] = round(steered_cos, 5)
        result['delta_clap'] = round(steered_cos - base_cos, 5)

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + '\n', newline='\n')
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
