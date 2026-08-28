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

"""Decides which steering concepts are actually grounded in a given library.

The paper offers open vocabulary steering but validates nothing, and admits that
concepts which are "weakly grounded, rare, abstract, or diffusely represented"
simply fail. Silent failure is the worst outcome for a UI, so this script scores
every candidate term against the real corpus and emits only the ones that work.

Equation 5 of the paper is what keeps a concept specific: the inverted code is
weighted by the inverse document frequency of each latent and only the top
coordinates survive, so the handful of neurons that fire on half the library
cannot end up representing a term. Keeping every non-zero coordinate instead
silently skips that penalty and lets broad terms absorb everything.

The score is an agreement test. For a concept C, take the tracks CLAP itself
ranks highest for C, and take the tracks that most strongly activate the SAE
latents the inversion assigned to C. If the dictionary really holds a detector
for C those two sets overlap; if the inversion merely grabbed whatever was
nearest, they do not.

Main Features:
* Emits concepts_validated.json: category, term, grounding score, support
  latents and example tracks, ready to drive a refinement UI.
* Precomputes each surviving concept's decoded steering direction into an npz,
  so the serving path never runs the optimisation at request time.
* Reports the rejected terms too, so a library gap is visible rather than hidden.
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
    invert_adam,
    nearest_neighbour_code,
    support_of,
)
from sae.data import load_track_names
from sae.dclap import DclapTextEncoder
from sae.model import BatchTopKSAE


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def read_candidates(path):
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        category, _, term = line.partition('|')
        if not term:
            category, term = 'general', category
        items.append((category.strip(), term.strip()))
    return items


@torch.no_grad()
def clap_top(corpus, text, top_k):
    return torch.topk(corpus @ text.squeeze(0), top_k).indices


@torch.no_grad()
def latent_top(model, corpus, support, top_k, batch_size=16384):
    if not support:
        return torch.zeros(0, dtype=torch.long, device=corpus.device)
    index = torch.as_tensor(support, device=corpus.device)
    scores = torch.empty(corpus.shape[0], device=corpus.device)
    for start in range(0, corpus.shape[0], batch_size):
        chunk = corpus[start : start + batch_size]
        acts = model.encode(model.normalize(chunk))
        scores[start : start + chunk.shape[0]] = acts[:, index].sum(dim=-1)
    return torch.topk(scores, top_k).indices


def steering_direction(model, code):
    with torch.no_grad():
        decoded = model.denormalize(model.decode(code))
        centre = model.denormalize(model.b_pre.unsqueeze(0))
        return unit(decoded - centre).squeeze(0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Validate steering concepts against a library')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text-model', required=True)
    parser.add_argument('--tokenizer', default='roberta-base')
    parser.add_argument('--data', default='data')
    parser.add_argument('--stats', default=None)
    parser.add_argument('--candidates', default='candidate_concepts.txt')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--min-grounding', type=float, default=0.10)
    parser.add_argument('--invert-k', type=int, default=0)
    parser.add_argument('--support-size', type=int, default=12)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--adam-l1', type=float, default=1e-4)
    parser.add_argument('--examples', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='runs/concepts_validated.json')
    parser.add_argument('--out-npz', default='runs/concept_directions.npz')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    data_dir = Path(args.data)
    model = BatchTopKSAE.load(args.checkpoint, device=device).eval()
    invert_k = args.invert_k or model.cfg.k

    corpus = unit(torch.from_numpy(np.load(data_dir / 'track_embeddings.npy')).float().to(device))
    names = load_track_names(data_dir)

    stats_path = Path(args.stats) if args.stats else Path(args.checkpoint).parent / 'latent_stats.npz'
    stats = np.load(stats_path)
    alive = torch.from_numpy(stats['df']).float().to(device) > 0
    idf = torch.from_numpy(stats['idf']).float().to(device).clamp_min(0.0) * alive.float()
    frequency = stats['frequency']

    mean, precision = audio_distribution(corpus)
    encoder = DclapTextEncoder(args.text_model, tokenizer_name=args.tokenizer)
    candidates = read_candidates(args.candidates)
    print(f'scoring {len(candidates)} candidate concepts against {corpus.shape[0]} tracks')

    accepted = []
    rejected = []
    directions = {}
    for position, (category, term) in enumerate(candidates, 1):
        text = torch.from_numpy(encoder.encode([term])).float().to(device)
        init, _neighbour = nearest_neighbour_code(model, corpus, text)
        code = invert_adam(
            model,
            text,
            init,
            invert_k,
            mean,
            precision,
            args.gamma,
            args.steps,
            args.lr,
            l1=args.adam_l1,
        )
        support, _weights = support_of(code, idf, args.support_size)
        masked = torch.zeros_like(code)
        if support:
            index = torch.as_tensor(support, device=code.device)
            masked[0, index] = code[0, index]

        hits = set(clap_top(corpus, text, args.top_k).tolist())
        fires = latent_top(model, corpus, support, args.top_k).tolist()
        grounding = len(hits & set(fires)) / float(args.top_k) if support else 0.0

        entry = {
            'category': category,
            'term': term,
            'grounding': round(grounding, 4),
            'support': support,
            'support_values': [round(float(code[0, i]), 6) for i in support],
            'support_frequency': [round(float(frequency[i]), 5) for i in support],
            'examples': [names[i] for i in fires[: args.examples]] if names else [],
        }
        if grounding >= args.min_grounding and support:
            accepted.append(entry)
            directions[term] = steering_direction(model, masked).cpu().numpy()
        else:
            rejected.append(entry)
        print(f'[{position:>3}/{len(candidates)}] {grounding:.3f} {category}/{term}')

    by_category = {}
    for entry in sorted(accepted, key=lambda e: (e['category'], -e['grounding'])):
        by_category.setdefault(entry['category'], []).append(entry)
    payload = {
        'checkpoint': str(args.checkpoint),
        'corpus_tracks': int(corpus.shape[0]),
        'top_k': args.top_k,
        'min_grounding': args.min_grounding,
        'accepted': len(accepted),
        'rejected': len(rejected),
        'categories': by_category,
        'rejected_terms': sorted(
            [
                {'category': e['category'], 'term': e['term'], 'grounding': e['grounding']}
                for e in rejected
            ],
            key=lambda e: -e['grounding'],
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n', newline='\n')

    npz_path = Path(args.out_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if directions:
        np.savez_compressed(
            npz_path,
            terms=np.array(list(directions), dtype=object),
            directions=np.stack([directions[t] for t in directions]).astype(np.float32),
        )
    print(f'\n{len(accepted)} accepted, {len(rejected)} rejected')
    print(f'wrote {out} and {npz_path}')


if __name__ == '__main__':
    main()
