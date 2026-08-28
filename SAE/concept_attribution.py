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

"""Maps a free form text concept onto a support of SAE latents.

This is the contribution of arXiv:2608.08757. Naively probing the dictionary
with the cosine similarity between a decoder column and a CLAP text embedding
(the Discover-Then-Name recipe) picks latents that look text aligned but are not
audio faithful, because text and audio occupy different cones of the CLAP space.
Instead the concept is INVERTED through the decoder: find the sparse code whose
reconstruction points at the text embedding while staying inside the empirical
audio distribution, then read its support.

Main Features:
* Equation 4 solved two ways, Adam and FISTA, both initialised from the sparse
  code of the nearest audio neighbour of the text embedding.
* Mahalanobis term Gamma keeps the reconstruction on the audio manifold, with
  the shrinkage regularised covariance estimated from the training corpus.
* Equation 5 IDF reweighting w_k = log(N / (df(k) + eps)) demotes latents that
  fire on almost everything before the support is read off. Latents with df = 0
  are masked out first: the same formula would otherwise hand a dead latent the
  largest possible weight and let it dominate every support.
* The cosine probing baseline is kept for comparison so the two supports can be
  diffed on the same concepts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae.data import load_track_names
from sae.dclap import DclapTextEncoder
from sae.model import BatchTopKSAE


def audio_distribution(embeddings, shrinkage=0.05):
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    cov = (centered.t() @ centered) / max(1, centered.shape[0] - 1)
    trace = torch.diagonal(cov).sum() / cov.shape[0]
    cov = (1.0 - shrinkage) * cov + shrinkage * trace * torch.eye(
        cov.shape[0], device=cov.device, dtype=cov.dtype
    )
    return mean, torch.linalg.inv(cov)


def mahalanobis(y, mean, precision):
    delta = y - mean
    return ((delta @ precision) * delta).sum(dim=-1)


def sparsify(u, k, straight_through=False):
    positive = torch.relu(u)
    if k >= positive.shape[-1]:
        return positive
    values, indices = torch.topk(positive, k, dim=-1)
    sparse = torch.zeros_like(positive).scatter(-1, indices, values)
    if not straight_through:
        return sparse
    return positive + (sparse - positive).detach()


def decode_raw(model, code):
    return model.denormalize(model.decode(code))


def objective(model, code, target, mean, precision, gamma):
    decoded = decode_raw(model, code)
    cosine = torch.nn.functional.cosine_similarity(decoded, target, dim=-1)
    return (1.0 - cosine) + gamma * mahalanobis(decoded, mean, precision), decoded


def nearest_neighbour_code(model, corpus, target):
    normalized = corpus / corpus.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    query = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    best = int(torch.argmax(normalized @ query.squeeze(0)))
    with torch.no_grad():
        pre = model.pre_activation(model.normalize(corpus[best : best + 1]))
    return pre.clone(), best


def invert_adam(model, target, init, k, mean, precision, gamma, steps, lr, l1=0.0):
    u = init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        code = sparsify(u, k, straight_through=True)
        loss, _decoded = objective(model, code, target, mean, precision, gamma)
        if l1 > 0.0:
            loss = loss + l1 * torch.relu(u).sum(dim=-1)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    return sparsify(u.detach(), k)


def invert_fista(model, target, init, k, mean, precision, gamma, steps, lr, l1):
    u = init.clone()
    y = u.clone()
    t = 1.0
    for _ in range(steps):
        y = y.detach().requires_grad_(True)
        loss, _decoded = objective(model, torch.relu(y), target, mean, precision, gamma)
        (grad,) = torch.autograd.grad(loss.sum(), y)
        candidate = torch.relu(y.detach() - lr * grad - lr * l1)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = candidate + ((t - 1.0) / t_next) * (candidate - u)
        u = candidate
        t = t_next
    return sparsify(u, k)


def cosine_probe(model, target, k):
    dictionary = model.dictionary()
    scale = float(model.norm_scale)
    direction = target.squeeze(0) * scale
    scores = dictionary.t() @ (direction / direction.norm().clamp_min(1e-8))
    values, indices = torch.topk(torch.relu(scores), k)
    code = torch.zeros(1, model.cfg.d_sae, device=target.device)
    code[0, indices] = values
    return code


def apply_idf(code, idf):
    return code * idf.unsqueeze(0)


def support_of(code, idf, size):
    weighted = apply_idf(code, idf)
    take = min(size, int((weighted > 0).sum()))
    if take == 0:
        return [], []
    values, indices = torch.topk(weighted.squeeze(0), take)
    return indices.tolist(), values.tolist()


def latent_examples(latent, stats, names, limit=4):
    if names is None or 'top_track_ids' not in stats:
        return []
    picks = []
    seen = set()
    for track, value in zip(stats['top_track_ids'][latent], stats['top_values'][latent]):
        if value <= 0 or int(track) in seen or int(track) >= len(names):
            continue
        seen.add(int(track))
        picks.append(names[int(track)])
        if len(picks) >= limit:
            break
    return picks


def describe_support(indices, stats, frequency, names):
    return [
        {
            'latent': int(latent),
            'frequency': round(float(frequency[latent]), 5),
            'top_tracks': latent_examples(int(latent), stats, names),
        }
        for latent in indices
    ]


def evaluate_support(model, corpus, indices, target):
    if not indices:
        return {}
    with torch.no_grad():
        acts = model.encode(model.normalize(corpus))
        mask = torch.zeros(model.cfg.d_sae, device=corpus.device, dtype=torch.bool)
        mask[torch.tensor(indices, device=corpus.device)] = True
        hit = (acts[:, mask] > 0).any(dim=1).float()
        overlap = (acts[:, mask] > 0).float().sum(dim=1)
        direct = torch.nn.functional.cosine_similarity(
            corpus / corpus.norm(dim=-1, keepdim=True).clamp_min(1e-8),
            target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8),
            dim=-1,
        )
        top = torch.topk(direct, min(50, direct.shape[0])).indices
    return {
        'corpus_hit_rate': float(hit.mean()),
        'mean_active_in_support': float(overlap.mean()),
        'hit_rate_on_text_top50': float(hit[top].mean()),
        'mean_active_in_support_top50': float(overlap[top].mean()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Attribute text concepts to SAE latents')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--text-model', required=True, help='CLAP text tower ONNX')
    parser.add_argument('--tokenizer', default='roberta-base')
    parser.add_argument('--data', default='data')
    parser.add_argument('--stats', default=None, help='latent_stats.npz from evaluate_sae.py')
    parser.add_argument('--concept', action='append', default=[])
    parser.add_argument('--concepts-file', default=None)
    parser.add_argument('--method', choices=['adam', 'fista', 'cosine', 'all'], default='all')
    parser.add_argument('--support-size', type=int, default=0, help='defaults to the SAE k')
    parser.add_argument('--gamma', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--adam-l1', type=float, default=1e-4)
    parser.add_argument('--fista-lr', type=float, default=0.05)
    parser.add_argument('--fista-l1', type=float, default=1e-3)
    parser.add_argument('--corpus', choices=['tracks', 'segments'], default='tracks')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='runs/concepts.json')
    args = parser.parse_args(argv)
    if args.concepts_file:
        lines = Path(args.concepts_file).read_text().splitlines()
        args.concept.extend([line.strip() for line in lines if line.strip()])
    if not args.concept:
        parser.error('pass at least one --concept or a --concepts-file')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    data_dir = Path(args.data)
    model = BatchTopKSAE.load(args.checkpoint, device=device).eval()
    support_size = args.support_size or model.cfg.k

    corpus_file = 'track_embeddings.npy' if args.corpus == 'tracks' else 'embeddings.npy'
    corpus = torch.from_numpy(np.load(data_dir / corpus_file)).float().to(device)
    corpus = corpus / corpus.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    stats_path = Path(args.stats) if args.stats else Path(args.checkpoint).parent / 'latent_stats.npz'
    if not stats_path.exists():
        raise SystemExit(f'{stats_path} missing, run evaluate_sae.py first')
    stats = np.load(stats_path)
    frequency = stats['frequency']
    alive = torch.from_numpy(stats['df']).float().to(device) > 0
    idf = torch.from_numpy(stats['idf']).float().to(device).clamp_min(0.0) * alive.float()

    names = load_track_names(data_dir)

    mean, precision = audio_distribution(corpus)
    encoder = DclapTextEncoder(args.text_model, tokenizer_name=args.tokenizer)
    text = torch.from_numpy(encoder.encode(args.concept)).float().to(device)

    methods = ['adam', 'fista', 'cosine'] if args.method == 'all' else [args.method]
    results = []
    for row, concept in enumerate(args.concept):
        target = text[row : row + 1]
        init, neighbour = nearest_neighbour_code(model, corpus, target)
        entry = {'concept': concept, 'nearest_corpus_row': neighbour, 'methods': {}}
        for method in methods:
            if method == 'cosine':
                code = cosine_probe(model, target, support_size)
            elif method == 'adam':
                code = invert_adam(
                    model,
                    target,
                    init,
                    model.cfg.k,
                    mean,
                    precision,
                    args.gamma,
                    args.steps,
                    args.lr,
                    args.adam_l1,
                )
            else:
                code = invert_fista(
                    model,
                    target,
                    init,
                    model.cfg.k,
                    mean,
                    precision,
                    args.gamma,
                    args.steps,
                    args.fista_lr,
                    args.fista_l1,
                )
            indices, weights = support_of(code, idf, support_size)
            with torch.no_grad():
                decoded = decode_raw(model, code)
                cosine = float(
                    torch.nn.functional.cosine_similarity(decoded, target, dim=-1).mean()
                )
                distance = float(mahalanobis(decoded, mean, precision).mean())
            entry['methods'][method] = {
                'support': indices,
                'idf_weights': [round(w, 4) for w in weights],
                'cosine_to_text': round(cosine, 4),
                'mahalanobis': round(distance, 2),
                'latents': describe_support(indices, stats, frequency, names),
                'audio_evidence': evaluate_support(model, corpus, indices, target),
            }
        if len(methods) > 1 and 'adam' in entry['methods'] and 'cosine' in entry['methods']:
            a = set(entry['methods']['adam']['support'])
            c = set(entry['methods']['cosine']['support'])
            entry['adam_vs_cosine_jaccard'] = round(
                len(a & c) / max(1, len(a | c)), 4
            )
        results.append(entry)
        print(json.dumps(entry, indent=2)[:2000])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + '\n', newline='\n')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
