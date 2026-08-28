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

"""Scores a trained SAE and dumps the per latent statistics used downstream.

Reports the usual dictionary numbers (L0, FVU, explained variance, cosine) but
also the one that decides whether this thing is safe to put near AudioMuse: the
neighbour overlap between a retrieval index built on the original DCLAP track
vectors and one built on their SAE reconstructions. A dictionary that halves
recall@10 is not usable for steering no matter how good its FVU looks.

Main Features:
* Chunked pass over the corpus keeping a running top-N of activating segments
  per latent, so latents can be read by listening to the tracks that fire them.
* Document frequency df(k) and the IDF weights w_k = log(N / (df(k) + eps)) of
  equation 5, saved for concept_attribution.py.
* Neighbour preservation: recall@k overlap and Spearman style rank agreement
  between original and reconstructed track embeddings.
* Writes report.json plus latent_stats.npz next to the checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae.data import EmbeddingStore, load_track_names
from sae.metrics import reconstruction_metrics
from sae.model import BatchTopKSAE


@torch.no_grad()
def full_metrics(model, rows, batch_size):
    totals = {}
    seen = 0
    for start in range(0, rows.shape[0], batch_size):
        chunk = rows[start : start + batch_size]
        x = model.normalize(chunk)
        acts = model.encode(x)
        stats = reconstruction_metrics(x, model.decode(acts), acts)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * chunk.shape[0]
        seen += chunk.shape[0]
    return {key: value / max(1, seen) for key, value in totals.items()}


@torch.no_grad()
def latent_statistics(model, rows, batch_size, top_tracks):
    d_sae = model.cfg.d_sae
    device = rows.device
    df = torch.zeros(d_sae, dtype=torch.float64, device=device)
    activation_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    best_values = torch.full((d_sae, top_tracks), -1.0, device=device)
    best_index = torch.zeros((d_sae, top_tracks), dtype=torch.long, device=device)

    for start in range(0, rows.shape[0], batch_size):
        chunk = rows[start : start + batch_size]
        acts = model.encode(model.normalize(chunk))
        active = acts > 0
        df += active.sum(dim=0).double()
        activation_sum += acts.sum(dim=0).double()

        transposed = acts.t().contiguous()
        take = min(top_tracks, transposed.shape[1])
        values, indices = torch.topk(transposed, take, dim=1)
        indices = indices + start
        merged_values = torch.cat([best_values, values], dim=1)
        merged_index = torch.cat([best_index, indices], dim=1)
        best_values, order = torch.topk(merged_values, top_tracks, dim=1)
        best_index = torch.gather(merged_index, 1, order)

    n_rows = float(rows.shape[0])
    return {
        'df': df.cpu().numpy(),
        'frequency': (df / n_rows).cpu().numpy(),
        'mean_activation': (activation_sum / df.clamp_min(1.0)).cpu().numpy(),
        'top_values': best_values.cpu().numpy(),
        'top_rows': best_index.cpu().numpy(),
    }


@torch.no_grad()
def reconstruct(model, rows, batch_size):
    out = torch.empty_like(rows)
    for start in range(0, rows.shape[0], batch_size):
        chunk = rows[start : start + batch_size]
        out[start : start + batch_size] = model.denormalize(
            model.decode(model.encode(model.normalize(chunk)))
        )
    return out


@torch.no_grad()
def neighbour_preservation(model, tracks, top_k, batch_size, chunk_size=256, queries=0, seed=0):
    original = tracks / tracks.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    recon = reconstruct(model, tracks, batch_size)
    recon = recon / recon.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    n = original.shape[0]
    take = min(top_k, max(1, n - 1))
    if queries and queries < n:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        probe = torch.randperm(n, generator=generator)[:queries].to(original.device)
    else:
        probe = torch.arange(n, device=original.device)

    overlap = 0.0
    top1 = 0.0
    for start in range(0, probe.shape[0], chunk_size):
        rows = probe[start : start + chunk_size]
        local = torch.arange(rows.shape[0], device=original.device)

        sim = original[rows] @ original.t()
        sim[local, rows] = -2.0
        idx_a = torch.topk(sim, take, dim=1).indices
        del sim

        sim = recon[rows] @ recon.t()
        sim[local, rows] = -2.0
        idx_b = torch.topk(sim, take, dim=1).indices
        del sim

        matches = (idx_a.unsqueeze(2) == idx_b.unsqueeze(1)).any(dim=2).float().sum(dim=1)
        overlap += float(matches.sum())
        top1 += float((idx_a[:, 0] == idx_b[:, 0]).float().sum())

    n_probe = int(probe.shape[0])
    return {
        'top_k': int(take),
        'catalogue': int(n),
        'queries': n_probe,
        'neighbour_overlap': overlap / (n_probe * take),
        'top1_agreement': top1 / n_probe,
        'mean_cosine_original_vs_reconstruction': float(
            torch.nn.functional.cosine_similarity(original, recon, dim=-1).mean()
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a trained DCLAP SAE')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='data')
    parser.add_argument('--split', choices=['val', 'train', 'all'], default='val')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--top-tracks', type=int, default=8)
    parser.add_argument('--examples', type=int, default=128)
    parser.add_argument('--neighbour-k', type=int, default=10)
    parser.add_argument('--neighbour-queries', type=int, default=20000)
    parser.add_argument('--neighbour-chunk', type=int, default=256)
    parser.add_argument('--val-fraction', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data)
    device = torch.device(args.device)
    model = BatchTopKSAE.load(args.checkpoint, device=device).eval()

    groups = data_dir / 'groups.npy'
    store = EmbeddingStore(
        data_dir / 'embeddings.npy',
        groups_path=str(groups) if groups.exists() else None,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
    if args.split == 'val':
        rows = store.val_tensor()
        row_groups = store.val_groups
    elif args.split == 'train':
        rows = store.train.to(device)
        row_groups = store.train_groups
    else:
        rows = torch.cat([store.train.to(device), store.val_tensor()], dim=0)
        row_groups = None
        if store.train_groups is not None:
            row_groups = np.concatenate([store.train_groups, store.val_groups])

    report = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'rows': int(rows.shape[0]),
        'd_in': model.cfg.d_in,
        'd_sae': model.cfg.d_sae,
        'k': model.cfg.k,
        'threshold': float(model.threshold),
        'reconstruction': full_metrics(model, rows, args.batch_size),
    }

    stats = latent_statistics(model, rows, args.batch_size, args.top_tracks)
    n_rows = int(rows.shape[0])
    del rows
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    frequency = stats['frequency']
    report['latents'] = {
        'never_active': int((stats['df'] == 0).sum()),
        'active_fraction': float((stats['df'] > 0).mean()),
        'median_frequency': float(np.median(frequency[frequency > 0])) if (frequency > 0).any() else 0.0,
        'max_frequency': float(frequency.max()),
        'frequency_above_10pct': int((frequency > 0.10).sum()),
    }

    track_path = data_dir / 'track_embeddings.npy'
    if track_path.exists():
        tracks = torch.from_numpy(np.load(track_path)).float().to(device)
        report['retrieval'] = neighbour_preservation(
            model,
            tracks,
            args.neighbour_k,
            args.batch_size,
            chunk_size=args.neighbour_chunk,
            queries=args.neighbour_queries,
            seed=args.seed,
        )

    names = load_track_names(data_dir)
    examples = []
    if row_groups is not None and names is not None:
        live = np.nonzero(frequency > 0)[0]
        ranked = live[np.argsort(-frequency[live])]
        head = ranked[: args.examples // 2]
        rest = ranked[args.examples // 2 :]
        chosen = head
        if rest.size:
            take = min(rest.size, args.examples - head.size)
            spread = rest[np.linspace(0, rest.size - 1, take).astype(int)]
            chosen = np.concatenate([head, spread])
        for latent in chosen:
            rows_for_latent = stats['top_rows'][latent]
            values = stats['top_values'][latent]
            picks = []
            seen = set()
            for row, value in zip(rows_for_latent, values):
                if value <= 0:
                    continue
                track = int(row_groups[int(row)])
                if track in seen:
                    continue
                seen.add(track)
                picks.append({'track': names[track], 'activation': round(float(value), 4)})
            examples.append(
                {
                    'latent': int(latent),
                    'frequency': round(float(frequency[latent]), 5),
                    'top_tracks': picks,
                }
            )
    report['top_latent_examples'] = examples

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-9
    payload = {
        'df': stats['df'],
        'frequency': frequency,
        'mean_activation': stats['mean_activation'],
        'idf': np.log(float(n_rows) / (stats['df'] + eps)),
        'top_rows': stats['top_rows'],
        'top_values': stats['top_values'],
    }
    if row_groups is not None:
        payload['top_track_ids'] = np.asarray(row_groups)[stats['top_rows']]
    np.savez_compressed(out_dir / 'latent_stats.npz', **payload)
    (out_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', newline='\n')
    print(json.dumps({key: report[key] for key in ('reconstruction', 'latents')}, indent=2))
    if 'retrieval' in report:
        print(json.dumps(report['retrieval'], indent=2))
    print(f'wrote {out_dir / "report.json"} and {out_dir / "latent_stats.npz"}')


if __name__ == '__main__':
    main()
