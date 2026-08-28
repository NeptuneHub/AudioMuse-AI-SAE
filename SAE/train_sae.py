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

"""Trains the BatchTopK sparse autoencoder over extracted DCLAP embeddings.

Follows the recipe in arXiv:2608.08757: AdamW at 1e-4 for 50k steps over an
expansion factor 8 dictionary with the decoder tied to the encoder, sparsity
enforced by the BatchTopK map rather than an L1 penalty. Everything the paper
leaves unspecified follows the standard top-k dictionary recipe, notably the
AuxK dead latent term at alpha = 1/32 and the EMA activation threshold that
makes inference batch independent.

Main Features:
* One run per sparsity level; --k-sweep loops the {5,10,20,50,100} grid the
  paper reports and writes one run directory per value.
* Validation is measured with the exported inference path (EMA threshold), not
  with BatchTopK, so the reported L0 is the one the ONNX graph will produce.
* Normalisation statistics and the pre-encoder bias are derived from the
  training split and stored in the checkpoint, keeping the graph self contained.
* Appends one JSON object per evaluation to train_log.jsonl for later plotting.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae.data import EmbeddingStore
from sae.metrics import dead_latent_fraction, reconstruction_metrics
from sae.model import BatchTopKSAE, SAEConfig


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    if args.lr_decay == 'none':
        return args.lr
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    if args.lr_decay == 'linear':
        return args.lr * max(0.0, 1.0 - progress)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, store, batch_size):
    model.eval()
    rows = store.val_tensor()
    totals = {}
    seen = 0
    for start in range(0, rows.shape[0], batch_size):
        chunk = rows[start : start + batch_size]
        x = model.normalize(chunk)
        acts = model.encode(x)
        recon = model.decode(acts)
        stats = reconstruction_metrics(x, recon, acts)
        weight = chunk.shape[0]
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * weight
        seen += weight
    model.train()
    return {key: value / max(1, seen) for key, value in totals.items()}


def train_one(args, k, out_dir):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    store = EmbeddingStore(
        args.embeddings,
        groups_path=args.groups,
        l2_normalize=not args.no_l2_normalize,
        center=args.center,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        preload=not args.no_preload,
        mmap=args.mmap,
    )
    print(json.dumps(store.summary()))

    d_sae = args.d_sae if args.d_sae else store.d_in * args.expansion
    cfg = SAEConfig(
        d_in=store.d_in,
        d_sae=d_sae,
        k=k,
        tied=not args.untied,
        aux_k=args.aux_k,
        aux_alpha=args.aux_alpha,
        dead_steps=args.dead_steps,
        threshold_beta=args.threshold_beta,
    )
    model = BatchTopKSAE(cfg).to(device)
    model.set_normalization(store.mean.to(device), store.scale)
    with torch.no_grad():
        sample = model.normalize(store.train[: min(65536, store.train.shape[0])].to(device))
        model.init_pre_bias(sample.mean(dim=0))
        if args.init == 'data':
            model.init_from_data(sample)
    model.unit_norm_decoder()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train_log.jsonl'
    log_path.write_text('', newline='\n')

    best = {'fvu': float('inf')}
    started = time.time()
    running = 0.0
    running_aux = 0.0
    running_n = 0

    for step, batch in enumerate(
        store.train_batches(args.batch_size, args.steps, seed=args.seed + 1)
    ):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, args)

        x = model.normalize(batch)
        recon, acts, pre, kept = model(x)
        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        aux_loss, n_dead = model.auxiliary_loss(x, recon, pre)
        loss = recon_loss + cfg.aux_alpha * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.project_decoder_grad()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.unit_norm_decoder()
        model.update_threshold(kept)
        model.update_firing_stats(acts)

        running += recon_loss.item()
        running_aux += float(aux_loss.detach())
        running_n += 1

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = evaluate(model, store, args.batch_size)
            record = {
                'step': step + 1,
                'k': k,
                'lr': lr_at(step, args),
                'train_recon_loss': running / max(1, running_n),
                'train_aux_loss': running_aux / max(1, running_n),
                'dead_latents': n_dead,
                'never_fired': dead_latent_fraction(model.fired_count),
                'threshold': float(model.threshold),
                'elapsed': round(time.time() - started, 1),
            }
            record.update({f'val_{key}': value for key, value in metrics.items()})
            running = running_aux = 0.0
            running_n = 0
            with log_path.open('a', newline='\n') as handle:
                handle.write(json.dumps(record) + '\n')
            print(
                f"step {record['step']:>6} | val fvu {record['val_fvu']:.4f} "
                f"| val cos {record['val_cosine']:.4f} | val L0 {record['val_l0']:.1f} "
                f"| never fired {record['never_fired'] * 100:.1f}% "
                f"| {record['elapsed']:.0f}s"
            )
            if metrics['fvu'] < best['fvu']:
                best = dict(metrics)
                best['step'] = step + 1
                model.save(out_dir / 'sae_best.pt')

    model.save(out_dir / 'sae.pt')
    final = evaluate(model, store, args.batch_size)
    summary = {
        'k': k,
        'd_in': cfg.d_in,
        'd_sae': cfg.d_sae,
        'tied': cfg.tied,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'samples_seen': args.steps * args.batch_size,
        'threshold': float(model.threshold),
        'never_fired': dead_latent_fraction(model.fired_count),
        'elapsed_seconds': round(time.time() - started, 1),
        'data': store.summary(),
        'final': final,
        'best': best,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n', newline='\n')
    print(json.dumps(summary['final'], indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train a BatchTopK SAE on DCLAP embeddings')
    parser.add_argument('--data', default='data', help='directory holding embeddings.npy')
    parser.add_argument('--embeddings', default=None)
    parser.add_argument('--groups', default=None)
    parser.add_argument('--out', default='runs/dclap_sae')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--k-sweep', default=None, help='comma separated k values, one run each')
    parser.add_argument('--expansion', type=int, default=8)
    parser.add_argument('--d-sae', type=int, default=0)
    parser.add_argument('--untied', action='store_true')
    parser.add_argument('--init', choices=['kaiming', 'data'], default='kaiming')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--lr-decay', choices=['none', 'linear', 'cosine'], default='none')
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--aux-k', type=int, default=512)
    parser.add_argument('--aux-alpha', type=float, default=0.03125)
    parser.add_argument('--dead-steps', type=int, default=500)
    parser.add_argument('--threshold-beta', type=float, default=0.999)
    parser.add_argument('--no-l2-normalize', action='store_true')
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--val-fraction', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-preload', action='store_true')
    parser.add_argument('--mmap', action='store_true')
    parser.add_argument('--tf32', action='store_true', default=True)
    args = parser.parse_args(argv)
    data_dir = Path(args.data)
    if args.embeddings is None:
        args.embeddings = str(data_dir / 'embeddings.npy')
    if args.groups is None:
        candidate = data_dir / 'groups.npy'
        args.groups = str(candidate) if candidate.exists() else None
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.k_sweep:
        summaries = []
        for value in [int(v) for v in args.k_sweep.split(',') if v.strip()]:
            print(f'=== training k={value} ===')
            summaries.append(train_one(args, value, Path(args.out) / f'k{value}'))
        sweep_path = Path(args.out) / 'sweep.json'
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps(summaries, indent=2) + '\n', newline='\n')
    else:
        train_one(args, args.k, args.out)


if __name__ == '__main__':
    main()
