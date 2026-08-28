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

"""Exports a trained BatchTopK SAE checkpoint to ONNX.

The exported graph consumes RAW DCLAP embeddings: the affine normalisation, the
pre-encoder bias and the EMA activation threshold are all baked in, so a runtime
only has to feed the same 512 dim vectors that tasks/clap_analyzer.py already
produces. Sparsity is applied as a fixed threshold rather than BatchTopK, which
is what makes the graph independent of batch size and therefore exportable.

Main Features:
* Three graph shapes: full (latents plus reconstruction), encoder only and
  decoder only, the last one being what steering needs to map an edited sparse
  code back into embedding space.
* Batch axis is dynamic, so one graph serves a single track or a whole library.
* Verifies the export by replaying random and real embeddings through both
  PyTorch and onnxruntime and reporting the maximum absolute difference.
* Writes a sidecar JSON with dimensions, k, threshold and normalisation values.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae.model import BatchTopKSAE, SAEInference

OUTPUT_NAMES = {
    'full': ['latents', 'reconstruction'],
    'encoder': ['latents'],
    'decoder': ['reconstruction'],
}


def export(model, part, path, opset, batch=4):
    module = SAEInference(model, part=part).eval()
    width = model.cfg.d_sae if part == 'decoder' else model.cfg.d_in
    dummy = torch.randn(batch, width, dtype=torch.float32)
    if part != 'decoder':
        dummy = dummy / dummy.norm(dim=-1, keepdim=True)
    else:
        dummy = dummy.abs() * (torch.rand_like(dummy) < 0.01)
    input_name = 'latents' if part == 'decoder' else 'embedding'
    output_names = OUTPUT_NAMES[part]
    dynamic_axes = {input_name: {0: 'batch'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch'}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (dummy,),
        str(path),
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return module, dummy, input_name, output_names


def verify(module, path, sample, input_name, output_names, tolerance):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    onnx_outputs = session.run(output_names, {input_name: sample.numpy()})
    with torch.no_grad():
        torch_outputs = module(sample)
    if not isinstance(torch_outputs, (tuple, list)):
        torch_outputs = (torch_outputs,)

    report = {}
    worst = 0.0
    for name, torch_out, onnx_out in zip(output_names, torch_outputs, onnx_outputs):
        diff = float(np.abs(torch_out.numpy() - onnx_out).max())
        report[name] = diff
        worst = max(worst, diff)
    if worst > tolerance:
        raise SystemExit(f'ONNX output diverges from PyTorch by {worst:.3e} (limit {tolerance:.1e})')
    return report


def real_sample(embeddings_path, limit, d_in):
    if not embeddings_path or not Path(embeddings_path).exists():
        return None
    matrix = np.load(embeddings_path, mmap_mode='r')
    if matrix.shape[1] != d_in:
        return None
    rows = np.ascontiguousarray(matrix[:limit]).astype(np.float32)
    rows = rows / np.linalg.norm(rows, axis=-1, keepdims=True).clip(1e-8)
    return torch.from_numpy(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Export a trained SAE to ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', default='model')
    parser.add_argument('--name', default='dclap_sae')
    parser.add_argument(
        '--parts',
        default='full,encoder,decoder',
        help='comma separated subset of full,encoder,decoder',
    )
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--embeddings', default='data/embeddings.npy')
    parser.add_argument('--verify-rows', type=int, default=512)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = BatchTopKSAE.load(args.checkpoint, device='cpu').eval()
    if float(model.threshold) <= 0.0:
        raise SystemExit('checkpoint has no positive activation threshold, retrain before export')

    sidecar = {
        'name': args.name,
        'checkpoint': str(args.checkpoint),
        'd_in': model.cfg.d_in,
        'd_sae': model.cfg.d_sae,
        'k': model.cfg.k,
        'tied': model.cfg.tied,
        'threshold': float(model.threshold),
        'norm_scale': float(model.norm_scale),
        'norm_mean_is_zero': bool(torch.count_nonzero(model.norm_mean) == 0),
        'opset': args.opset,
        'inputs_are_raw_dclap_embeddings': True,
        'graphs': {},
    }

    sample_real = real_sample(args.embeddings, args.verify_rows, model.cfg.d_in)
    for part in [p.strip() for p in args.parts.split(',') if p.strip()]:
        if part not in OUTPUT_NAMES:
            raise SystemExit(f'unknown part {part}')
        path = Path(args.out_dir) / f'{args.name}_{part}.onnx'
        module, dummy, input_name, output_names = export(model, part, path, args.opset)
        checks = {'random': verify(module, path, dummy, input_name, output_names, args.tolerance)}
        if part != 'decoder' and sample_real is not None:
            checks['real'] = verify(
                module, path, sample_real, input_name, output_names, args.tolerance
            )
        size_mb = round(path.stat().st_size / (1024 * 1024), 3)
        sidecar['graphs'][part] = {
            'file': path.name,
            'input': input_name,
            'outputs': output_names,
            'size_mb': size_mb,
            'max_abs_diff': checks,
        }
        print(f'{path} ({size_mb} MB) max abs diff {checks}')

    if sample_real is not None:
        with torch.no_grad():
            latents = model.encode(model.normalize(sample_real))
        sidecar['sanity'] = {
            'rows': int(sample_real.shape[0]),
            'mean_l0': float((latents > 0).float().sum(dim=-1).mean()),
            'active_latents': int((latents > 0).any(dim=0).sum()),
        }

    sidecar_path = Path(args.out_dir) / f'{args.name}.json'
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + '\n', newline='\n')
    print(f'wrote {sidecar_path}')


if __name__ == '__main__':
    main()
