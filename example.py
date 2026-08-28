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

"""Turns a text query into a DCLAP embedding and enforces a concept on it.

Run it with no arguments to see the worked example from the README: the query
"POP Viola with Female vocalist" is embedded, the concept "viola" is amplified
inside the sparse autoencoder, and the edited embedding is printed next to the
original. Use the second embedding as the query vector for your search.

Each concept is a unit norm mask over its latents, so a strength setting means
the same step for every concept. Only the difference between the edited and the
unedited reconstruction is applied to the query, because the autoencoder does not
round trip a text embedding exactly and returning the raw reconstruction would
change most of the results before any concept was touched.

Main Features:
* Embeds free text with the CLAP text tower, exactly as DCLAP search does.
* Amplifies or suppresses any concept in the catalogue, at any strength.
* Prints both embeddings and how far the edit moved the query.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

TEXT_MAX_LENGTH = 77


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Enforce an SAE concept on a text query')
    parser.add_argument('--models', default='models', help='folder holding the downloaded files')
    parser.add_argument('--query', default='POP Viola with Female vocalist')
    parser.add_argument('--concept', default='viola')
    parser.add_argument('--strength', type=float, default=5.0, help='1, 3, 5 or 10')
    parser.add_argument('--direction', choices=['more', 'less'], default='more')
    return parser.parse_args(argv)


def embed_text(models, query):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    session = ort.InferenceSession(str(models / 'clap_text_model.onnx'))
    encoded = tokenizer(
        [query],
        max_length=TEXT_MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='np',
    )
    inputs = {
        'input_ids': encoded['input_ids'].astype(np.int64),
        'attention_mask': encoded['attention_mask'].astype(np.int64),
    }
    vector = session.run([session.get_outputs()[0].name], inputs)[0].astype(np.float32)
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)


def enforce(models, embedding, term, strength, direction):
    catalogue = json.loads((models / 'dclap_sae_concepts.json').read_text())
    concepts = {c['term']: c for c in catalogue['concepts']}
    if term not in concepts:
        raise SystemExit(
            f'"{term}" is not in the catalogue. Available terms include: '
            + ', '.join(sorted(concepts)[:12])
            + ' ...'
        )
    concept = concepts[term]

    encoder = ort.InferenceSession(str(models / 'dclap_sae_k20_d1024_best_encoder.onnx'))
    decoder = ort.InferenceSession(str(models / 'dclap_sae_k20_d1024_best_decoder.onnx'))

    original = encoder.run(None, {encoder.get_inputs()[0].name: embedding})[0].astype(np.float32)
    edited = original.copy()
    index = np.asarray(concept['support'], dtype=np.int64)
    step = np.asarray(concept['mask'], dtype=np.float32) * strength
    if direction == 'less':
        step = -step
    edited[0, index] = np.maximum(0.0, edited[0, index] + step)

    pair = np.concatenate([original, edited], axis=0)
    decoded = decoder.run(None, {decoder.get_inputs()[0].name: pair})[0].astype(np.float32)

    steered = embedding[0] + (decoded[1] - decoded[0])
    steered = steered / np.linalg.norm(steered)
    return steered.reshape(1, -1), concept, original, edited


def show(label, vector):
    flat = vector.reshape(-1)
    head = ' '.join(f'{v:+.4f}' for v in flat[:8])
    print(f'{label}\n  [{head} ...]  ({flat.size} values, norm {np.linalg.norm(flat):.4f})')


def main(argv=None):
    args = parse_args(argv)
    models = Path(args.models)
    required = [
        'clap_text_model.onnx',
        'dclap_sae_k20_d1024_best_encoder.onnx',
        'dclap_sae_k20_d1024_best_decoder.onnx',
        'dclap_sae_concepts.json',
    ]
    missing = [name for name in required if not (models / name).exists()]
    if missing:
        raise SystemExit(
            f'missing from {models}/: {", ".join(missing)}\n'
            'See the Quick start in README.md for the download commands.'
        )

    print(f'query   : "{args.query}"')
    print(f'enforce : {args.direction} {args.concept} at strength {args.strength}\n')

    embedding = embed_text(models, args.query)
    steered, concept, original, edited = enforce(
        models, embedding, args.concept, args.strength, args.direction
    )

    show('ORIGINAL query embedding', embedding)
    print()
    show('NEW query embedding after enforcing the concept', steered)

    moved = float(np.dot(embedding.reshape(-1), steered.reshape(-1)))
    changed = int((np.abs(edited - original) > 1e-6).sum())
    print(
        f'\nconcept "{concept["term"]}" uses {len(concept["support"])} of '
        f'{original.shape[1]} latents, {changed} of them were changed'
    )
    print(f'cosine(original, new) = {moved:.4f}   (1.0 would mean no change)')
    print('\nSearch your index with the NEW embedding to get the refined results.')


if __name__ == '__main__':
    main()
