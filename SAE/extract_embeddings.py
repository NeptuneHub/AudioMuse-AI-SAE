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

"""Builds the DCLAP embedding corpus the sparse autoencoder is trained on.

Two sources are supported. The default one streams an existing AudioMuse-AI
Postgres instance, whose clap_embedding table already holds one mean pooled and
L2 normalised 512 dim vector per track, so a library of hundreds of thousands of
tracks becomes a training corpus without decoding a single audio file. The other
one decodes a local audio library through the DCLAP ONNX audio tower exactly the
way tasks/clap_analyzer.py does it, which yields one embedding per 10 second
segment instead of one per track.

Main Features:
* Postgres mode uses a named server side cursor and a preallocated array, so a
  large table is materialised once at its final size rather than buffered as
  Python objects. Table and column names are arguments, and only the row id and
  the embedding are read, so no titles or other metadata ever leave the database.
* Audio mode runs a process pool over tracks, one ONNX session per worker, with
  sharded output so a long extraction can be interrupted and resumed.
* Both emit the same file set: embeddings.npy, groups.npy (track id per row),
  track_embeddings.npy, track_names.json and meta.json, so train_sae.py does not
  care where the corpus came from.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sae.dclap import EMBEDDING_DIM, DclapAudioEncoder

AUDIO_SUFFIXES = {'.mp3', '.flac', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.wma'}

_WORKER = {}


def _init_worker(model_path, providers):
    _WORKER['encoder'] = DclapAudioEncoder(model_path, providers=providers, intra_threads=1)


def _encode_one(job):
    index, path = job
    try:
        segments, track, duration = _WORKER['encoder'].encode_file(path)
    except Exception as exc:
        return {'index': index, 'path': path, 'error': f'{type(exc).__name__}: {exc}'}
    if segments.shape[0] == 0:
        return {'index': index, 'path': path, 'error': 'no segments decoded'}
    return {
        'index': index,
        'path': path,
        'segments': segments,
        'track': track,
        'duration': duration,
    }


def find_audio_files(root, limit=0):
    root = Path(root)
    files = [p for p in sorted(root.rglob('*')) if p.suffix.lower() in AUDIO_SUFFIXES]
    if limit:
        files = files[:limit]
    return [str(p) for p in files]


def load_shards(shard_dir):
    done = {}
    for shard in sorted(Path(shard_dir).glob('shard_*.npz')):
        try:
            payload = np.load(shard, allow_pickle=True)
        except Exception:
            continue
        paths = list(payload['paths'])
        counts = list(payload['counts'])
        durations = list(payload['durations'])
        segments = payload['segments']
        tracks = payload['tracks']
        offset = 0
        for path, count, duration, track in zip(paths, counts, durations, tracks):
            done[str(path)] = {
                'segments': segments[offset : offset + int(count)],
                'track': track,
                'duration': float(duration),
            }
            offset += int(count)
    return done


def write_shard(shard_dir, shard_index, results):
    if not results:
        return
    np.savez_compressed(
        Path(shard_dir) / f'shard_{shard_index:05d}.npz',
        paths=np.array([r['path'] for r in results], dtype=object),
        counts=np.array([r['segments'].shape[0] for r in results], dtype=np.int64),
        durations=np.array([r['duration'] for r in results], dtype=np.float32),
        segments=np.concatenate([r['segments'] for r in results], axis=0),
        tracks=np.stack([r['track'] for r in results], axis=0),
    )


def extract_from_audio(args):
    from multiprocessing import Pool

    from tqdm import tqdm

    out_dir = Path(args.out_dir)
    shard_dir = out_dir / 'shards'
    shard_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(args.audio_dir, args.limit)
    if not files:
        raise SystemExit(f'no audio files found under {args.audio_dir}')

    done = load_shards(shard_dir) if args.resume else {}
    pending = [(i, p) for i, p in enumerate(files) if p not in done]
    print(f'{len(files)} tracks found, {len(done)} already extracted, {len(pending)} to do')

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu else None
    failures = []
    shard_index = len(list(shard_dir.glob('shard_*.npz')))
    buffer = []
    started = time.time()

    if pending:
        with Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.model, providers),
        ) as pool:
            iterator = pool.imap_unordered(_encode_one, pending, chunksize=1)
            for result in tqdm(iterator, total=len(pending), unit='track'):
                if 'error' in result:
                    failures.append(result)
                    continue
                done[result['path']] = {
                    'segments': result['segments'],
                    'track': result['track'],
                    'duration': result['duration'],
                }
                buffer.append(result)
                if len(buffer) >= args.checkpoint_every:
                    write_shard(shard_dir, shard_index, buffer)
                    shard_index += 1
                    buffer = []
        write_shard(shard_dir, shard_index, buffer)

    kept = [p for p in files if p in done]
    if not kept:
        raise SystemExit('every track failed to decode, nothing to write')

    segments = np.concatenate([done[p]['segments'] for p in kept], axis=0).astype(np.float32)
    groups = np.concatenate(
        [np.full(done[p]['segments'].shape[0], i, dtype=np.int32) for i, p in enumerate(kept)]
    )
    tracks = np.stack([done[p]['track'] for p in kept], axis=0).astype(np.float32)

    meta = {
        'source': 'audio',
        'model': str(args.model),
        'audio_dir': str(args.audio_dir),
        'embedding_dim': int(segments.shape[1]),
        'n_tracks': len(kept),
        'n_segments': int(segments.shape[0]),
        'elapsed_seconds': round(time.time() - started, 1),
        'failures': [{'path': f['path'], 'error': f['error']} for f in failures],
        'tracks': [
            {
                'index': i,
                'path': p,
                'n_segments': int(done[p]['segments'].shape[0]),
                'duration': round(float(done[p]['duration']), 2),
            }
            for i, p in enumerate(kept)
        ],
    }
    write_outputs(
        out_dir, segments, groups, tracks, meta, names=[Path(p).name for p in kept]
    )
    if failures:
        print(f'{len(failures)} tracks failed, see meta.json')


def _dsn_label(dsn):
    parsed = urlsplit(dsn or '')
    if not parsed.hostname:
        return 'unknown'
    port = f':{parsed.port}' if parsed.port else ''
    return f'{parsed.hostname}{port}{parsed.path}'


def extract_from_postgres(args):
    import psycopg2
    from psycopg2 import sql
    from tqdm import tqdm

    out_dir = Path(args.out_dir)
    started = time.time()
    table = sql.Identifier(args.table)
    id_column = sql.Identifier(args.id_column)
    embedding_column = sql.Identifier(args.embedding_column)

    conn = psycopg2.connect(args.database_url, connect_timeout=args.connect_timeout)
    conn.set_session(readonly=True)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL('SELECT count(*) FROM {} WHERE {} IS NOT NULL').format(table, embedding_column)
        )
        available = int(cur.fetchone()[0])
    expected = min(available, args.limit) if args.limit else available
    if expected == 0:
        raise SystemExit(f'{args.table} holds no row with a non null {args.embedding_column}')
    print(
        f'{available} embeddings in {args.table} at {_dsn_label(args.database_url)}, '
        f'fetching {expected}'
    )

    query = sql.SQL('SELECT {}, {} FROM {} WHERE {} IS NOT NULL ORDER BY {}').format(
        id_column, embedding_column, table, embedding_column, id_column
    )
    if args.limit:
        query = sql.SQL('{} LIMIT {}').format(query, sql.Literal(int(args.limit)))

    segments = np.empty((expected, args.embedding_dim), dtype=np.float32)
    item_ids = []
    kept = 0
    rejected = 0
    cursor = conn.cursor(name='sae_embedding_export')
    cursor.itersize = args.fetch_size
    cursor.execute(query)
    for item_id, blob in tqdm(cursor, total=expected, unit='row'):
        vector = np.frombuffer(bytes(blob), dtype=np.float32)
        if vector.size != args.embedding_dim or not np.isfinite(vector).all():
            rejected += 1
            continue
        segments[kept] = vector
        item_ids.append(str(item_id))
        kept += 1
    cursor.close()
    conn.rollback()
    conn.close()

    if kept == 0:
        raise SystemExit(f'every row of {args.table} was unusable')
    segments = segments[:kept]
    norms = np.linalg.norm(segments, axis=1)
    groups = np.arange(kept, dtype=np.int32)
    meta = {
        'source': 'postgres',
        'database': _dsn_label(args.database_url),
        'table': args.table,
        'granularity': 'track',
        'embedding_dim': int(segments.shape[1]),
        'n_tracks': kept,
        'n_segments': kept,
        'rejected_rows': rejected,
        'norm_min': round(float(norms.min()), 6),
        'norm_mean': round(float(norms.mean()), 6),
        'norm_max': round(float(norms.max()), 6),
        'elapsed_seconds': round(time.time() - started, 1),
        'names_file': 'track_names.json',
    }
    write_outputs(out_dir, segments, groups, segments, meta, names=item_ids, item_ids=item_ids)
    if rejected:
        print(f'{rejected} rows rejected for a bad length or a non finite value')


def write_outputs(out_dir, segments, groups, tracks, meta, names=None, item_ids=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', segments)
    np.save(out_dir / 'groups.npy', groups)
    np.save(out_dir / 'track_embeddings.npy', tracks)
    if names is not None:
        payload = {'names': list(names)}
        if item_ids is not None:
            payload['item_ids'] = list(item_ids)
        (out_dir / 'track_names.json').write_text(json.dumps(payload) + '\n', newline='\n')
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n', newline='\n')
    print(
        f'wrote {segments.shape[0]} embeddings of dim {segments.shape[1]} '
        f"from {meta['n_tracks']} tracks to {out_dir}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Extract DCLAP embeddings for SAE training')
    parser.add_argument('--source', choices=['audio', 'postgres'], default='postgres')
    parser.add_argument('--audio-dir', default=None)
    parser.add_argument('--model', default=None, help='path to the DCLAP audio ONNX model')
    parser.add_argument('--database-url', default=os.environ.get('DATABASE_URL'))
    parser.add_argument('--out-dir', default='data')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--table', default='clap_embedding')
    parser.add_argument('--id-column', default='item_id')
    parser.add_argument('--embedding-column', default='embedding')
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--fetch-size', type=int, default=5000)
    parser.add_argument('--connect-timeout', type=int, default=15)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu', action='store_true', help='try CUDAExecutionProvider for ONNX')
    args = parser.parse_args(argv)
    if args.source == 'audio':
        if not args.audio_dir or not args.model:
            parser.error('--audio-dir and --model are required for --source audio')
    elif not args.database_url:
        parser.error('--database-url or DATABASE_URL is required for --source postgres')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.source == 'audio':
        extract_from_audio(args)
    else:
        extract_from_postgres(args)


if __name__ == '__main__':
    main()
