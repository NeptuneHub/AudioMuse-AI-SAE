#!/usr/bin/env bash
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
#
# End to end pipeline: extract DCLAP embeddings, train the BatchTopK SAE,
# evaluate it and export the ONNX graphs. Override any of the variables below
# from the environment. Run from the SAE directory under WSL.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

SOURCE="${SOURCE:-postgres}"
DATABASE_URL="${DATABASE_URL:-}"
DCLAP_MODEL="${DCLAP_MODEL:-/path/to/AudioMuse-AI/model/model_epoch_36.onnx}"
AUDIO_DIR="${AUDIO_DIR:-/path/to/music}"
DATA_DIR="${DATA_DIR:-$HERE/data}"
RUN_DIR="${RUN_DIR:-$HERE/runs/dclap_sae}"
MODEL_DIR="${MODEL_DIR:-$HERE/model}"
K="${K:-20}"
STEPS="${STEPS:-50000}"
BATCH="${BATCH:-4096}"
WORKERS="${WORKERS:-10}"

if [ ! -x "$PY" ]; then
    echo "venv missing, run: bash setup_venv.sh" >&2
    exit 1
fi

if [ ! -f "$DATA_DIR/embeddings.npy" ]; then
    echo "== extracting DCLAP embeddings from $SOURCE =="
    if [ "$SOURCE" = "postgres" ]; then
        if [ -z "$DATABASE_URL" ]; then
            echo "set DATABASE_URL to the AudioMuse-AI Postgres instance" >&2
            exit 1
        fi
        "$PY" "$HERE/extract_embeddings.py" \
            --source postgres \
            --database-url "$DATABASE_URL" \
            --out-dir "$DATA_DIR"
    else
        "$PY" "$HERE/extract_embeddings.py" \
            --source audio \
            --audio-dir "$AUDIO_DIR" \
            --model "$DCLAP_MODEL" \
            --out-dir "$DATA_DIR" \
            --workers "$WORKERS" \
            --resume
    fi
else
    echo "== reusing $DATA_DIR/embeddings.npy =="
fi

echo "== training SAE (k=$K) =="
"$PY" "$HERE/train_sae.py" \
    --data "$DATA_DIR" \
    --k "$K" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --out "$RUN_DIR"

echo "== evaluating =="
"$PY" "$HERE/evaluate_sae.py" \
    --checkpoint "$RUN_DIR/sae.pt" \
    --data "$DATA_DIR" \
    --split all

echo "== exporting ONNX =="
"$PY" "$HERE/export_onnx.py" \
    --checkpoint "$RUN_DIR/sae.pt" \
    --out-dir "$MODEL_DIR" \
    --name "dclap_sae_k${K}" \
    --embeddings "$DATA_DIR/embeddings.npy"

echo "== done =="
ls -la "$MODEL_DIR"
