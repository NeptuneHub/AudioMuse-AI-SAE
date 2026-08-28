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
# Creates the SAE/.venv virtual environment under WSL/Linux and installs the
# CUDA build of PyTorch plus the extraction dependencies. Run it from the SAE
# directory: bash setup_venv.sh [--with-optional]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
WITH_OPTIONAL=0
for arg in "$@"; do
    if [ "$arg" = "--with-optional" ]; then WITH_OPTIONAL=1; fi
done

if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv not found, installing it to ~/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

echo "Creating venv at $VENV"
uv venv "$VENV" --python 3.12

echo "Installing requirements"
VIRTUAL_ENV="$VENV" uv pip install -r "$HERE/requirements.txt"

if [ "$WITH_OPTIONAL" = "1" ]; then
    echo "Installing optional requirements"
    VIRTUAL_ENV="$VENV" uv pip install -r "$HERE/requirements-optional.txt"
fi

echo "Checking CUDA visibility"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "Done. Activate with: source $VENV/bin/activate"
