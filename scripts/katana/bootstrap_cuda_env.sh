#!/bin/bash
set -euo pipefail

code_directory="${KATANA_CODE_DIR:-$HOME/lidc_baseline}"
conda_root="${KATANA_CONDA_ROOT:-/srv/scratch/z5448417/conda}"
bootstrap_directory="${KATANA_P0_DIR:-/srv/scratch/z5448417/lidc-baseline-p0}/bootstrap"
environment_prefix="$conda_root/envs/lidc-baseline-v1"
installer="$bootstrap_directory/Miniconda3-py311_25.7.0-2-Linux-x86_64.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "This script must run inside an allocated Katana GPU node." >&2
  exit 2
fi
nvidia-smi >/dev/null

mkdir -p "$bootstrap_directory"
if [[ ! -x "$conda_root/bin/conda" ]]; then
  if [[ -e "$conda_root" ]]; then
    echo "Incomplete Conda prefix already exists: $conda_root" >&2
    exit 2
  fi
  curl --fail --location --output "$installer" \
    "https://repo.anaconda.com/miniconda/Miniconda3-py311_25.7.0-2-Linux-x86_64.sh"
  bash "$installer" -b -p "$conda_root"
fi

if [[ ! -x "$environment_prefix/bin/python" ]]; then
  CONDA_SOLVER=classic "$conda_root/bin/conda" create \
    --prefix "$environment_prefix" \
    --override-channels \
    --channel conda-forge \
    python=3.11 pip setuptools=80.10.2 --yes
fi

"$environment_prefix/bin/python" -m pip install \
  "torch==2.5.1" \
  --index-url "https://download.pytorch.org/whl/cu121"
"$environment_prefix/bin/python" -m pip install "$code_directory[dev]"
"$environment_prefix/bin/python" -m pip check
"$environment_prefix/bin/python" -c \
  "import pylidc; import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, pylidc.__version__)"
