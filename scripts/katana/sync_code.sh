#!/bin/bash
set -euo pipefail

source_directory="${1:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
rsync_binary="${RSYNC_BIN:-rsync}"
ssh_key="${KATANA_SSH_KEY:?Set KATANA_SSH_KEY to the dedicated Katana key path}"

if [[ ! -d "$source_directory" ]]; then
  echo "Source directory does not exist: $source_directory" >&2
  exit 2
fi
if [[ ! -f "$ssh_key" ]]; then
  echo "SSH key does not exist: $ssh_key" >&2
  exit 2
fi

"$rsync_binary" -avhP \
  -e "ssh -i $ssh_key -o BatchMode=yes" \
  --exclude=.git \
  --exclude=.DS_Store \
  --exclude=.pytest_cache \
  --exclude=__pycache__ \
  --exclude='*.egg-info' \
  --exclude=artifacts \
  --exclude=runs \
  --exclude=reports/baseline_v1 \
  --exclude=reports/baseline_v2 \
  --exclude=lidc_data \
  "$source_directory/" \
  "z5448417@kdm.restech.unsw.edu.au:lidc_baseline/"
