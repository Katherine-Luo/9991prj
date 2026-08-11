#!/bin/bash
set -euo pipefail

source_directory="${1:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
destination="/srv/scratch/z5448417/lidc-baseline-v2"
rsync_binary="${RSYNC_BIN:-rsync}"
ssh_binary="${SSH_BIN:-ssh}"
python_binary="${PYTHON_BIN:-python3}"
ssh_key="${KATANA_SSH_KEY:?Set KATANA_SSH_KEY to the dedicated Katana key path}"
remote="z5448417@kdm.restech.unsw.edu.au"
manifest="artifacts/baseline_v2/manifests/p8_stage_a_transfer_manifest.json"

if [[ ! -d "$source_directory" ]]; then
  echo "Source directory does not exist: $source_directory" >&2
  exit 2
fi
if [[ ! -f "$ssh_key" ]]; then
  echo "SSH key does not exist: $ssh_key" >&2
  exit 2
fi
if [[ ! -f "$source_directory/$manifest" ]]; then
  echo "Build the P8 Stage A transfer manifest before synchronization." >&2
  exit 2
fi

"$ssh_binary" -i "$ssh_key" -o BatchMode=yes "$remote" "mkdir -p '$destination'"

cd "$source_directory"
transfer_list="$(mktemp)"
trap 'rm -f "$transfer_list"' EXIT
PYTHONPATH="$source_directory/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_binary" -m lidc_baseline.p8_katana transfer-list \
  --repository-root "$source_directory" \
  --transfer-manifest "$manifest" \
  > "$transfer_list"
"$rsync_binary" -avhP \
  -e "ssh -i $ssh_key -o BatchMode=yes" \
  --relative \
  --files-from="$transfer_list" \
  --exclude=.DS_Store \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  "$source_directory/" \
  "$remote:$destination/"
