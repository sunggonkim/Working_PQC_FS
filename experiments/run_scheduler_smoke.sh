#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-${root_dir}/artifacts}"

mkdir -p "${out_dir}"

if [[ ! -x "${root_dir}/build/pqc_fuse" ]]; then
  echo "missing build/pqc_fuse; run cmake --build build first" >&2
  exit 1
fi

"${root_dir}/build/pqc_fuse" --scheduler-smoke 2>&1 | tee "${out_dir}/scheduler_smoke.jsonl"
