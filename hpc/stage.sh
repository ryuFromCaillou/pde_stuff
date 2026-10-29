#!/bin/bash
# Stage the pinned research source on HPC2. Run from Ark2.
# Usage: hpc/stage.sh [remote-base]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_BASE="${1:-/home/032480329/hpc_pde_smooth_noise/20261028}"
KEY="/home/ghost/.ssh/id_ed25519_hpc"
SSH=(ssh -o BatchMode=yes -o PasswordAuthentication=no -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -i "$KEY")
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
BUNDLE="/tmp/opencode/burgers_smooth_noise_${COMMIT:0:12}.tar.gz"
mkdir -p /tmp/opencode
git -C "$REPO" archive --format=tar.gz -o "$BUNDLE" HEAD -- \
  Datasets/data/processed/burg_gen Datasets/__init__.py \
  Datasets/data/processed/__init__.py prog utils runs tests hpc setup \
  BURGERS_NOISE_SAMPLING_SPEC.md
SHA="$(sha256sum "$BUNDLE" | awk '{print $1}')"
"${SSH[@]}" 032480329@hpc2.its.csulb.edu "mkdir -p '$REMOTE_BASE' && rm -rf '$REMOTE_BASE/src' && mkdir -p '$REMOTE_BASE/src'"
"${SSH[@]}" 032480329@hpc2.its.csulb.edu "tar xzf - -C '$REMOTE_BASE/src'" < "$BUNDLE"
"${SSH[@]}" 032480329@hpc2.its.csulb.edu "printf '%s  %s\n' '$SHA' 'burgers_smooth_noise_${COMMIT:0:12}.tar.gz' > '$REMOTE_BASE/BUNDLE.sha256'; printf 'commit=%s\n' '$COMMIT' > '$REMOTE_BASE/SOURCE_COMMIT.txt'"
# Verify the staged tree imports with the read-only pdenet environment.
"${SSH[@]}" 032480329@hpc2.its.csulb.edu "cd '$REMOTE_BASE/src' && test -f runs/run_burgers_smooth_noise_sampling.py && /home/032480329/mambaforge/envs/pdenet/bin/python -c 'import utils.burgers_smooth_noise_sampling as m; print(\"staged_import_ok\", m.COUNTS, len(m.run_manifest()))'"
echo "staged commit=$COMMIT sha256=$SHA remote=$REMOTE_BASE"
