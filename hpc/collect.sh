#!/bin/bash
# Retrieve HPC results to Ark2 without deleting local files.
# Usage: hpc/collect.sh [remote-base] [local-destination]
set -euo pipefail
REMOTE_BASE="${1:-/home/032480329/hpc_pde_smooth_noise/20261028}"
LOCAL="${2:-/home/ghost/reports/pde/burgers_smooth_noise_20261028}"
KEY="/home/ghost/.ssh/id_ed25519_hpc"
mkdir -p "$LOCAL"
rsync -av --partial --human-readable \
  -e "ssh -o BatchMode=yes -o PasswordAuthentication=no -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -i $KEY" \
  032480329@hpc2.its.csulb.edu:"$REMOTE_BASE/" "$LOCAL/"
echo "retrieved $REMOTE_BASE -> $LOCAL"
