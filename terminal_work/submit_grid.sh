#!/bin/bash
set -euo pipefail
mkdir -p logs

EPOCHS_LIST=( 100000  )
BATCH_LIST=( 1048 2096 3192 )
LR_LIST=( 0.001 )
NOISE_LIST=( 0.00 0.05 0.1 0.3 0.5 )

for ep in "${EPOCHS_LIST[@]}"; do
  for bs in "${BATCH_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for nz in "${NOISE_LIST[@]}"; do
        tag="E${ep}_B${bs}_LR${lr/_/}_N${nz/_/}"
          
            # Submit: pass vars with -v; values contain NO spaces/commas
        msub \
          -N "pdenet_${tag}" \
          -o "logs/${tag}.out" \
          -e "logs/${tag}.err" \
          -v EPOCHS="$ep",BATCH="$bs",LR="$lr",NOISE="$nz" \
           run_pdenet.msub
           
        echo "Submitted pdenet_${tag}"      
      done
    done
  done
done
