#!/bin/bash
# 2026-11-09 (review): adaptive outlet-condition runs again with build/cfa_v18d_review, which also
# balances the flux of the transferred outlet velocities after a remesh (cfa_v18b_review left a single-
# step drag spike when the outlet was coarsened; those runs are in runs/review/superseded_v18b_outlet).
# Uses the run()/idle()/watch() of review_queue.sh.
set -u
cd "$(dirname "$0")/.."
eval "$(sed -n '/^B=/,/^trap/p' scripts/review_queue.sh)"
F="--steps 8000 --stats-window 1.5"
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level 2"
A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level 2"
run d2_A1_outdir $F $A1 --outlet dirichlet
run d3_A2_outdir --case 2d3 --steps 8000 $A2 --outlet dirichlet
log "REVIEW3 QUEUE DONE"
