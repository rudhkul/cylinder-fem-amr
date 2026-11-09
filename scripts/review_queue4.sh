#!/bin/bash
# 2026-11-09 (third review): periodic-state check (U4 and A1 to t = 30 s) and the sensitivity to the
# adaptation interval (A1 at 5 and 20 ms, overlap K = interval / 2), with build/cfa_v18d_review.
# Uses the run()/idle()/watch() of review_queue.sh.
set -u
cd "$(dirname "$0")/.."
eval "$(sed -n '/^B=/,/^trap/p' scripts/review_queue.sh)"
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level 2"
run U4_t30 --steps 30000 --stats-window 1.5 --uniform-level 4
run A1_t30 --steps 30000 --stats-window 1.5 $A1
run A1_int5  --steps 8000 --stats-window 1.5 $A1 --amr-interval 5  --amr-overlap 2
run A1_int20 --steps 8000 --stats-window 1.5 $A1 --amr-interval 20 --amr-overlap 10
log "REVIEW4 QUEUE DONE"
