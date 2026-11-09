#!/bin/bash
# DFG 2D-3 time-step study (one job at a time): lift time convergence on A2 and U4.
#   scripts/d3_dt.sh <solver_binary> <output_dir>
set -u
B=$(realpath "$1"); OUT=$2
mkdir -p "$OUT" && cd "$OUT"
log() { echo "$(date +%F' '%T) $*"; }
run() { local n=$1; shift
    if [ -f "$n/done" ]; then log "$n skipped (done)"; return; fi
    mkdir -p "$n" && cd "$n"
    "$B" "$@" > run.log 2> err.log; local rc=$?
    log "$n exit $rc $(grep -o 'Timing.*' run.log | head -1 | cut -c1-90)"
    [ $rc -eq 0 ] && touch done
    cd ..; }
A2="--amr-tol-vort 0.05 --amr-tol-interp 0.005 --max-level 5 --max-elems 400000 --min-level 2"
run A2_dt025 --case 2d3 --steps 32000 --dt 0.00025 --amr-interval 40 $A2
run U4_dt05  --case 2d3 --steps 16000 --dt 0.0005 --uniform-level 4
run U4_dt025 --case 2d3 --steps 32000 --dt 0.00025 --uniform-level 4
log "D3 DT STUDY DONE"
