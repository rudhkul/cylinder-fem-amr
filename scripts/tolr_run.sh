#!/bin/bash
# One DFG 2D-2 run (t = 8 s) with extra AMR options, idle-GPU gate and CONTAMINATED marker as in
# timing.sh. Used for the distance-graded tolerance study (HANDOVER 11.15).
#   scripts/tolr_run.sh <binary> <run_dir> <A1|A2|A3> [extra solver args...]
set -u
B=$(realpath "$1"); D=$2; LV=$3; shift 3
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level 2"
A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level 2"
A3="--amr-tol-vort 0.025 --amr-tol-interp 0.0025 --max-level 6 --max-elems 1500000 --min-level 2"
eval "BASE=\$$LV"
ok=0; tries=0
while [ $ok -lt 3 ]; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$u" -lt 25 ] && [ "$c" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
    tries=$((tries+1)); [ $tries -gt 600 ] && { echo "GPU never idle, not started"; exit 2; }
    sleep 2
done
mkdir -p "$D" && cd "$D"
"$B" --steps 8000 --stats-window 1.5 $BASE "$@" > run.log 2> err.log &
pid=$!
while kill -0 $pid 2>/dev/null; do
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$pid\$" | grep -c .)
    [ "$others" -gt 0 ] && echo "other GPU process seen at $(date +%T)" >> CONTAMINATED
    sleep 5
done
wait $pid; rc=$?
[ $rc -eq 0 ] && touch done
echo "exit $rc $(grep -a -o 'Timing\] [0-9.]* s wall, [0-9.]* s in AMR' run.log | head -1)"
[ -f CONTAMINATED ] && echo CONTAMINATED
