#!/bin/bash
# Short remesh-cost benchmark on the A3 settings (about 0.85M-1.2M elements after step 500).
# Waits for an idle GPU (same gate as timing.sh) and marks the run CONTAMINATED if a compute
# process appears while it runs.
#   scripts/amrcost_bench.sh <binary> <run_dir> [steps] [extra solver args...]
set -u
B=$(realpath "$1"); D=$2; STEPS=${3:-600}; shift 3 2>/dev/null || shift $#
idle() { local ok=0 tries=0
    while [ $ok -lt 3 ]; do
        local u c
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
        if [ "$u" -lt 25 ] && [ "$c" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
        tries=$((tries+1)); [ $tries -gt 300 ] && return 1
        sleep 2
    done; }
idle || { echo "GPU never idle, not started"; exit 2; }
mkdir -p "$D" && cd "$D"
A3="--amr-tol-vort 0.025 --amr-tol-interp 0.0025 --max-level 6 --max-elems 1500000 --min-level 2"
${WRAP:-} "$B" --steps "$STEPS" $A3 "$@" > run.log 2> err.log &
pid=$!
while kill -0 $pid 2>/dev/null; do
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$pid\$" | grep -c .)
    [ "$others" -gt 0 ] && echo "other GPU process seen at $(date +%T)" >> CONTAMINATED
    sleep 5
done
wait $pid; echo "exit $?"
grep -a -A16 'Timing\] [0-9.]* s wall' run.log
[ -f CONTAMINATED ] && echo "CONTAMINATED"
