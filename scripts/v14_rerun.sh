#!/bin/bash
# 2026-10-22 rerun with v14 (HANDOVER 11.12): timing campaign (1 repetition) and 2D-3 A2/A3 at 1 ms.
# A watcher logs any foreign GPU compute process to runs/v14/CONTAMINATED.log (match by time with
# the start/exit lines in runs/v14/*.out).
#   scripts/v14_rerun.sh <solver_binary>
set -u
cd "$(dirname "$0")/.."
B=$(realpath "$1")
OUT=runs/v14
mkdir -p $OUT
watch() {
    while true; do
        mine=$(pgrep -f "$B" | tr '\n' '|'); mine=${mine%|}
        others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vE "^(${mine:-none})," )
        [ -n "$others" ] && echo "$(date +%F' '%T) $others" >> $OUT/CONTAMINATED.log
        sleep 5
    done; }
watch & wpid=$!
REPS=1 scripts/timing.sh "$B" $OUT/timing 1 > $OUT/timing.out 2>&1
D3_A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level 2"
D3_A3="--amr-tol-vort 0.025 --amr-tol-interp 0.0025 --max-level 6 --max-elems 1500000 --min-level 2"
log() { echo "$(date +%F' '%T) $*"; }
idle() { local ok=0 tries=0
    while [ $ok -lt 3 ]; do
        local u c
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
        if [ "$u" -lt 25 ] && [ "$c" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
        tries=$((tries+1)); [ $tries -gt 1800 ] && return 1
        sleep 2
    done; }
d3() { local n=$1; shift
    idle || { log "$n GPU never idle, skipped"; return; }
    mkdir -p $OUT/d3/$n && pushd $OUT/d3/$n > /dev/null
    log "$n start"
    "$B" --case 2d3 --steps 8000 "$@" > run.log 2> err.log
    log "$n exit $? $(grep -a -o 'Timing\] [0-9.]* s wall, [0-9.]* s in AMR' run.log | head -1)"
    popd > /dev/null; }
d3 A2 $D3_A2 >> $OUT/d3.out
d3 A3 $D3_A3 >> $OUT/d3.out
echo "$(date +%F' '%T) RERUN DONE" >> $OUT/d3.out
kill $wpid
