#!/bin/bash
# Wall-time campaign on an otherwise idle machine: DFG 2D-2, t = 8 s, uniform and adaptive meshes,
# REPS repetitions each (error and time come from the same run). One job at a time; before each
# run it waits for the GPU to be idle and logs GPU utilisation and load average.
#   scripts/timing.sh <solver_binary> <output_dir> [reps]
# Env: FIRST=n starts at repetition n; ONLY="A3 U5" restricts the cases.
set -u
B=$(realpath "$1"); OUT=$2; REPS=${3:-2}; FIRST=${FIRST:-1}; ONLY=${ONLY:-U3 U4 U5 A1 A2 A3}
mkdir -p "$OUT" && cd "$OUT"
log() { echo "$(date +%F' '%T) $*"; }
idle() {   # GPU utilisation < 25 % (the desktop alone uses 4-14 %) and no compute process, 3 consecutive samples
    local ok=0 tries=0
    while [ $ok -lt 3 ]; do
        local u; u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        local c; c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
        if [ "$u" -lt 25 ] && [ "$c" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
        tries=$((tries+1)); [ $tries -gt 600 ] && return 1
        sleep 2
    done
}
run() { local n=$1; shift
    [ -f "$n/done" ] && { log "$n skipped (done)"; return; }
    idle || { log "$n GPU never idle, skipped"; return; }
    mkdir -p "$n" && cd "$n"
    log "$n start | load $(cut -d' ' -f1-3 /proc/loadavg) | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    "$B" "$@" > run.log 2> err.log; local rc=$?
    log "$n exit $rc $(grep -a -o 'Timing\] [0-9.]* s wall, [0-9.]* s in AMR' run.log | head -1)"
    [ $rc -eq 0 ] && touch done
    cd ..; }
F="--steps 8000 --stats-window 1.5"
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level 2"
A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level 2"
A3="--amr-tol-vort 0.025 --amr-tol-interp 0.0025 --max-level 6 --max-elems 1500000 --min-level 2"
want() { [[ " $ONLY " == *" $1 "* ]]; }
for r in $(seq $FIRST $REPS); do
    want U3 && run U3_r$r $F --uniform-level 3
    want U4 && run U4_r$r $F --uniform-level 4
    want U5 && run U5_r$r $F --uniform-level 5
    want A1 && run A1_r$r $F $A1
    want A2 && run A2_r$r $F $A2
    want A3 && run A3_r$r $F $A3
done
log "TIMING DONE"
