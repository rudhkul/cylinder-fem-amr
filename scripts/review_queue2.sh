#!/bin/bash
# 2026-11-09 (review, second round): re-projection on an unchanged mesh (--reproject-test) and the
# analogue of the next-step term of Braack, Lang & Taschenberger (2013), Eq. (21) (--reproject-shift),
# with build/cfa_v18c_review (= v18b plus these two options; v17 at default options).
# One job at a time, each waits for an idle GPU; a watcher logs foreign GPU processes to
# <out>/CONTAMINATED.log. Finished runs are marked <dir>/done and skipped on restart.
set -u
cd "$(dirname "$0")/.."
B=$(realpath "$1"); OUT=$(realpath -m "${2:-runs/review2}")
mkdir -p "$OUT"
log() { echo "$(date +%F' '%T) $*"; }
watch() {
    while true; do
        mine=$(pgrep -f "$B" | tr '\n' '|'); mine=${mine%|}
        others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -vE "^(${mine:-none})," )
        [ -n "$others" ] && echo "$(date +%F' '%T) $others" >> "$OUT/CONTAMINATED.log"
        sleep 5
    done; }
idle() { local ok=0 tries=0
    while [ $ok -lt 3 ]; do
        local u c
        u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
        if [ "$u" -lt 25 ] && [ "$c" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
        tries=$((tries+1)); [ $tries -gt 1800 ] && return 1
        sleep 2
    done; }
run() { local n=$1; shift
    [ -f "$OUT/$n/done" ] && { log "$n skipped (done)"; return; }
    idle || { log "$n GPU never idle, skipped"; return; }
    mkdir -p "$OUT/$n" && pushd "$OUT/$n" > /dev/null
    echo "$B $*" > cmd.txt
    log "$n start | load $(cut -d' ' -f1-3 /proc/loadavg) | gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    "$B" "$@" > run.log 2> err.log; local rc=$?
    log "$n exit $rc $(grep -a -o 'Timing\] [0-9.]* s wall, [0-9.]* s in AMR' run.log | head -1)"
    [ $rc -eq 0 ] && touch done
    popd > /dev/null; }
watch & wpid=$!
trap 'kill $wpid 2>/dev/null' EXIT

F="--steps 8000 --stats-window 1.5"
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level 2"

# 1. 2D-2 on U3: re-projection every 10 steps on the unchanged mesh
run U3_base            $F --uniform-level 3
run U3_rt_scheme       $F --uniform-level 3 --reproject-test 10
run U3_rt_scheme_shift $F --uniform-level 3 --reproject-test 10 --reproject-shift on
run U3_rt_scheme_cur   $F --uniform-level 3 --reproject-test 10 --remesh-projection scheme-current
run U3_rt_zero_cur     $F --uniform-level 3 --reproject-test 10 --remesh-projection current
run U3_rt_zero_both    $F --uniform-level 3 --reproject-test 10 --remesh-projection both
# 2. A1 with the next-step term
run abl_reproject_shift $F $A1 --reproject-shift on
# 3. MMS, dt ~ h, with the re-projection every 10 steps on the unchanged mesh
for L in 1 2 3 4 5; do
    dt=$(python3 -c "print(1.0e-3 / 2**($L - 1))")
    steps=$(python3 -c "print(int(round(0.1 * 2**($L - 1) / 1.0e-3)))")
    run mms_L${L}_rt_scheme --mms --uniform-level $L --dt $dt --steps $steps --min-level 0 --reproject-test 10
    run mms_L${L}_rt_zero   --mms --uniform-level $L --dt $dt --steps $steps --min-level 0 --reproject-test 10 --remesh-projection current
done
log "REVIEW2 QUEUE DONE"
