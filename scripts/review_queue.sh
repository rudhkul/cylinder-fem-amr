#!/bin/bash
# 2026-11-09 (review): runs that answer the external review of the CMAF draft, with build/cfa_v18b_review
# (= v17 at default options; adds --outlet dirichlet and --remesh-projection scheme-current):
# (MMS, re-projection and U4 dt runs were made with the first build, cfa_v18a_review, whose default and
# --remesh-projection paths are identical; its outlet runs, with an unscaled outlet parabola, are in
# runs/review/superseded_v18a_outlet.)
# re-projection factorial on A1, 2D-2 time step on U4, outlet-condition sensitivity (John 2004
# prescribes the parabola at the outlet), and the rotational-term variants on the MMS at fixed h.
# One job at a time, each waits for an idle GPU; a watcher logs foreign GPU processes to
# <out>/CONTAMINATED.log. Finished runs are marked <dir>/done and skipped on restart.
#   scripts/review_queue.sh build/cfa_v18b_review [<output_dir> = runs/review]
set -u
cd "$(dirname "$0")/.."
B=$(realpath "$1"); OUT=$(realpath -m "${2:-runs/review}")
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
A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level 2"

# 1. MMS on U3 at three dt: rotational term from the increment (default) and from u* (velocity)
for k in 0 1 2; do
    dt=$(python3 -c "print(1.0e-3 / 2**$k)"); steps=$((100 * 2**k))
    run mms_L3_dt$k     --mms --uniform-level 3 --min-level 0 --dt $dt --steps $steps
    run mms_L3_dt${k}_rotvel --mms --uniform-level 3 --min-level 0 --dt $dt --steps $steps --rotational-div velocity
done
# 2. re-projection factorial on A1 (baseline: both levels, scheme step = runs/v17/timing/A1_r1;
#    u^n only to zero divergence = runs/paper/abl_projection_current)
run abl_projection_scheme_current $F $A1 --remesh-projection scheme-current
run abl_projection_both           $F $A1 --remesh-projection both
run abl_projection_none           $F $A1 --remesh-projection none
# 3. 2D-2 time step on U4 (dt = 1 ms: runs/v14/timing/U4_r1)
run d2_U4_dt2  --steps 4000  --dt 0.002  --stats-window 1.5 --uniform-level 4
run d2_U4_dt05 --steps 16000 --dt 0.0005 --stats-window 1.5 --uniform-level 4
# 4. outlet condition: inflow parabola at the outlet (John 2004)
run d2_U4_outdir $F --uniform-level 4 --outlet dirichlet
run d2_A1_outdir $F $A1 --outlet dirichlet
run d3_U4_outdir       --case 2d3 --steps 8000  --uniform-level 4 --outlet dirichlet
run d3_U4_outdir_dt05  --case 2d3 --steps 16000 --dt 0.0005  --uniform-level 4 --outlet dirichlet
run d3_U4_outdir_dt025 --case 2d3 --steps 32000 --dt 0.00025 --uniform-level 4 --outlet dirichlet
run d3_A2_outdir       --case 2d3 --steps 8000 $A2 --outlet dirichlet
log "REVIEW QUEUE DONE"
