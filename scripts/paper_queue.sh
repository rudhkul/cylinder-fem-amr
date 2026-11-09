#!/bin/bash
# 2026-11-09: rerun with the current default binary (v17) every paper result that still came from
# an older binary: manufactured solution, 2D-2 time-step study on A2, A1 ablations, 2D-3 uniform
# runs and the 2D-3 time-step study; plus two A2 runs with VTK snapshots for the mesh figures.
# One job at a time, each waits for an idle GPU; a watcher logs foreign GPU processes to
# <out>/CONTAMINATED.log. Finished runs are marked <dir>/done and skipped on restart.
#   scripts/paper_queue.sh <solver_binary> [<output_dir> = runs/paper]
set -u
cd "$(dirname "$0")/.."
B=$(realpath "$1"); OUT=$(realpath -m "${2:-runs/paper}")
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

# 1. manufactured solution, dt ~ h, t = 0.1 s (uniform meshes: no AMR, no overlap)
for L in 1 2 3 4 5; do
    dt=$(python3 -c "print(1.0e-3 / 2**($L - 1))")
    steps=$(python3 -c "print(int(round(0.1 * 2**($L - 1) / 1.0e-3)))")
    run mms_L$L --mms --uniform-level $L --dt $dt --steps $steps --min-level 0   # the default floor 2 would turn L1 into L2
done
# 2. 2D-2 time step on A2: adaptation interval 10 ms of physical time, overlap K = interval / 2
run d2_A2_dt2  --steps 4000  --dt 0.002  --amr-interval 5  --amr-overlap 2  --stats-window 1.5 $A2
run d2_A2_dt05 --steps 16000 --dt 0.0005 --amr-interval 20 --amr-overlap 10 --stats-window 1.5 $A2
# 3. ablations on A1 (baseline = runs/v17/timing/A1_r1)
run abl_min_level_0        $F $A1 --min-level 0
run abl_keep_radius_0      $F $A1 --keep-radius 0
run abl_projection_current $F $A1 --remesh-projection current
run abl_oss_003            $F $A1 --pressure-stab 0.03
run abl_rotvel             $F $A1 --rotational-div velocity
run abl_supg_T05           $F $A1 --supg-tau-time 0.0005
run abl_supg_T2            $F $A1 --supg-tau-time 0.002
run abl_overlap_0          $F $A1 --amr-overlap 0
# 4. 2D-3 uniform meshes and time steps
run d3_U3       --case 2d3 --steps 8000 --uniform-level 3
run d3_U4       --case 2d3 --steps 8000 --uniform-level 4
run d3_U4_dt05  --case 2d3 --steps 16000 --dt 0.0005  --uniform-level 4
run d3_U4_dt025 --case 2d3 --steps 32000 --dt 0.00025 --uniform-level 4
# 5. snapshots for the mesh figures (every 1 s)
run snap_d2_A2 $F $A2 --vtk-interval 1000
run snap_d3_A2 --case 2d3 --steps 8000 $A2 --vtk-interval 1000
# 6. 2D-3 time step on A2 (longest, last)
run d3_A2_dt05  --case 2d3 --steps 16000 --dt 0.0005  --amr-interval 20 --amr-overlap 10 $A2
run d3_A2_dt025 --case 2d3 --steps 32000 --dt 0.00025 --amr-interval 40 --amr-overlap 20 $A2
log "PAPER QUEUE DONE"
