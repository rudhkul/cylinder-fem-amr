#!/bin/bash
# Verification and convergence campaign for the paper (runs one job at a time).
#
#   scripts/campaign.sh <solver_binary> <output_dir> [--amr-floor L]
#
# Each run gets its own directory; a run that finished is marked <dir>/done and skipped when the
# script is restarted. Afterwards:
#   scripts/mms_table.py <out>/mms_L*/cylinder_fem_amr_mms.csv
#   scripts/convergence_table.py <out>/U3 <out>/U4 <out>/U5        (and A1 A2 A3)
#   scripts/plot_convergence.py fig.png --uniform <out>/U3 ... --amr <out>/A1 ...
set -u
B=$(realpath "$1"); OUT=$2; shift 2
FLOOR=2
[ "${1:-}" = "--amr-floor" ] && FLOOR=$2
mkdir -p "$OUT" && cd "$OUT"
log() { echo "$(date +%F' '%T) $*"; }
run() {  # name args...
    local n=$1; shift
    if [ -f "$n/done" ]; then log "$n skipped (done)"; return; fi
    mkdir -p "$n" && cd "$n"
    "$B" "$@" > run.log 2>&1
    local rc=$?
    log "$n exit $rc $(grep -o 'Timing.*' run.log | head -1 | cut -c1-90)"
    [ $rc -eq 0 ] && touch done
    cd ..
}
# 1. manufactured solution, space-time refinement dt ~ h, t = 0.1 s
for L in 1 2 3 4 5; do
    dt=$(python3 -c "print(1.0e-3 / 2**($L - 1))")
    steps=$(python3 -c "print(int(round(0.1 * 2**($L - 1) / 1.0e-3)))")
    run mms_L$L --mms --uniform-level $L --dt $dt --steps $steps --min-level 0   # the default floor 2 would turn L1 into L2
done
# 2. DFG 2D-2 to t = 8 s (statistics t >= 6.5 s): uniform and adaptive meshes
F="--steps 8000 --stats-window 1.5"
A1="--amr-tol-vort 0.1   --amr-tol-interp 0.01   --max-level 4 --max-elems 200000  --min-level $FLOOR"
A2="--amr-tol-vort 0.05  --amr-tol-interp 0.005  --max-level 5 --max-elems 400000  --min-level $FLOOR"
A3="--amr-tol-vort 0.025 --amr-tol-interp 0.0025 --max-level 6 --max-elems 1500000 --min-level $FLOOR"
run U3 $F --uniform-level 3
run U4 $F --uniform-level 4
run A1 $F $A1
run A2 $F $A2
run U5 $F --uniform-level 5
run A3 $F $A3
# 3. time step on A2 (adaptation interval kept at 10 ms of physical time)
run A2_dt2  --steps 4000  --dt 0.002  --amr-interval 5  --stats-window 1.5 $A2
run A2_dt05 --steps 16000 --dt 0.0005 --amr-interval 20 --stats-window 1.5 $A2
# 4. DFG 2D-3 (inflow 1.5 sin(pi t / 8), 0 <= t <= 8 s)
run d3_U3 --case 2d3 --steps 8000 --uniform-level 3
run d3_U4 --case 2d3 --steps 8000 --uniform-level 4
run d3_A2 --case 2d3 --steps 8000 $A2
run d3_A3 --case 2d3 --steps 8000 $A3
log "CAMPAIGN DONE"
