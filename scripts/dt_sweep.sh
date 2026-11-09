#!/bin/bash
# Time-step convergence sweep on a FIXED mesh (adapted twice to the initial condition, then frozen).
#
#   scripts/dt_sweep.sh <solver_binary> <output_dir> <bdf2|euler> [extra solver options...]
#
# Runs dt = 4, 2, 1, 0.5, 0.25 ms to t = 6 s (the 0.25 ms run is the reference) in two parallel
# chains, then prints the self-convergence table:
#   python3 -I scripts/dt_analyze.py <output_dir> <scheme>
# Example (the study that confirmed second order, see HANDOVER.md section 5):
#   scripts/dt_sweep.sh ./cylinder_fem_amr runs/dt bdf2 --supg-tau-time 0.001
set -u
SOLVER=$(realpath "$1"); OUT=$(realpath -m "$2"); SCHEME=$3; shift 3
EXTRA=("$@")
run() {
    for dt in "$@"; do
        steps=$(python3 -c "print(round(6.0/$dt))")
        d=$OUT/${SCHEME}_$dt; mkdir -p "$d"
        (cd "$d" && "$SOLVER" --steps "$steps" --dt "$dt" --amr-interval 0 --init-amr 2 \
            --time-scheme "$SCHEME" --print-interval 100000 --stats-window 2 "${EXTRA[@]}" > run.log 2>&1)
        echo "$SCHEME dt=$dt exit=$?"
    done
}
run 0.00025 &
run 0.004 0.002 0.001 0.0005 &
wait
python3 -I "$(dirname "$0")/dt_analyze.py" "$OUT" "$SCHEME"
