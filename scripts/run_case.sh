#!/bin/bash
# Run one solver case in its own directory and render every VTK snapshot to PNG as it appears.
#
#   scripts/run_case.sh <solver_binary> <output_dir> [solver options...]
#
# Example (6 s DFG 2D-2 benchmark, snapshot every 0.5 s):
#   scripts/run_case.sh ./cylinder_fem_amr runs/bench --steps 6000 --vtk-interval 500 --stats-window 2
#
# PNGs go to <output_dir>/png/. Force statistics are printed at the end.
set -u
SOLVER=$(realpath "$1"); OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/png" && cd "$OUT"

"$SOLVER" "$@" > run.log 2>&1 &
PID=$!

render() {
    f=$1
    step=$(basename "$f" .vtk | sed 's/.*_0*//')
    [ "$step" = "final" ] && title="final" || title="step $step"
    python3 -I "$HERE/render_vtk.py" "$f" "png/$(basename "$f" .vtk).png" "$title" && echo "rendered $f"
}

done_list=""
while kill -0 $PID 2>/dev/null; do
    for f in cylinder_fem_amr_0*.vtk; do
        [ -f "$f" ] || continue
        case " $done_list " in *" $f "*) continue ;; esac
        sleep 3   # let the writer finish the file
        render "$f"; done_list="$done_list $f"
    done
    sleep 5
done
wait $PID; echo "solver exit code $?"
for f in cylinder_fem_amr_0*.vtk; do
    [ -f "$f" ] || continue
    case " $done_list " in *" $f "*) ;; *) render "$f" ;; esac
done
[ -f cylinder_fem_amr_final.vtk ] && render cylinder_fem_amr_final.vtk
grep -A7 "Force statistics" run.log; grep -E "abort|failed" run.log
sed -n '/^\[Timing\]/,/^\[Output\]/p' run.log | grep -v "^\[Output\]"
