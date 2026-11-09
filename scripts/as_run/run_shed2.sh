#!/bin/bash
# Runs the 6 s shedding case and renders every VTK snapshot to a PNG in ~/Downloads as it appears.
S=${SCRATCH:-/tmp/cfa_scratch}
OUT=/home/ani/Downloads
mkdir -p $S/shed6b && cd $S/shed6b
$S/shed_solver2 --steps 6000 --vtk-interval 500 --print-interval 500 --stats-window 2 > run.log 2>&1 &
PID=$!
render() {  # $1 = vtk file
  f=$1; step=$(basename $f .vtk | sed 's/.*_0*//'); [ "$step" = "final" ] && step=6000
  t=$(python3 -c "print(f'{int($step)/1000:g}')")
  python3 -I $S/tools/viz2.py $f $OUT/amr_combined_t${t}s.png $S/tools "t = ${t} s" && echo "rendered t=${t}s"
}
done_list=""
while kill -0 $PID 2>/dev/null; do
  for f in cylinder_fem_amr_0*.vtk; do
    [ -f "$f" ] || continue
    case " $done_list " in *" $f "*) continue;; esac
    sleep 3   # let the writer finish the file
    render $f; done_list="$done_list $f"
  done
  sleep 5
done
wait $PID; echo "solver exit code $?"
for f in cylinder_fem_amr_0*.vtk; do case " $done_list " in *" $f "*) ;; *) render $f;; esac; done
render cylinder_fem_amr_final.vtk
grep -A7 "Force statistics" run.log; grep -E "Timing|abort|failed" run.log
