#!/bin/bash
# Ablations of the design choices on the A1 configuration (DFG 2D-2, t = 0..8 s), one at a time.
#   scripts/ablations.sh <solver_binary> <output_dir>
set -u
B=$(realpath "$1"); OUT=$2
mkdir -p "$OUT" && cd "$OUT"
log() { echo "$(date +%F' '%T) $*"; }
run() {
    local n=$1; shift
    if [ -f "$n/done" ]; then log "$n skipped (done)"; return; fi
    mkdir -p "$n" && cd "$n"
    "$B" --steps 8000 --stats-window 1.5 "$@" > run.log 2>&1
    local rc=$?
    log "$n exit $rc"
    [ $rc -eq 0 ] && touch done
    cd ..
}
run abl_projection_current --remesh-projection current     # pre-2026-10 remesh projection
run abl_keep_radius_0      --keep-radius 0                 # coarsening allowed near the cylinder
run abl_oss_003            --pressure-stab 0.03            # old OSS factor
run abl_rotvel             --rotational-div velocity       # old rotational term
run abl_supg_T05           --supg-tau-time 0.0005
run abl_supg_T2            --supg-tau-time 0.002
log "ABLATIONS DONE"
