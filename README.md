# cylinder_fem_amr

GPU (CUDA) finite-element solver for incompressible flow past a cylinder (DFG benchmarks 2D-2
and 2D-3, Re = 100) with dynamic adaptive mesh refinement.

* **Elements and stabilisation:** P1-P1 elements with SUPG and skew-symmetric convection,
  and orthogonal-subscale (OSS) pressure stabilisation.
* **Time stepping:** BDF2 incremental pressure correction in rotational form.
* **Solvers:** smoothed-aggregation AMG for the pressure; BiCGSTAB with multicolour ILU(0) for
  momentum (both velocity components in one solve).
* **AMR:** newest-vertex bisection with coarsening, an advective look-ahead, and a
  scheme-consistent re-projection after each remesh.
* **Forces:** volume-integral drag and lift.

This repository accompanies the paper *Remeshing-induced force oscillations and computational cost
in a GPU finite-element projection solver* (A. Kulkarni). It holds the solver, the run scripts
with every command line, the force histories and statistics of every run in the paper, and the
scripts that regenerate its tables and figures, and the manuscript itself (`paper/`). Comments of the form `HANDOVER 11.x` refer to the
author's internal development log, which is not part of this repository.

## Build

The code needs CUDA ≥ 12 (tested with 13.3), a C++17 host compiler and OpenMP.

```bash
nvcc -O3 -arch=native -std=c++17 -Xcompiler -Wall,-Wextra,-fopenmp src/cylinder_fem_amr.cu -o cylinder_fem_amr
# or
cmake -S . -B build && cmake --build build -j && (cd build && ctest)
```

`src/cylinder_fem_amr.cu` holds the driver. It includes `src/cfa/*.cuh`, so the whole solver
compiles as one translation unit.

## Run

```bash
./cylinder_fem_amr --help
./cylinder_fem_amr --steps 8000 --stats-window 1.5            # DFG 2D-2, statistics over t >= 6.5 s
./cylinder_fem_amr --case 2d3 --steps 8000                    # DFG 2D-3
./cylinder_fem_amr --uniform-level 4 --steps 8000             # uniform mesh, no adaptation
./cylinder_fem_amr --mms --uniform-level 3 --dt 2.5e-4 --steps 400   # manufactured solution
```

Each run writes the following files:

| File | Contents |
|---|---|
| `<prefix>_forces.csv` | One row per step: C_D, C_L, Δp, mesh size, probe pressures, surface-traction forces |
| `<prefix>_stats.csv` | DFG quantities, compared with the reference |
| `<prefix>_final.vtk` | Final solution (legacy ASCII VTK), plus snapshots if `--vtk-interval` is set |

## Verification and reproduction

| What | How |
|---|---|
| Unit tests | `tests/test_cfa.cu`: NVB conformity, level cap, refine/coarsen round trip, minimum level, exact transfer of linear fields, zero force at rest, manufactured-solution properties, multicolour ILU(0) and the momentum BiCGSTAB |
| Manufactured solution | `--mms`; tables with `scripts/mms_table.py` |
| Benchmark quantities | `scripts/benchmark_stats.py` (DFG definitions, works on our CSV and on FEATFLOW logs) |
| Convergence and extrapolation | `scripts/convergence_table.py`, `scripts/plot_convergence.py` |
| Remesh glitches / wall pressure | `scripts/remesh_glitch.py`, `scripts/surface_pressure.py` |
| Full campaign | `scripts/campaign.sh <binary> <outdir>` |

Reference data are not redistributed here. The FEATFLOW DFG 2D-2 values used in the paper
(level 6, Q2/P1disc) are built into `scripts/convergence_table.py`; the full time series are
available from featflow.de (DFG benchmark 2D-2). The DFG 2D-3 reference values are from
John, IJNMF 44 (2004) 777–788.

The Python scripts need only the standard library; `plot_convergence.py` and `render_vtk.py`
also need numpy and matplotlib.

## Reproducing the paper

`paper/README.md` maps every table and figure to the run directories and binaries. The runs are in
`runs/`: for each one, `cmd.txt` (exact command line, where recorded), `run.log` and `err.log`
(solver output, including the timing breakdown), `cylinder_fem_amr_forces.csv` and
`cylinder_fem_amr_stats.csv`. The VTK solution files (about 2 GB) are not included; the cylinder
wall pressure needed for the wall-pressure figure and the wall-oscillation column is extracted
into `cylinder_wall_pressure.csv`. With these,

```bash
scripts/make_paper.sh [python]     # python with numpy + matplotlib
```

regenerates every table in `paper/tables/` and every figure except the two mesh snapshots
(`fig_mesh_d3`, `fig_mesh_zoom`), which need the VTK snapshots of `runs/paper/snap_*`; rerun those
two cases with the commands in `scripts/paper_queue.sh` to obtain them. The solver source is the
state used for the review runs (`cfa_v18d_review`); `src/history/v17_2026-11-09_paper` is the
source of the binary used for most runs (v17). Both give the same forces at default options to
round-off (1e-11 over 400 steps on U3 and A1).

## Licence

MIT; see `LICENSE`.
