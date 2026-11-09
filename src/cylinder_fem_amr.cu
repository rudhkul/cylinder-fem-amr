/**
 * @file cylinder_fem_amr.cu
 * @brief 2D Incompressible Navier-Stokes Finite Element Method (FEM) Solver with
 *        Adaptive Mesh Refinement (AMR) in CUDA C++ for Flow Across a Circular Cylinder.
 *
 * Benchmark Configuration (Schäfer-Turek DFG 2D-2 Laminar Vortex Shedding):
 *   - Domain: Rectangular channel [0, 2.2] x [0, 0.41] m
 *   - Obstacle: Circular cylinder centered at (0.2, 0.2) m with diameter D = 0.1 m (R = 0.05 m)
 *   - Inflow (x = 0): Parabolic velocity profile u(0, y) = 4 * U_max * y * (H - y) / H^2, v = 0
 *                     with U_max = 1.5 m/s (mean velocity U_bar = 1.0 m/s)
 *   - Walls (y = 0, y = H) & Cylinder boundary: No-slip (u = 0, v = 0)
 *   - Outflow (x = L): Do-nothing / zero reference pressure (p = 0)
 *   - Kinematic viscosity: nu = 1.0e-3 m^2/s  =>  Reynolds number Re = U_bar * D / nu = 100
 *
 * Numerical Methods & GPU Architecture:
 *   - Spatial Discretization: Unstructured P1-P1 continuous Galerkin triangular finite elements
 *     with Streamline-Upwind/Petrov-Galerkin (SUPG) convective stabilization, skew-symmetric
 *     (energy-conserving) convection, and orthogonal-subscale (OSS) pressure stabilization with
 *     an implicit gradient projection (Codina 2001). Without OSS the equal-order projection
 *     scheme is only stable for dt >~ c h^2 and diverges on coarse cells at small dt.
 *   - Temporal Discretization: BDF2 incremental pressure-correction projection in rotational
 *     form (Guermond, Minev & Shen 2006); --time-scheme euler selects backward Euler:
 *       1. Advection-Diffusion-SUPG Tentative Velocity Predictor with BDF2 time derivative and
 *          extrapolated advecting velocity 2u^n - u^{n-1} (GPU Preconditioned BiCGSTAB)
 *       2. Pressure Poisson Equation (PPE) on P1 space with projection step 2dt/3
 *          (GPU CG, warm-started, preconditioned by smoothed-aggregation AMG: host setup per
 *          remesh / every --amg-rebuild steps, GPU V-cycle; mesh-independent ~15-25 iterations)
 *       3. Lumped-Mass Divergence-Free Velocity Projection & Nodal Vorticity Recovery
 *       4. Rotational pressure update p += dp - rho*nu*div(u*) (default; --pressure-update standard
 *          selects the non-rotational scheme). div u* is taken as (dt'/rho) Lap_h dp = -(dt'/rho)
 *          M_L^{-1} K dp: the discrete divergence of u* also contains the OSS stabilization of the
 *          whole pressure, which fed back into p every step and made the pressure error grow like
 *          1/dt (manufactured solution) and uniform meshes diverge (--rotational-div velocity)
 *     The first step is backward Euler; u^{n-1} is transferred through every remesh, so the
 *     scheme stays second order under AMR.
 *   - Hydrodynamic forces: volume-integral (variational) drag/lift evaluation
 *     c_D = -2/(U^2 D) [ (du/dt, v_d) + (u.grad u, v_d) + nu (grad u, grad v_d) - (p, div v_d) ]
 *     (John, 2004), which is far more accurate than integrating surface tractions.
 *
 * Adaptive Mesh Refinement (AMR):
 *   - Refinement indicators (GPU), in m/s, measured relative to the undisturbed inflow
 *     (Poiseuille) profile so that the channel flow itself never attracts refinement:
 *       disturbance vorticity, eta_v = h_K * |omega_h - omega_P|: the velocity jump across
 *         the element from rotation and shear; captures the cylinder boundary layer, the
 *         separated shear layers, the shed vortices and the wake.
 *       disturbance interpolation error, eta_i = max_edges |e^T H e| / 8: captures the
 *         irrotational stagnation and acceleration flow around the cylinder that sets the
 *         surface pressure (and hence drag and lift) but carries no vorticity.
 *     Both use the recovered (area-averaged) gradient, which is smooth and free of
 *     mesh-orientation patterns.
 *   - Marking: multi-criterion tagging. Each indicator is normalised by its own tolerance
 *     (--amr-tol-vort, --amr-tol-interp); an element is refined if any score exceeds 1, by as
 *     many levels as its worst criterion needs (eta_v ~ h, eta_i ~ h^2), and coarsened where
 *     the merged parent is predicted below coarsen_safety on every criterion. The band between
 *     coarsen_safety and 1 is the hysteresis that prevents refine/coarsen cycling; nothing is
 *     refined once the flow is resolved. Coarse cells surrounded by finer ones are filled in.
 *   - Advective look-ahead: indicators above tol are traced downstream with the local velocity
 *     over one adaptation interval and imposed (scaled by (h_T/h_K)^q) on the elements along
 *     the path, so shed vortices move into mesh that is already refined. A vertex-neighbour
 *     buffer layer and an element budget (soft: closure may add a few elements) complete it.
 *   - Mesh adaptation: Newest-Vertex Bisection (NVB) on a refinement forest. The base mesh
 *     is labelled so that every refinement edge is shared by its neighbour (Stevenson's
 *     matching condition), which guarantees
 *       * conforming meshes (no hanging nodes) after every adaptation,
 *       * a bounded number of element similarity classes (minimum angle bounded below),
 *       * conformity closure only ever bisects coarser elements (the level cap is exact).
 *     One AMR level equals two bisections (h halved). Coarsening reverses bisections by
 *     removing newest vertices whose bisection patch consists of unrefined, coarsening-marked
 *     elements, up to one level per adaptation.
 *   - Curved boundary: midpoints of cylinder edges are projected onto the exact circle.
 *   - Solution transfer: P1 interpolation on refinement (exact for the P1 field), nodal
 *     injection on coarsening (exact P1 interpolant on the nested coarse space), followed by
 *     a re-projection of both velocity levels u^n, u^{n-1} with the scheme's own (OSS) pressure
 *     step, which is nearly the identity on an unchanged mesh. A zero-divergence re-projection
 *     of u^n only (the earlier behaviour) left the BDF2 history inconsistent and put spikes of
 *     ~100x the smooth step-to-step change into C_D, C_L and delta_p after every remesh.
 *   - Elements within --keep-radius (2.5 D by default) of the cylinder centre are never
 *     coarsened, so the mesh that carries the forces becomes static once the shedding is
 *     periodic: refine/coarsen cycling there changed the discretization error under the force
 *     functional at every remesh (with 1.0 D the glitches remained, with 2.5 D they vanished).
 *   - Every adapted mesh is validated (positive orientation, edge manifoldness, no
 *     non-boundary free edges = no hanging nodes).
 *
 * Compilation:
 *   nvcc -O3 -arch=native -std=c++17 -Xcompiler -fopenmp cylinder_fem_amr.cu -o cylinder_fem_amr
 *   (or an explicit target, e.g. -arch=sm_80, sm_86, sm_89, sm_120). OpenMP threads the host-side
 *   AMR look-ahead and the AMG setup; without -fopenmp the code runs serially with identical results.
 *
 * Usage:
 *   ./cylinder_fem_amr [options]            (see --help)
 *   ./cylinder_fem_amr [num_steps] [amr_interval] [max_amr_level] [amr_tol_vort]   (legacy)
 *   Outputs ParaView .vtk unstructured grid files and a per-step C_D, C_L, delta_p CSV history.
 *
 * DFG 2D-2 benchmark run (periodic shedding is established by t ~ 3-4 s):
 *   ./cylinder_fem_amr --steps 6000 --stats-window 2
 *   The statistics follow the DFG definitions per lift period (C_D,max, C_L,max, St and
 *   delta_p(t0 + 1/(2f)), t0 = time of maximum lift) and are compared with the FEATFLOW
 *   Q2/P1disc level-6 reference (reference/featflow_2d2): C_D,max 3.22739, C_L,max 0.98658,
 *   St 0.30184, delta_p 2.48520 (DFG 1996 ranges: 3.22-3.24, 0.99-1.01, 0.295-0.305, 2.46-2.50).
 *
 * Verification: --mms runs a manufactured solution on a uniform mesh (--uniform-level L) and
 * reports L2/H1 errors; tests/test_cfa.cu holds unit tests (CMakeLists.txt: ctest).
 */

// The solver is split into headers that form one translation unit (order matters).
#include "cfa/common.cuh"
#include "cfa/mesh.cuh"
#include "cfa/fem_kernels.cuh"
#include "cfa/linalg.cuh"
#include "cfa/amr.cuh"
#include "cfa/diagnostics.cuh"
#include "cfa/device.cuh"
#include "cfa/options.cuh"

// ============================================================================
// Main Simulation Driver
// ============================================================================
#ifndef CFA_NO_MAIN   // tests/ include this file and provide their own main
int main(int argc, char** argv) {
    RunOptions opt;
    if (!parse_options(argc, argv, opt)) return 0;
    g_outlet_dirichlet = opt.outlet_dirichlet;
    {
        int flag = opt.outlet_dirichlet ? 1 : 0;
        CUDA_CHECK(cudaMemcpyToSymbol(c_outlet_dirichlet, &flag, sizeof(int)));
        if (flag) printf("[Setup] outlet: inflow parabola (Dirichlet), pressure increment pinned at one outlet node\n");
    }
    const double dt = opt.dt;
    opt.amr.lookahead_time = opt.lookahead * opt.amr_interval * dt;
    const double rot_coeff = opt.rotational ? SimConfig::RHO * SimConfig::NU : 0.0;
    const double force_scale = -2.0 / (SimConfig::U_MEAN * SimConfig::U_MEAN * 2.0 * SimConfig::R_CYL);
    const double surface_scale = -force_scale;
    int host_threads = 1;
#ifdef _OPENMP
    // One thread per logical CPU oversubscribes the cores (SMT siblings, the busy-waiting CUDA
    // host thread, other jobs) and spin barriers then stall the AMG setup by 10x or more. Unless
    // OMP_NUM_THREADS says otherwise, use half the logical CPUs, at most 8 (scaling flattens there).
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / 2)));
    host_threads = omp_get_max_threads();
#endif

    std::cout << "========================================================================\n";
    std::cout << "  CUDA 2D FEM + AMR Incompressible Navier-Stokes Cylinder Flow Solver\n";
    std::cout << "  Domain: [0, " << SimConfig::L_CHANNEL << "] x [0, " << SimConfig::H_CHANNEL
              << "] m | Cylinder: (" << SimConfig::X_CYL << ", " << SimConfig::Y_CYL
              << "), R = " << SimConfig::R_CYL << " m\n";
    std::cout << "  Re = " << (SimConfig::U_MEAN * 2.0 * SimConfig::R_CYL / SimConfig::NU)
              << " | dt = " << dt << " s | Steps = " << opt.num_steps
              << " | AMR Interval = " << opt.amr_interval << " (Max Level = " << opt.amr.max_level << ")\n";
    std::cout << "  AMR: NVB refine+coarsen | tol vorticity " << opt.amr.tol_vort
              << " m/s, interpolation " << opt.amr.tol_interp << " m/s | coarsen below "
              << opt.amr.coarsen_safety << " x tol | look-ahead " << opt.amr.lookahead_time
              << " s | buffer " << opt.amr.buffer_layers
              << " | element budget " << opt.amr.max_elems
              << " | remesh " << (opt.amr_overlap > 0 ? "overlapped by " + std::to_string(opt.amr_overlap) + " steps" : "synchronous");
    if (opt.amr.tol_radius > 0.0)
        std::cout << " | tol x (d / " << opt.amr.tol_radius << " m)^" << opt.amr.tol_power << " beyond d = "
                  << opt.amr.tol_radius << " m";
    std::cout << "\n";
    std::cout << "  Time scheme: " << (opt.bdf2 ? "BDF2" : "backward Euler") << " | pressure update: "
              << (opt.rotational ? "rotational" : "standard") << " incremental projection"
              << " | host threads " << host_threads << "\n";
    std::cout << "  Solvers: momentum BiCGSTAB + " << (opt.momentum_ilu ? "multicolour ILU(0)" : "Jacobi")
              << " | pressure CG + " << (opt.amg ? "SA-AMG" : "Jacobi") << "\n";
    std::cout << "========================================================================\n";

    // 1. Generate Initial Body-Fitted Cylinder Mesh (NVB-labelled) & CSR Adjacency Topology
    Mesh2D mesh;
    generate_cylinder_mesh(mesh, /*n_quarter=*/8, /*n_radial=*/7, /*n_wake_x=*/28);
    MeshQuality q0 = finalize_mesh(mesh);

    std::cout << std::fixed << std::setprecision(4)
              << "[Init] Base Unstructured Mesh: " << mesh.num_nodes << " nodes, "
              << mesh.num_elems << " triangles, " << mesh.nnz << " CSR nonzeros, min angle "
              << std::setprecision(2) << q0.min_angle_deg << " deg.\n" << std::setprecision(4);

    // 2. Initialize Flow Field with Smooth Potential-Like Channel Profile (2D-3: at rest)
    const double inflow0 = opt.case_2d3 ? 0.0 : SimConfig::U_MAX;
    std::vector<double> h_u, h_v, h_p, h_uo, h_vo;
    std::vector<double> h_vort(mesh.num_nodes, 0.0);
    std::vector<double> h_elem_err(2 * mesh.num_elems, 0.0);
    set_initial_condition(mesh, h_u, h_v, h_p, inflow0);
    h_uo = h_u;
    h_vo = h_v;

    AdaptiveMesh amr(mesh, h_u, h_v, h_p);
    amr.extract(mesh, h_u, h_v, h_p, h_uo, h_vo);
    finalize_mesh(mesh);
    if (opt.uniform_level > 0 || opt.amr.min_level > 0) {   // uniform (fixed) mesh, or the AMR floor
        AmrStats st = amr.refine_uniform(std::max(opt.uniform_level, opt.amr.min_level));
        amr.extract(mesh, h_u, h_v, h_p, h_uo, h_vo);
        MeshQuality q = finalize_mesh(mesh);
        set_initial_condition(mesh, h_u, h_v, h_p, inflow0);
        h_uo = h_u;
        h_vo = h_v;
        amr.scatter_solution(h_u, h_v, h_p, h_uo, h_vo);
        print_amr_report("uniform", st, mesh, q, -1);
    }

    if (opt.mms) {   // exact solution at t = 0 (the first step is backward Euler)
        for (int i = 0; i < mesh.num_nodes; ++i) {
            MmsValues r = mms_exact(mesh.x[i], mesh.y[i], 0.0);
            h_u[i] = h_uo[i] = r.u;
            h_v[i] = h_vo[i] = r.v;
            h_p[i] = r.p;
        }
    }
    DeviceMeshBuffers gpu_buf;
    gpu_buf.sync_from_host(mesh, h_u, h_v, h_p, h_uo, h_vo);
    GpuLinearSolver solver;

    // Pre-adapt the mesh to the initial condition; the analytic field is re-evaluated on the new nodes
    if (opt.init_amr > 0) {
        for (int cycle = 0; cycle < opt.init_amr; ++cycle) {
            compute_error_indicator(gpu_buf, inflow0);
            gpu_buf.copy_solution_to_host(h_u, h_v, h_p, h_uo, h_vo, h_vort, h_elem_err);
            amr.scatter_solution(h_u, h_v, h_p, h_uo, h_vo);
            amr.inlet_umax = inflow0;
            AmrStats st = amr.adapt(h_elem_err, opt.amr);
            amr.extract(mesh, h_u, h_v, h_p, h_uo, h_vo);
            MeshQuality q = finalize_mesh(mesh);
            set_initial_condition(mesh, h_u, h_v, h_p, inflow0);
            h_uo = h_u;
            h_vo = h_v;
            amr.scatter_solution(h_u, h_v, h_p, h_uo, h_vo);
            gpu_buf.sync_from_host(mesh, h_u, h_v, h_p, h_uo, h_vo);
            print_amr_report("init", st, mesh, q, -1);
        }
    }

    std::ofstream csv(opt.prefix + "_forces.csv");
    if (!csv.is_open()) {
        fprintf(stderr, "[Output] cannot open %s_forces.csv\n", opt.prefix.c_str());
        return EXIT_FAILURE;
    }
    csv << "step,time,C_D,C_L,delta_p,nodes,elements,p_front,p_back,C_D_surface,C_L_surface\n" << std::setprecision(10);

    ForceHistory history;
    int nonconverged_solves = 0;
    // Counts a solve that missed its tolerance and reports the first few on stderr
    auto note_nonconverged = [&nonconverged_solves](const GpuLinearSolver& sv, int at_step, const char* what, int iters) {
        if (sv.converged) return;
        if (++nonconverged_solves <= 20)
            fprintf(stderr, "[Solver] step %d: %s solve not converged after %d iterations (|r|/|r0| = %.2e)\n",
                    at_step, what, iters, sv.last_rel_residual);
    };
    const bool use_amg = opt.amg;
    AmgPreconditioner amg;
    bool amg_dirty = true;
    bool phi_pending = false;   // --reproject-shift: d_phi enters the next PPE
    int amg_built_step = 0;
    int amg_remaps_in_row = 0;   // remeshes since the last full AMG setup (--amg-remap)
    // --amr-overlap: remesh built by a worker thread, joined at join_step (see Step F)
    struct PendingRemesh {
        std::future<bool> job;
        int join_step = -1;
        AmrStats st;
        MeshQuality q;
        Mesh2D mesh;
        std::vector<double> u, v, p, uo, vo;
        std::vector<int> old_vids;
        double extract_ms = 0.0, csr_ms = 0.0, validate_ms = 0.0;
    } pend;
    double amg_setup_ms_total = 0.0;
    auto wall_start = std::chrono::steady_clock::now();
    double amr_ms_total = 0.0;
    double momentum_ms_total = 0.0, pressure_ms_total = 0.0;   // assembly + solves (host clock)
    RemeshProfile remesh_prof;

    // Momentum preconditioner: the colouring follows the mesh, the factors are redone every step
    MulticolourIlu0 mom_ilu;
    GpuBicgstab2 mom_solver;
    if (opt.momentum_ilu)
        mom_ilu.set_pattern(mesh.num_nodes, mesh.csr_row_ptr, mesh.csr_col_ind, gpu_buf.d_row_ptr, gpu_buf.d_col_ind);

    // 3. Time-Stepping & Dynamic Adaptive Mesh Refinement Loop
    for (int step = 1; step <= opt.num_steps; ++step) {
        int n_nodes = gpu_buf.n_nodes;
        int n_elems = gpu_buf.n_elems;
        int nnz     = gpu_buf.nnz;

        int threads = 256;
        int blocks_nodes = (n_nodes + threads - 1) / threads;
        int blocks_elems = (n_elems + threads - 1) / threads;

        // Time levels: u_prev <- u^{n-1}, u_old <- u^n. BDF2 needs one completed step of history
        // (the first step is backward Euler); the history is transferred across remeshing.
        const bool use_bdf2 = opt.bdf2 && step > 1;
        const double dt_proj = use_bdf2 ? (2.0 / 3.0) * dt : dt;   // projection step of the scheme
        const double a0 = use_bdf2 ? 1.5 / dt : 1.0 / dt;
        const double a1 = use_bdf2 ? 2.0 / dt : 1.0 / dt;
        const double a2 = use_bdf2 ? -0.5 / dt : 0.0;
        const double c1 = use_bdf2 ? 2.0 : 1.0;
        const double c2 = use_bdf2 ? -1.0 : 0.0;
        CUDA_CHECK(cudaMemcpy(gpu_buf.d_u_prev, gpu_buf.d_u_old, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_buf.d_v_prev, gpu_buf.d_v_old, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_buf.d_u_old, gpu_buf.d_u, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_buf.d_v_old, gpu_buf.d_v, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));

        // --- Step A: Assemble & Solve SUPG-Stabilized Momentum Predictor ---
        auto t_phase = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemset(gpu_buf.d_A_val,       0, nnz * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_rhs_u,       0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_rhs_v,       0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_mass_lumped, 0, n_nodes * sizeof(double)));

        // MMS body force at t^{n+1} (grad_dp_x/y are free until the pressure step)
        if (opt.mms) {
            k_mms_forcing<<<blocks_nodes, threads>>>(n_nodes, gpu_buf.d_x, gpu_buf.d_y, step * dt,
                                                    gpu_buf.d_grad_dp_x, gpu_buf.d_grad_dp_y);
        }
        k_assemble_momentum_fem<<<blocks_elems, threads>>>(
            n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_u_prev, gpu_buf.d_v_prev, gpu_buf.d_p,
            dt, SimConfig::NU, SimConfig::RHO, a0, a1, a2, c1, c2,
            opt.supg_tau_time > 0.0 ? opt.supg_tau_time : dt,
            opt.skew_convection ? 0.5 : 0.0,
            gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
            gpu_buf.d_A_val, gpu_buf.d_rhs_u, gpu_buf.d_rhs_v, gpu_buf.d_mass_lumped,
            opt.mms ? gpu_buf.d_grad_dp_x : nullptr, opt.mms ? gpu_buf.d_grad_dp_y : nullptr
        );

        k_apply_bc_momentum<<<blocks_nodes, threads>>>(
            n_nodes, gpu_buf.d_bc_tag, gpu_buf.d_y,
            inflow_umax(step * dt, opt.case_2d3), SimConfig::H_CHANNEL,
            gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
            gpu_buf.d_A_val, gpu_buf.d_rhs_u, gpu_buf.d_rhs_v,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_x, opt.mms ? step * dt : -1.0
        );
        CUDA_CHECK_LAUNCH();

        // Solver studies: int32 n, nnz; row_ptr, col_ind (int32); val, rhs_u, rhs_v, u0, v0, x, y
        // (float64); bc_tag (int32); then the solutions u, v (float64) appended after the solves
        const bool dump_mom = (step == opt.dump_systems);
        std::ofstream dump;
        auto dump_dev = [&](const void* d, size_t bytes) {
            std::vector<char> h(bytes);
            CUDA_CHECK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
            dump.write(h.data(), (std::streamsize)bytes);
        };
        if (dump_mom) {
            dump.open(opt.prefix + "_momentum.bin", std::ios::binary);
            int hdr[2] = {n_nodes, nnz};
            dump.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            dump_dev(gpu_buf.d_row_ptr, (n_nodes + 1) * sizeof(int));
            dump_dev(gpu_buf.d_col_ind, nnz * sizeof(int));
            dump_dev(gpu_buf.d_A_val,   nnz * sizeof(double));
            for (const double* d : std::initializer_list<const double*>{gpu_buf.d_rhs_u, gpu_buf.d_rhs_v, gpu_buf.d_u, gpu_buf.d_v,
                                    gpu_buf.d_x, gpu_buf.d_y})
                dump_dev(d, n_nodes * sizeof(double));
            dump_dev(gpu_buf.d_bc_tag, n_nodes * sizeof(int));
        }

        int it_u, it_v;
        if (opt.momentum_ilu) {   // u and v together, multicolour ILU(0)
            mom_ilu.factor(gpu_buf.d_A_val);
            mom_solver.solve(n_nodes, gpu_buf.d_row_ptr, gpu_buf.d_col_ind, gpu_buf.d_A_val,
                             gpu_buf.d_rhs_u, gpu_buf.d_rhs_v, gpu_buf.d_u, gpu_buf.d_v, 150, 1.0e-7, mom_ilu);
            it_u = mom_solver.iters[0];
            it_v = mom_solver.iters[1];
            for (int c = 0; c < 2; ++c) {
                if (mom_solver.converged[c]) continue;
                if (++nonconverged_solves <= 20)
                    fprintf(stderr, "[Solver] step %d: momentum %c solve not converged after %d iterations (|r|/|r0| = %.2e)\n",
                            step, c ? 'v' : 'u', mom_solver.iters[c], mom_solver.last_rel_residual[c]);
            }
        } else {                  // Jacobi, one component at a time (the pre-2026-10-22 solver)
            it_u = solver.solve_bicgstab(n_nodes, gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
                                         gpu_buf.d_A_val, gpu_buf.d_rhs_u, gpu_buf.d_u, 150, 1.0e-7);
            note_nonconverged(solver, step, "momentum u", it_u);
            it_v = solver.solve_bicgstab(n_nodes, gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
                                         gpu_buf.d_A_val, gpu_buf.d_rhs_v, gpu_buf.d_v, 150, 1.0e-7);
            note_nonconverged(solver, step, "momentum v", it_v);
        }
        if (dump_mom) {
            dump_dev(gpu_buf.d_u, n_nodes * sizeof(double));
            dump_dev(gpu_buf.d_v, n_nodes * sizeof(double));
            dump.close();
            printf("[Dump] momentum system of step %d (%d nodes, nnz %d, iterations %d/%d) -> %s_momentum.bin\n",
                   step, n_nodes, nnz, it_u, it_v, opt.prefix.c_str());
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        momentum_ms_total += ms_since(t_phase);
        t_phase = std::chrono::steady_clock::now();

        // --- Step B: Assemble & Solve Pressure Poisson Equation (PPE), warm-started from the last dp ---
        CUDA_CHECK(cudaMemset(gpu_buf.d_Kp_val, 0, nnz * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_rhs_p,  0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_div_num, 0, n_nodes * sizeof(double)));

        // OSS pressure stabilization: weights s_K and G(p^n) (grad_dp_x/y are reused as
        // scratch; they are cleared again before the velocity correction)
        const bool oss = opt.pressure_stab > 0.0;
        CUDA_CHECK(cudaMemset(gpu_buf.d_grad_dp_x, 0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_grad_dp_y, 0, n_nodes * sizeof(double)));
        // --reproject-shift (first step after a scheme re-projection): the stabilization term of
        // this PPE uses p^n + phi, so the step continues from the constraint the re-projection
        // imposed, as in Braack, Lang & Taschenberger (2013), Eq. (21); p itself is unchanged
        const bool shift_now = phi_pending && oss;
        if (shift_now) k_axpy<<<blocks_nodes, threads>>>(n_nodes, 1.0, gpu_buf.d_phi, gpu_buf.d_p);
        phi_pending = false;
        if (oss) {
            k_compute_oss_weights<<<blocks_elems, threads>>>(
                n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y, gpu_buf.d_u, gpu_buf.d_v,
                dt_proj, SimConfig::RHO, SimConfig::NU, opt.pressure_stab, gpu_buf.d_elem_s);
            k_accumulate_scalar_gradient<<<blocks_elems, threads>>>(
                n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y, gpu_buf.d_p, gpu_buf.d_elem_s,
                gpu_buf.d_grad_dp_x, gpu_buf.d_grad_dp_y);
        }

        k_assemble_pressure_poisson<<<blocks_elems, threads>>>(
            n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_u, gpu_buf.d_v,
            gpu_buf.d_p, gpu_buf.d_grad_dp_x, gpu_buf.d_grad_dp_y, gpu_buf.d_mass_lumped, gpu_buf.d_elem_s,
            dt_proj, SimConfig::RHO, opt.pressure_stab,
            gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
            gpu_buf.d_Kp_val, gpu_buf.d_rhs_p, gpu_buf.d_div_num
        );
        if (shift_now) k_axpy<<<blocks_nodes, threads>>>(n_nodes, -1.0, gpu_buf.d_phi, gpu_buf.d_p);

        apply_bc_pressure(n_nodes, gpu_buf.d_bc_tag, gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
                          gpu_buf.d_Kp_val, gpu_buf.d_rhs_p, gpu_buf.d_dp, gpu_buf.d_mass_lumped);
        CUDA_CHECK_LAUNCH();

        // Solver studies: int32 n, nnz; row_ptr, col_ind (int32); K_p values, rhs, initial guess
        // (float64). The matrix-free OSS projection is not included.
        if (dump_mom) {
            dump.open(opt.prefix + "_pressure.bin", std::ios::binary);
            int hdr[2] = {n_nodes, nnz};
            dump.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            dump_dev(gpu_buf.d_row_ptr, (n_nodes + 1) * sizeof(int));
            dump_dev(gpu_buf.d_col_ind, nnz * sizeof(int));
            dump_dev(gpu_buf.d_Kp_val,  nnz * sizeof(double));
            dump_dev(gpu_buf.d_rhs_p,   n_nodes * sizeof(double));
            dump_dev(gpu_buf.d_dp,      n_nodes * sizeof(double));
            dump.close();
        }

        // -B^T M_L^{-1} B part of the OSS operator, assembled for this step's weights, applied inside CG
        if (oss) assemble_oss_projection(gpu_buf);
        std::function<void(const double*, double*)> oss_projection = [&](const double* in, double* out) {
            apply_oss_projection(gpu_buf, in, out);
        };
        // SA-AMG preconditioner, rebuilt after every remesh and every amg_rebuild steps (the
        // operator values drift with the OSS weights); a frozen SPD preconditioner keeps CG valid
        if (use_amg && (amg_dirty || step - amg_built_step >= opt.amg_rebuild)) {
            amg.setup(n_nodes, nnz, gpu_buf.d_row_ptr, gpu_buf.d_col_ind, gpu_buf.d_Kp_val);
            amg_dirty = false;
            amg_built_step = step;
            amg_setup_ms_total += amg.setup_ms;
        }
        std::function<void(const double*, double*)> amg_prec = [&](const double* r, double* z) {
            amg.apply(r, z);
        };
        int it_p = solver.solve_pcg(n_nodes, gpu_buf.d_row_ptr, gpu_buf.d_col_ind,
                                    gpu_buf.d_Kp_val, gpu_buf.d_rhs_p, gpu_buf.d_dp, 4000, 1.0e-7,
                                    oss ? &oss_projection : nullptr,
                                    use_amg ? &amg_prec : nullptr);
        note_nonconverged(solver, step, "pressure", it_p);

        // Rotational correction -rho nu div u*: in the continuous scheme div u* = (dt'/rho) Lap dp.
        // The discrete divergence of u* also carries the OSS stabilization of the whole pressure,
        // (dt'/rho) S p^{n+1}, which would be fed back into p every step. That accumulates as
        // O(nu tau / dt) and destabilizes the scheme (MMS: pressure error ~ 1/dt; uniform meshes
        // diverge), so by default the divergence is taken as -(dt'/rho) M^{-1} K dp.
        if (opt.rotational && opt.rotational_from_dp) {
            CUDA_CHECK(cudaMemset(gpu_buf.d_div_num, 0, n_nodes * sizeof(double)));
            k_stiffness_apply<<<blocks_elems, threads>>>(
                n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y, gpu_buf.d_dp,
                -dt_proj / SimConfig::RHO, gpu_buf.d_div_num);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        pressure_ms_total += ms_since(t_phase);

        // --- Step C: Divergence-Free Velocity Projection & Vorticity Recovery ---
        CUDA_CHECK(cudaMemset(gpu_buf.d_grad_dp_x, 0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_grad_dp_y, 0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(gpu_buf.d_vort_num,  0, n_nodes * sizeof(double)));

        k_accumulate_grad_dp_and_vorticity<<<blocks_elems, threads>>>(
            n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_dp, gpu_buf.d_u, gpu_buf.d_v,
            gpu_buf.d_grad_dp_x, gpu_buf.d_grad_dp_y, gpu_buf.d_vort_num
        );

        k_velocity_pressure_correction<<<blocks_nodes, threads>>>(
            n_nodes, gpu_buf.d_bc_tag, dt_proj, SimConfig::RHO, /*p_update=*/1.0, rot_coeff,
            gpu_buf.d_div_num, gpu_buf.d_mass_lumped, gpu_buf.d_grad_dp_x, gpu_buf.d_grad_dp_y,
            gpu_buf.d_vort_num, gpu_buf.d_dp,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_p, gpu_buf.d_vort
        );

        // --- Step D: Volume-Integral Drag/Lift & Surface Pressure Probes ---
        CUDA_CHECK(cudaMemset(gpu_buf.d_diag, 0, 6 * sizeof(double)));
        k_volume_forces<<<blocks_elems, threads>>>(
            n_elems, gpu_buf.d_elem_v, gpu_buf.d_bc_tag, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_u_old, gpu_buf.d_v_old,
            gpu_buf.d_u_prev, gpu_buf.d_v_prev, gpu_buf.d_p,
            a0, -a1, -a2, SimConfig::NU, SimConfig::RHO, gpu_buf.d_diag
        );
        k_sample_pressure_probes<<<1, 1>>>(
            gpu_buf.d_p,
            gpu_buf.probe_front.a, gpu_buf.probe_front.b, gpu_buf.probe_front.w,
            gpu_buf.probe_back.a,  gpu_buf.probe_back.b,  gpu_buf.probe_back.w,
            gpu_buf.d_diag
        );
        k_surface_forces<<<(gpu_buf.n_cyl_edges + 127) / 128, 128>>>(
            gpu_buf.n_cyl_edges, gpu_buf.d_cyl_edges, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_p, SimConfig::NU, SimConfig::RHO,
            SimConfig::X_CYL, SimConfig::Y_CYL, gpu_buf.d_diag + 4);
        CUDA_CHECK_LAUNCH();
        double diag[6];
        CUDA_CHECK(cudaMemcpy(diag, gpu_buf.d_diag, 6 * sizeof(double), cudaMemcpyDeviceToHost));

        double t_now   = step * dt;
        double C_D     = force_scale * diag[0];
        double C_L     = force_scale * diag[1];
        double delta_p = diag[2] - diag[3];
        if (!std::isfinite(C_D) || !std::isfinite(C_L) || !std::isfinite(delta_p)) {
            fprintf(stderr, "[Step %d] non-finite solution detected, aborting\n", step);
            return EXIT_FAILURE;
        }
        history.add(t_now, C_D, C_L, delta_p);
        csv << step << "," << t_now << "," << C_D << "," << C_L << "," << delta_p << ","
            << n_nodes << "," << n_elems << "," << diag[2] << "," << diag[3] << ","
            << surface_scale * diag[4] << "," << surface_scale * diag[5] << "\n";

        if (step % opt.print_interval == 0 || step == 1 || step == opt.num_steps) {
            std::cout << std::fixed << std::setprecision(4)
                      << "[Step " << std::setw(5) << step << " | t = " << t_now << " s] "
                      << "Nodes: " << std::setw(6) << n_nodes
                      << " | Elems: " << std::setw(6) << n_elems
                      << " | Iters(u,v,p): (" << it_u << "," << it_v << "," << it_p << ")"
                      << " | C_D = " << C_D << " | C_L = " << C_L
                      << " | dP = " << delta_p << std::endl;   // flush: long runs are followed via tail -f
        }

        // --- Step E: Periodic VTK Snapshots ---
        if (opt.vtk_interval > 0 && step % opt.vtk_interval == 0 && step < opt.num_steps) {
            compute_error_indicator(gpu_buf);
            gpu_buf.copy_solution_to_host(h_u, h_v, h_p, h_uo, h_vo, h_vort, h_elem_err);
            std::ostringstream name;
            name << opt.prefix << "_" << std::setw(6) << std::setfill('0') << step << ".vtk";
            write_vtk_unstructured(name.str(), mesh, h_u, h_v, h_p, h_vort, h_elem_err, opt.amr);
        }

        // --- Step F: Adaptive Mesh Refinement & Coarsening ---
        // Default: indicators, new mesh, transfer and re-projection all at step n. --amr-overlap K:
        // the host builds the mesh from the state at step n in a worker thread while the GPU takes
        // K more steps on the old mesh; at step n + K the state of that step is moved to the new
        // mesh with the rules of adapt() (AdaptiveMesh::retransfer). The join step is fixed, so a
        // run does not depend on thread timing. The look-ahead horizon grows by K steps, since the
        // mesh is used until step n + K + interval.
        const bool overlap = opt.amr_overlap > 0;
        const bool remesh_due = opt.amr_interval > 0 && step % opt.amr_interval == 0 && step < opt.num_steps
                                && !(overlap && step + opt.amr_overlap >= opt.num_steps);
        auto t0 = std::chrono::steady_clock::now();
        auto tp = t0;
        auto lap = [&tp]() { double ms = ms_since(tp); tp = std::chrono::steady_clock::now(); return ms; };
        // --amg-remap: keep the old node identities (persistent vertex + coordinates; vertex
        // slots are recycled, so a slot alone does not identify a node) for AmgPreconditioner::remap
        const bool amg_remap_now = use_amg && opt.remesh_projection >= 3 && opt.amg_remap > 0
                                   && amg_remaps_in_row < opt.amg_remap;
        // Re-projection of the transferred velocities (after a remesh, or --reproject-test)
        auto reproject = [&](const std::function<void()>& build_amg) -> int {
            int proj_iters = 0;
            const bool scheme_proj = opt.remesh_projection >= 3;
            if (scheme_proj) {
                const int remaps_before = amg.remaps;
                proj_iters = project_scheme_consistent(gpu_buf, solver, opt.bdf2 ? (2.0 / 3.0) * dt : dt,
                                                       opt.pressure_stab, use_amg ? &amg : nullptr,
                                                       nonconverged_solves, build_amg,
                                                       opt.remesh_projection == 4,
                                                       opt.reproject_shift ? gpu_buf.d_phi : nullptr);
                amg_remaps_in_row = amg.remaps > remaps_before ? amg_remaps_in_row + 1 : 0;
                phi_pending = opt.reproject_shift;
            }
            if (!scheme_proj && opt.remesh_projection >= 2) {
                proj_iters += project_divergence_free(gpu_buf, solver, dt, use_amg ? &amg : nullptr, true,
                                                      gpu_buf.d_u_old, gpu_buf.d_v_old, gpu_buf.d_oss_gx);
                nonconverged_solves += solver.converged ? 0 : 1;
            }
            if (!scheme_proj && opt.remesh_projection >= 1) {
                proj_iters += project_divergence_free(gpu_buf, solver, dt, use_amg ? &amg : nullptr,
                                                      opt.remesh_projection == 1,
                                                      gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_vort);
            }
            // The hierarchy now belongs to the cleanup operator K. Reusing it for the stabilized
            // PPE (K + S) was tried: PPE iterations rose from ~20 to ~44, so it is rebuilt.
            amg_dirty = true;
            if (scheme_proj && use_amg) {   // built for K + S: the next PPE can use it
                amg_dirty = false;
                amg_built_step = step;
            }
            if (!scheme_proj) nonconverged_solves += solver.converged ? 0 : 1;
            return proj_iters;
        };
        // Upload of the new mesh and solution, ILU pattern, re-projection, report
        auto finish_remesh = [&](const AmrStats& st, const MeshQuality& q, const std::vector<int>& old_vids,
                                 const std::vector<double>& old_x, const std::vector<double>& old_y) {
            gpu_buf.sync_from_host(mesh, h_u, h_v, h_p, h_uo, h_vo);
            CUDA_CHECK(cudaDeviceSynchronize());
            remesh_prof.upload += lap();
            if (opt.momentum_ilu)
                mom_ilu.set_pattern(mesh.num_nodes, mesh.csr_row_ptr, mesh.csr_col_ind, gpu_buf.d_row_ptr, gpu_buf.d_col_ind);
            remesh_prof.colouring += lap();
            // Both time levels are re-projected, so the BDF2 history u^{n-1} is consistent with the
            // scheme on the new mesh too (otherwise the forces jump after every remesh)
            std::function<void()> build_amg;
            if (amg_remap_now) build_amg = [&]() {
                std::vector<int> slot_to_old(amr.num_vertex_slots(), -1), new_to_old(mesh.num_nodes);
                for (int j = 0; j < (int)old_vids.size(); ++j)
                    if (old_vids[j] < (int)slot_to_old.size()) slot_to_old[old_vids[j]] = j;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < mesh.num_nodes; ++i) {
                    int j = slot_to_old[amr.node_vertices()[i]];
                    new_to_old[i] = (j >= 0 && old_x[j] == mesh.x[i] && old_y[j] == mesh.y[i]) ? j : -1;
                }
                amg.remap(gpu_buf.n_nodes, gpu_buf.nnz, gpu_buf.d_row_ptr, gpu_buf.d_col_ind, gpu_buf.d_Kp_val,
                          mesh.csr_row_ptr.data(), mesh.csr_col_ind.data(), new_to_old);
            };
            int proj_iters = reproject(build_amg);
            double proj_ms = lap();
            double proj_amg_ms = (use_amg && opt.remesh_projection > 0) ? amg.setup_ms : 0.0;
            remesh_prof.amg_setup  += proj_amg_ms;
            remesh_prof.projection += proj_ms - proj_amg_ms;
            amr_ms_total += ms_since(t0);
            print_amr_report("remesh", st, mesh, q, proj_iters);
        };

        // --reproject-test: the remesh re-projection on the unchanged mesh (identity test)
        if (opt.reproject_test > 0 && step % opt.reproject_test == 0 && step < opt.num_steps) {
            const int n = gpu_buf.n_nodes;
            std::vector<double> u0(n), v0(n), u1(n), v1(n);
            CUDA_CHECK(cudaMemcpy(u0.data(), gpu_buf.d_u, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v0.data(), gpu_buf.d_v, n * sizeof(double), cudaMemcpyDeviceToHost));
            reproject(nullptr);
            CUDA_CHECK(cudaMemcpy(u1.data(), gpu_buf.d_u, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v1.data(), gpu_buf.d_v, n * sizeof(double), cudaMemcpyDeviceToHost));
            double du = 0.0, um = 0.0;
            for (int i = 0; i < n; ++i) {
                du = std::max(du, std::hypot(u1[i] - u0[i], v1[i] - v0[i]));
                um = std::max(um, std::hypot(u0[i], v0[i]));
            }
            printf("[Reproject-test] step %d: max|u_new - u| = %.3e (max|u| = %.3e)\n", step, du, um);
        }

        if (remesh_due) {
            compute_error_indicator(gpu_buf, inflow_umax(step * dt, opt.case_2d3));
            amr.inlet_umax = inflow_umax(step * dt, opt.case_2d3);
            gpu_buf.copy_solution_to_host(h_u, h_v, h_p, h_uo, h_vo, h_vort, h_elem_err);
            amr.scatter_solution(h_u, h_v, h_p, h_uo, h_vo);
            remesh_prof.indicator += lap();
            // Tolerances are velocities: with a time-varying inflow (2D-3) they follow its magnitude,
            // so the mesh does not coarsen wholesale while the flow decelerates (floor 10 %)
            AmrParams prm_now = opt.amr;
            if (opt.case_2d3 && opt.relative_tol) {
                double scale = std::max(0.1, std::fabs(inflow_umax(step * dt, true)) / SimConfig::U_MAX);
                prm_now.tol_vort *= scale;
                prm_now.tol_interp *= scale;
            }
            if (!overlap) {
                AmrStats st = amr.adapt(h_elem_err, prm_now);
                remesh_prof.add(st);
                std::vector<int> old_vids;
                std::vector<double> old_x, old_y;
                if (amg_remap_now) { old_vids = amr.node_vertices();  old_x = mesh.x;  old_y = mesh.y; }
                lap();
                amr.extract(mesh, h_u, h_v, h_p, h_uo, h_vo);
                remesh_prof.extract += lap();
                build_csr_and_boundary_topology(mesh);
                remesh_prof.csr += lap();
                MeshQuality q;
                if (!validate_mesh(mesh, q)) {
                    fprintf(stderr, "[AMR] mesh validation failed, aborting\n");
                    return EXIT_FAILURE;
                }
                remesh_prof.validate += lap();
                finish_remesh(st, q, old_vids, old_x, old_y);
            } else {
                prm_now.lookahead_time += opt.amr_overlap * dt;
                pend.old_vids = amr.node_vertices();
                pend.join_step = step + opt.amr_overlap;
                // The worker owns the forest and pend until the join; the main thread only steps
                pend.job = std::async(std::launch::async, [&amr, &pend, prm_now, err = h_elem_err]() {
                    auto tw = std::chrono::steady_clock::now();
                    pend.st = amr.adapt(err, prm_now);   // phase times in pend.st
                    tw = std::chrono::steady_clock::now();
                    amr.extract(pend.mesh, pend.u, pend.v, pend.p, pend.uo, pend.vo);   // values replaced at the join
                    pend.extract_ms = ms_since(tw);  tw = std::chrono::steady_clock::now();
                    build_csr_and_boundary_topology(pend.mesh);
                    pend.csr_ms = ms_since(tw);  tw = std::chrono::steady_clock::now();
                    bool ok = validate_mesh(pend.mesh, pend.q);
                    pend.validate_ms = ms_since(tw);
                    return ok;
                });
                amr_ms_total += ms_since(t0);
            }
        }
        if (overlap && step == pend.join_step) {
            tp = t0 = std::chrono::steady_clock::now();
            bool ok = pend.job.get();
            remesh_prof.join_wait += lap();
            if (!ok) {
                fprintf(stderr, "[AMR] mesh validation failed, aborting\n");
                return EXIT_FAILURE;
            }
            remesh_prof.add(pend.st);
            remesh_prof.extract += pend.extract_ms;  remesh_prof.csr += pend.csr_ms;  remesh_prof.validate += pend.validate_ms;
            remesh_prof.overlapped += pend.st.ms_lookahead + pend.st.ms_mark + pend.st.ms_neighbours + pend.st.ms_coarsen
                                    + pend.st.ms_refine + pend.extract_ms + pend.csr_ms + pend.validate_ms;
            gpu_buf.copy_solution_to_host(h_u, h_v, h_p, h_uo, h_vo, h_vort, h_elem_err);
            amr.inlet_umax = inflow_umax(step * dt, opt.case_2d3);
            std::vector<double> old_x, old_y;
            if (amg_remap_now) { old_x = mesh.x;  old_y = mesh.y; }
            amr.retransfer(pend.old_vids, h_u, h_v, h_p, h_uo, h_vo, pend.u, pend.v, pend.p, pend.uo, pend.vo);
            mesh = std::move(pend.mesh);
            pend.mesh = Mesh2D();
            h_u.swap(pend.u);  h_v.swap(pend.v);  h_p.swap(pend.p);  h_uo.swap(pend.uo);  h_vo.swap(pend.vo);
            remesh_prof.transfer += lap();
            pend.join_step = -1;
            finish_remesh(pend.st, pend.q, pend.old_vids, old_x, old_y);
        }
    }
    csv.close();

    double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    // 4. Export Final Solution & Adapted Mesh to VTK UnstructuredGrid File
    compute_error_indicator(gpu_buf);
    gpu_buf.copy_solution_to_host(h_u, h_v, h_p, h_uo, h_vo, h_vort, h_elem_err);
    std::string vtk_file = opt.prefix + "_final.vtk";
    write_vtk_unstructured(vtk_file, mesh, h_u, h_v, h_p, h_vort, h_elem_err, opt.amr);

    if (opt.mms) {
        double* d_err = nullptr;
        CUDA_CHECK(cudaMalloc(&d_err, 6 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_err, 0, 6 * sizeof(double)));
        k_mms_errors<<<(gpu_buf.n_elems + 255) / 256, 256>>>(
            gpu_buf.n_elems, gpu_buf.d_elem_v, gpu_buf.d_x, gpu_buf.d_y,
            gpu_buf.d_u, gpu_buf.d_v, gpu_buf.d_p, opt.num_steps * dt, d_err);
        CUDA_CHECK_LAUNCH();
        double e2[6];
        CUDA_CHECK(cudaMemcpy(e2, d_err, 6 * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_err);
        // pointwise pressure error on the cylinder surface (nodes and the two DFG probes)
        std::vector<double> hp(gpu_buf.n_nodes);
        CUDA_CHECK(cudaMemcpy(hp.data(), gpu_buf.d_p, gpu_buf.n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        const double t_end = opt.num_steps * dt;
        double cyl_max = 0.0;
        for (int i = 0; i < mesh.num_nodes; ++i) {
            if (mesh.bc_tag[i] != BC_CYLINDER) continue;
            cyl_max = std::max(cyl_max, std::fabs(hp[i] - mms_exact(mesh.x[i], mesh.y[i], t_end).p));
        }
        auto probe_err = [&](const PressureProbe& pr, double px, double py) {
            return std::fabs((1.0 - pr.w) * hp[pr.a] + pr.w * hp[pr.b] - mms_exact(px, py, t_end).p);
        };
        double probe_max = std::max(probe_err(gpu_buf.probe_front, 0.15, 0.20),
                                    probe_err(gpu_buf.probe_back,  0.25, 0.20));
        double h_max = 0.0;
        for (const auto& tri : mesh.elems) {
            for (int k = 0; k < 3; ++k) {
                int a = tri.v[k], b = tri.v[(k + 1) % 3];
                h_max = std::max(h_max, std::hypot(mesh.x[a] - mesh.x[b], mesh.y[a] - mesh.y[b]));
            }
        }
        std::cout << std::scientific << std::setprecision(6)
                  << "[MMS] t = " << opt.num_steps * dt << " | nodes " << mesh.num_nodes
                  << " | h_max " << h_max << " | dt " << dt << "\n"
                  << "[MMS] ||u-u_h||_L2 = " << std::sqrt(e2[0]) << " (rel " << std::sqrt(e2[0] / e2[3]) << ")"
                  << " | |u-u_h|_H1 = " << std::sqrt(e2[1]) << " (rel " << std::sqrt(e2[1] / e2[4]) << ")"
                  << " | ||p-p_h||_L2 = " << std::sqrt(e2[2]) << " (rel " << std::sqrt(e2[2] / e2[5]) << ")\n"
                  << "[MMS] max |p-p_h| on cylinder nodes = " << cyl_max
                  << " | at the DFG probes = " << probe_max << "\n"
                  << std::fixed;
        std::ofstream f(opt.prefix + "_mms.csv");
        f << "nodes,elements,h_max,dt,t,err_u_L2,err_u_H1,err_p_L2,norm_u_L2,norm_u_H1,norm_p_L2,err_p_cyl_max,err_p_probe_max\n"
          << std::setprecision(10) << mesh.num_nodes << "," << mesh.num_elems << "," << h_max << ","
          << dt << "," << opt.num_steps * dt;
        for (int k = 0; k < 6; ++k) f << "," << std::sqrt(e2[k]);
        f << "," << cyl_max << "," << probe_max << "\n";
    } else if (opt.case_2d3) {
        history.report_2d3(opt.prefix + "_stats.csv");
    } else {
        history.report(opt.stats_window, opt.prefix + "_stats.csv");
    }
    std::cout << std::setprecision(2)
              << "[Timing] " << wall_s << " s wall, " << amr_ms_total / 1000.0 << " s in AMR (incl. transfer)"
              << " | PPE AMG setup " << amg_setup_ms_total / 1000.0 << " s"
              << " | AMG remaps " << amg.remaps
              << " | non-converged linear solves: " << nonconverged_solves << "\n";
    std::cout << "[Timing] step phases: momentum " << momentum_ms_total / 1000.0 << " s"
              << (opt.momentum_ilu ? " (ILU colouring " : "") ;
    if (opt.momentum_ilu) std::cout << mom_ilu.colour_ms / 1000.0 << " s, " << mom_ilu.n_colours << " colours at the end)";
    std::cout << " | pressure " << pressure_ms_total / 1000.0 << " s (assembly, AMG setup, CG)\n";
    remesh_prof.report(amr_ms_total);
    {
        static const char* dn[] = {"adjacency", "traces", "trace merge", "leaf collect", "morton keys",
                                   "morton sort", "vertex compaction", "mesh fill"};
        std::cout << "[Timing] AMR sub-phases:";
        for (int i = 0; i < AdaptiveMesh::D_COUNT; ++i)
            std::cout << " " << dn[i] << " " << std::setprecision(2) << amr.dbg_ms[i] / 1000.0 << " s;";
        std::cout << "\n";
    }
    if (use_amg) {
        static const char* names[] = {"download", "D^-1 + rho", "aggregation", "P smoothing", "transpose",
                                      "Galerkin product", "upload", "coarse inverse", "remap"};
        std::cout << "[Timing] AMG setup phases (all setups):";
        for (int i = 0; i < AmgPreconditioner::PH_COUNT; ++i)
            std::cout << " " << names[i] << " " << std::setprecision(2) << amg.phase_ms[i] / 1000.0 << " s;";
        std::cout << "\n";
    }
    std::cout << "[Output] Wrote ParaView unstructured mesh & flow field to: " << vtk_file << "\n";
    std::cout << "[Output] Wrote force history to: " << opt.prefix << "_forces.csv\n";

    return 0;
}
#endif  // CFA_NO_MAIN
