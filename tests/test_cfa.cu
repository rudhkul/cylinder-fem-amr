/**
 * Unit tests for cylinder_fem_amr: mesh adaptation (NVB conformity, level cap, refine/coarsen
 * round trip, exact transfer of linear fields), the volume-integral force functional and the
 * momentum solver (multicolour ILU(0), two-component BiCGSTAB) and the assembled OSS projection.
 *
 *   nvcc -O2 -arch=native -std=c++17 -Xcompiler -fopenmp tests/test_cfa.cu -o test_cfa && ./test_cfa
 *
 * Exit code 0 if every check passes.
 */
#define CFA_NO_MAIN
#include "../src/cylinder_fem_amr.cu"

#include <random>

static int g_failures = 0;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
        }                                                                   \
    } while (0)

static Mesh2D base_mesh() {
    Mesh2D mesh;
    generate_cylinder_mesh(mesh, 8, 7, 28);
    build_csr_and_boundary_topology(mesh);
    return mesh;
}

// Linear test field (no boundary values are imposed on p, so transfers must keep it exact)
static double linear(double x, double y) { return 0.3 + 1.7 * x - 2.9 * y; }

struct Fixture {
    Mesh2D mesh;
    std::vector<double> u, v, p, uo, vo;
    AdaptiveMesh amr;
    explicit Fixture(const Mesh2D& base)
        : mesh(base), u(base.num_nodes, 0.0), v(base.num_nodes, 0.0), p(make_p(base)),
          uo(base.num_nodes, 0.0), vo(base.num_nodes, 0.0), amr(base, u, v, p) {
        amr.extract(mesh, u, v, p, uo, vo);
    }
    static std::vector<double> make_p(const Mesh2D& m) {
        std::vector<double> r(m.num_nodes);
        for (int i = 0; i < m.num_nodes; ++i) r[i] = linear(m.x[i], m.y[i]);
        return r;
    }
    MeshQuality refresh() {
        amr.extract(mesh, u, v, p, uo, vo);
        build_csr_and_boundary_topology(mesh);
        MeshQuality q;
        bool ok = validate_mesh(mesh, q);
        CHECK(ok, "adapted mesh is not a valid conforming triangulation");
        return q;
    }
};

// Random adaptation cycles: conformity, level cap, angle bound, exact transfer of a linear field
static void test_random_adaptation() {
    const Mesh2D base = base_mesh();
    MeshQuality q0;
    validate_mesh(base, q0);
    Fixture f(base);
    AmrParams prm;
    prm.max_level = 4;
    prm.tol_vort = 1.0;
    prm.tol_interp = 1.0;
    prm.lookahead_time = 0.0;
    prm.keep_radius = 0.0;
    prm.min_level = 0;
    prm.max_elems = 400000;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double total_area0 = q0.total_area;
    for (int cycle = 0; cycle < 12; ++cycle) {
        int L = f.mesh.num_elems;
        std::vector<double> eta(2 * L);
        // smooth random field of "features": refine near a moving blob, coarsen elsewhere
        double bx = 0.2 + 1.8 * uni(rng), by = 0.41 * uni(rng);
        for (int e = 0; e < L; ++e) {
            const Triangle& t = f.mesh.elems[e];
            double xc = (f.mesh.x[t.v[0]] + f.mesh.x[t.v[1]] + f.mesh.x[t.v[2]]) / 3.0;
            double yc = (f.mesh.y[t.v[0]] + f.mesh.y[t.v[1]] + f.mesh.y[t.v[2]]) / 3.0;
            double d2 = (xc - bx) * (xc - bx) + (yc - by) * (yc - by);
            eta[e] = 8.0 * std::exp(-d2 / 0.01) * (0.5 + uni(rng));
            eta[L + e] = 0.05 * uni(rng);
        }
        f.amr.scatter_solution(f.u, f.v, f.p, f.uo, f.vo);
        f.amr.adapt(eta, prm);
        MeshQuality q = f.refresh();
        CHECK(q.max_level <= prm.max_level, "cycle %d: level %d above cap %d", cycle, q.max_level, prm.max_level);
        CHECK(q.min_angle_deg >= q0.min_angle_deg - 1.0,
              "cycle %d: min angle %.2f fell well below the base %.2f", cycle, q.min_angle_deg, q0.min_angle_deg);
        CHECK(std::fabs(q.total_area - total_area0) < 1.0e-3 * total_area0,
              "cycle %d: area %.9f vs base %.9f (only cylinder arcs may differ)", cycle, q.total_area, total_area0);
        double max_err = 0.0;
        for (int i = 0; i < f.mesh.num_nodes; ++i) {
            // Cylinder midpoints are moved onto the arc but keep the chord-midpoint value, so
            // nodes interpolated from them (within a base element of the cylinder) are exempt
            double r = std::hypot(f.mesh.x[i] - SimConfig::X_CYL, f.mesh.y[i] - SimConfig::Y_CYL);
            if (r < SimConfig::R_CYL + 0.03) continue;
            max_err = std::max(max_err, std::fabs(f.p[i] - linear(f.mesh.x[i], f.mesh.y[i])));
        }
        CHECK(max_err < 1.0e-12, "cycle %d: linear field not transferred exactly (%.3e)", cycle, max_err);
    }
    printf("  random adaptation: %d elements after 12 cycles\n", f.mesh.num_elems);
}

// Uniform refinement then coarsening with zero indicators returns the base mesh
static void test_round_trip() {
    const Mesh2D base = base_mesh();
    Fixture f(base);
    AmrStats st = f.amr.refine_uniform(2);
    MeshQuality q = f.refresh();
    CHECK(f.mesh.num_elems == 16 * base.num_elems, "uniform level 2: %d elements, expected %d",
          f.mesh.num_elems, 16 * base.num_elems);
    CHECK(st.closure_bisections == 0, "uniform refinement needed %d closure bisections (matching labels)",
          st.closure_bisections);
    CHECK(q.max_level == 2, "uniform level 2 has max level %d", q.max_level);
    AmrParams prm;
    prm.max_level = 4;
    prm.min_level = 0;
    prm.lookahead_time = 0.0;
    prm.keep_radius = 0.0;
    for (int cycle = 0; cycle < 6; ++cycle) {
        std::vector<double> eta(2 * f.mesh.num_elems, 0.0);
        f.amr.scatter_solution(f.u, f.v, f.p, f.uo, f.vo);
        f.amr.adapt(eta, prm);
        f.refresh();
    }
    CHECK(f.mesh.num_elems == base.num_elems && f.mesh.num_nodes == base.num_nodes,
          "after coarsening: %d elements / %d nodes, base %d / %d",
          f.mesh.num_elems, f.mesh.num_nodes, base.num_elems, base.num_nodes);
    // same vertex set as the base mesh
    std::vector<std::pair<double, double>> a, b;
    for (int i = 0; i < base.num_nodes; ++i) a.push_back({base.x[i], base.y[i]});
    for (int i = 0; i < f.mesh.num_nodes; ++i) b.push_back({f.mesh.x[i], f.mesh.y[i]});
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    CHECK(a == b, "coarsened mesh vertices differ from the base mesh");
}

// Coarsening stops at the minimum level
static void test_min_level() {
    const Mesh2D base = base_mesh();
    Fixture f(base);
    f.amr.refine_uniform(3);
    f.refresh();
    AmrParams prm;
    prm.max_level = 4;
    prm.min_level = 2;
    prm.lookahead_time = 0.0;
    prm.keep_radius = 0.0;
    for (int cycle = 0; cycle < 6; ++cycle) {
        std::vector<double> eta(2 * f.mesh.num_elems, 0.0);
        f.amr.scatter_solution(f.u, f.v, f.p, f.uo, f.vo);
        f.amr.adapt(eta, prm);
        f.refresh();
    }
    int below = 0;
    for (const auto& t : f.mesh.elems) below += (t.level < prm.min_level) ? 1 : 0;
    CHECK(below == 0 && f.mesh.num_elems == 16 * base.num_elems,
          "min level 2: %d elements below the floor, %d elements (expected %d)",
          below, f.mesh.num_elems, 16 * base.num_elems);
}

// Volume-integral force: zero for a fluid at rest with constant pressure
static void test_force_at_rest() {
    Mesh2D mesh = base_mesh();
    std::vector<double> z(mesh.num_nodes, 0.0), p(mesh.num_nodes, 7.25);
    DeviceMeshBuffers g;
    g.sync_from_host(mesh, z, z, p, z, z);
    CUDA_CHECK(cudaMemset(g.d_diag, 0, 4 * sizeof(double)));
    k_volume_forces<<<(g.n_elems + 255) / 256, 256>>>(
        g.n_elems, g.d_elem_v, g.d_bc_tag, g.d_x, g.d_y, g.d_u, g.d_v, g.d_u_old, g.d_v_old,
        g.d_u_prev, g.d_v_prev, g.d_p, 1500.0, -2000.0, 500.0, SimConfig::NU, SimConfig::RHO, g.d_diag);
    CUDA_CHECK_LAUNCH();
    double d[4];
    CUDA_CHECK(cudaMemcpy(d, g.d_diag, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(std::fabs(d[0]) < 1.0e-12 && std::fabs(d[1]) < 1.0e-12,
          "force on a cylinder in fluid at rest: (%.3e, %.3e)", d[0], d[1]);
}

// Manufactured solution satisfies the code's boundary conditions and is divergence-free
static void test_mms_solution() {
    double max_div = 0.0, max_out = 0.0;
    for (int k = 0; k <= 40; ++k) {
        double y = SimConfig::H_CHANNEL * k / 40.0, t = 0.013 * k;
        MmsValues r = mms_exact(SimConfig::L_CHANNEL, y, t);
        max_out = std::max({max_out, std::fabs(r.ux), std::fabs(r.vx), std::fabs(r.p)});
        for (int j = 0; j <= 40; ++j) {
            MmsValues s = mms_exact(SimConfig::L_CHANNEL * j / 40.0, y, t);
            max_div = std::max(max_div, std::fabs(s.ux + s.vy));
        }
    }
    CHECK(max_div < 1.0e-12, "manufactured velocity not divergence-free (%.3e)", max_div);
    CHECK(max_out < 1.0e-12, "manufactured solution violates the outflow conditions (%.3e)", max_out);
}

// Multicolour ILU(0): valid colouring, the ILU(0) identity (LU)_ij = a_ij on the pattern, and the
// two-component BiCGSTAB against the Jacobi solver on a non-symmetric, diagonally dominant matrix
static void test_momentum_ilu() {
    Fixture f(base_mesh());
    f.amr.refine_uniform(2);
    f.refresh();
    const Mesh2D& m = f.mesh;
    const int n = m.num_nodes, nnz = m.nnz;
    const std::vector<int>& rp = m.csr_row_ptr;
    const std::vector<int>& ci = m.csr_col_ind;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> a(nnz), b(2 * n), x0(2 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        double off = 0.0;
        int d = -1;
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            if (ci[k] == i) { d = k; continue; }
            a[k] = -(0.2 + U(rng)) * (ci[k] > i ? 1.6 : 0.6);   // convection-like asymmetry
            off += std::fabs(a[k]);
        }
        a[d] = 1.05 * off + 0.1;
        b[i] = U(rng) - 0.5;  b[n + i] = std::sin(0.01 * i);
    }
    int *d_rp, *d_ci;
    double *d_a, *d_b, *d_x, *d_xj;
    CUDA_CHECK(cudaMalloc(&d_rp, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, 2 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, 2 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_xj, 2 * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_rp, rp.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ci, ci.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), 2 * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x0.data(), 2 * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xj, x0.data(), 2 * n * sizeof(double), cudaMemcpyHostToDevice));

    MulticolourIlu0 ilu;
    ilu.set_pattern(n, rp, ci, d_rp, d_ci);
    ilu.factor(d_a);
    std::vector<int> col(n);
    std::vector<double> lu(nnz);
    CUDA_CHECK(cudaMemcpy(col.data(), ilu.d_colour, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lu.data(), ilu.d_lu, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    int bad_colour = 0;
    for (int i = 0; i < n; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k) bad_colour += (ci[k] != i && col[ci[k]] == col[i]);
    CHECK(bad_colour == 0, "colouring: %d neighbouring pairs share a colour", bad_colour);
    // (LU)_ij = sum over k in both patterns, k before or at min(i, j) in the colour order, of l_ik u_kj
    // (l_ii = 1); the order is (colour, index), neighbours never share a colour
    auto at = [&](int r, int c) -> int {
        for (int k = rp[r]; k < rp[r + 1]; ++k) if (ci[k] == c) return k;
        return -1;
    };
    auto before = [&](int p, int q) { return col[p] < col[q] || (col[p] == col[q] && p < q); };
    double ilu_err = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int e = rp[i]; e < rp[i + 1]; ++e) {
            const int j = ci[e];
            double s = 0.0;
            for (int q = rp[i]; q < rp[i + 1]; ++q) {
                const int k = ci[q];
                if (!(k == i || before(k, i))) continue;      // l_ik with k <= i
                if (!(k == j || before(k, j))) continue;      // u_kj with k <= j
                const int kj = at(k, j);
                if (kj < 0) continue;
                const double l = (k == i) ? 1.0 : lu[q];
                s += l * lu[kj];
            }
            ilu_err = std::max(ilu_err, std::fabs(s - a[e]) / std::fabs(a[at(i, i)]));
        }
    }
    CHECK(ilu_err < 1.0e-12, "ILU(0) identity violated: max |(LU)_ij - a_ij| / a_ii = %.3e", ilu_err);

    GpuBicgstab2 two;
    two.solve(n, d_rp, d_ci, d_a, d_b, d_b + n, d_x, d_x + n, 200, 1.0e-10, ilu);
    GpuLinearSolver one;
    one.solve_bicgstab(n, d_rp, d_ci, d_a, d_b, d_xj, 400, 1.0e-10);
    one.solve_bicgstab(n, d_rp, d_ci, d_a, d_b + n, d_xj + n, 400, 1.0e-10);
    std::vector<double> x(2 * n), xj(2 * n);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, 2 * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xj.data(), d_xj, 2 * n * sizeof(double), cudaMemcpyDeviceToHost));
    double diff = 0.0, scale = 0.0;
    for (int i = 0; i < 2 * n; ++i) { diff = std::max(diff, std::fabs(x[i] - xj[i])); scale = std::max(scale, std::fabs(xj[i])); }
    CHECK(two.converged[0] && two.converged[1], "two-component BiCGSTAB did not converge");
    CHECK(diff < 1.0e-8 * scale, "ILU and Jacobi solutions differ: %.3e (scale %.3e)", diff, scale);
    printf("  momentum ILU: %d nodes, %d colours, iterations ILU (%d, %d)\n", n, ilu.n_colours, two.iters[0], two.iters[1]);
    for (void* p : {(void*)d_rp, (void*)d_ci, (void*)d_a, (void*)d_b, (void*)d_x, (void*)d_xj}) cudaFree(p);
}

// Assembled OSS projection -B^T M_L^{-1} B equals the matrix-free element form
static void test_oss_assembled() {
    Fixture f(base_mesh());
    f.amr.refine_uniform(2);
    f.refresh();
    const Mesh2D& m = f.mesh;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> z(m.num_nodes, 0.0), xin(m.num_nodes), s(m.num_elems);
    for (double& v : xin) v = U(rng) - 0.5;
    for (double& v : s) v = 0.1 + U(rng);
    DeviceMeshBuffers g;
    g.sync_from_host(m, z, z, z, z, z);
    const int n = g.n_nodes, tb = 256;
    CUDA_CHECK(cudaMemcpy(g.d_elem_s, s.data(), m.num_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.d_mass_lumped, 0, n * sizeof(double)));
    k_assemble_lumped_mass<<<(g.n_elems + tb - 1) / tb, tb>>>(g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_mass_lumped);
    double *d_in, *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_in, xin.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_a, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_b, 0, n * sizeof(double)));
    apply_oss_projection_matrix_free(g, d_in, d_a);
    assemble_oss_projection(g);
    apply_oss_projection(g, d_in, d_b);
    std::vector<double> a(n), b(n);
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_b, n * sizeof(double), cudaMemcpyDeviceToHost));
    double diff = 0.0, scale = 0.0;
    for (int i = 0; i < n; ++i) { diff = std::max(diff, std::fabs(a[i] - b[i])); scale = std::max(scale, std::fabs(a[i])); }
    CHECK(scale > 0.0 && diff < 1.0e-12 * scale, "assembled OSS projection differs: %.3e (scale %.3e)", diff, scale);
    for (void* p : {(void*)d_in, (void*)d_a, (void*)d_b}) cudaFree(p);
}

// AmgPreconditioner::remap: on an unchanged mesh it reproduces setup() exactly; after a local
// refinement the carried-over hierarchy is still a usable SPD preconditioner for PCG
static void test_amg_remap() {
    // shifted graph Laplacian on the mesh graph (SPD, nearly singular, so the coarse levels matter)
    auto upload_matrix = [](const Mesh2D& m, int*& d_rp, int*& d_ci, double*& d_a) {
        const int n = m.num_nodes, nnz = m.nnz;
        std::vector<double> a(nnz);
        for (int i = 0; i < n; ++i) {
            int d = -1, deg = 0;
            for (int k = m.csr_row_ptr[i]; k < m.csr_row_ptr[i + 1]; ++k) {
                if (m.csr_col_ind[k] == i) { d = k; continue; }
                a[k] = -1.0;  ++deg;
            }
            a[d] = deg * (1.0 + 1.0e-4);
        }
        CUDA_CHECK(cudaMalloc(&d_rp, (n + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ci, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_a, nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_rp, m.csr_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ci, m.csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a, a.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    };
    auto vcycle_of = [](AmgPreconditioner& amg, int n) {
        std::vector<double> r(n), z(n);
        for (int i = 0; i < n; ++i) r[i] = std::sin(0.37 * i) + 0.2;
        double *d_r, *d_z;
        CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_r, r.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        amg.apply(d_r, d_z);
        CUDA_CHECK(cudaMemcpy(z.data(), d_z, n * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_r);  cudaFree(d_z);
        return z;
    };
    Fixture f(base_mesh());
    f.amr.refine_uniform(3);
    f.refresh();
    int *d_rp, *d_ci;
    double* d_a;
    upload_matrix(f.mesh, d_rp, d_ci, d_a);
    const int n0 = f.mesh.num_nodes;
    AmgPreconditioner amg;
    amg.setup(n0, f.mesh.nnz, d_rp, d_ci, d_a);
    CHECK(amg.num_levels() > 2, "test hierarchy too shallow (%d levels)", amg.num_levels());
    std::vector<double> z_setup = vcycle_of(amg, n0);
    std::vector<int> ident(n0);
    for (int i = 0; i < n0; ++i) ident[i] = i;
    amg.remap(n0, f.mesh.nnz, d_rp, d_ci, d_a, f.mesh.csr_row_ptr.data(), f.mesh.csr_col_ind.data(), ident);
    CHECK(amg.remaps == 1, "remap fell back to setup on an unchanged mesh");
    std::vector<double> z_remap = vcycle_of(amg, n0);
    double dz = 0.0, sz = 0.0;
    for (int i = 0; i < n0; ++i) { dz = std::max(dz, std::fabs(z_remap[i] - z_setup[i])); sz = std::max(sz, std::fabs(z_setup[i])); }
    CHECK(dz <= 1.0e-14 * sz, "remap on an unchanged mesh changes the V-cycle: %.3e (scale %.3e)", dz, sz);
    cudaFree(d_rp);  cudaFree(d_ci);  cudaFree(d_a);

    // local refinement around a point, as one remesh does
    std::vector<int> old_vids = f.amr.node_vertices();
    std::vector<double> old_x = f.mesh.x, old_y = f.mesh.y;
    AmrParams prm;
    prm.max_level = 5;  prm.tol_vort = 1.0;  prm.tol_interp = 1.0;  prm.lookahead_time = 0.0;
    prm.keep_radius = 0.0;  prm.min_level = 0;  prm.max_elems = 400000;
    const int L = f.mesh.num_elems;
    std::vector<double> eta(2 * L, 0.5);   // keep everything except a blob behind the cylinder
    for (int e = 0; e < L; ++e) {
        const Triangle& t = f.mesh.elems[e];
        double xc = (f.mesh.x[t.v[0]] + f.mesh.x[t.v[1]] + f.mesh.x[t.v[2]]) / 3.0;
        double yc = (f.mesh.y[t.v[0]] + f.mesh.y[t.v[1]] + f.mesh.y[t.v[2]]) / 3.0;
        double d2 = (xc - 0.4) * (xc - 0.4) + (yc - 0.2) * (yc - 0.2);
        eta[e] = d2 < 0.0004 ? 8.0 : 0.5;
    }
    f.amr.scatter_solution(f.u, f.v, f.p, f.uo, f.vo);
    f.amr.adapt(eta, prm);
    f.refresh();
    const Mesh2D& m = f.mesh;
    const int n = m.num_nodes;
    std::vector<int> slot_to_old(f.amr.num_vertex_slots(), -1), new_to_old(n);
    for (int j = 0; j < (int)old_vids.size(); ++j) slot_to_old[old_vids[j]] = j;
    int kept = 0;
    for (int i = 0; i < n; ++i) {
        int j = slot_to_old[f.amr.node_vertices()[i]];
        new_to_old[i] = (j >= 0 && old_x[j] == m.x[i] && old_y[j] == m.y[i]) ? j : -1;
        kept += new_to_old[i] >= 0;
    }
    CHECK(n > n0 && kept < n, "refinement did not add nodes (%d -> %d, kept %d)", n0, n, kept);
    upload_matrix(m, d_rp, d_ci, d_a);
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = std::cos(0.11 * i);
    double *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    std::function<void(const double*, double*)> prec = [&amg](const double* r, double* z) { amg.apply(r, z); };
    GpuLinearSolver solver;
    amg.remap(n, m.nnz, d_rp, d_ci, d_a, m.csr_row_ptr.data(), m.csr_col_ind.data(), new_to_old);
    CHECK(amg.remaps == 2, "remap fell back to setup after a local refinement");
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
    int it_remap = solver.solve_pcg(n, d_rp, d_ci, d_a, d_b, d_x, 400, 1.0e-10, nullptr, &prec);
    bool conv_remap = solver.converged;
    amg.setup(n, m.nnz, d_rp, d_ci, d_a);
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
    int it_setup = solver.solve_pcg(n, d_rp, d_ci, d_a, d_b, d_x, 400, 1.0e-10, nullptr, &prec);
    CHECK(conv_remap && solver.converged, "PCG did not converge (remap %d, setup %d)", (int)conv_remap, (int)solver.converged);
    // ~3% new nodes in one step (a real remesh changes ~1%): 36 against 14 iterations when written
    CHECK(it_remap <= 3 * it_setup + 5, "remapped hierarchy is a poor preconditioner: %d vs %d iterations", it_remap, it_setup);
    printf("  AMG remap: %d -> %d nodes (%d new), PCG iterations remap %d, fresh setup %d\n",
           n0, n, n - kept, it_remap, it_setup);
    for (void* p : {(void*)d_rp, (void*)d_ci, (void*)d_a, (void*)d_b, (void*)d_x}) cudaFree(p);
}

// AdaptiveMesh::retransfer (--amr-overlap): applied to the state the adapt() started from it
// reproduces adapt()'s own transfer bit for bit; applied to a later (here: different linear)
// pressure field it transfers that one exactly
static void test_retransfer() {
    Fixture f(base_mesh());
    f.amr.refine_uniform(2);
    f.refresh();
    AmrParams prm;
    prm.max_level = 4;  prm.tol_vort = 1.0;  prm.tol_interp = 1.0;  prm.lookahead_time = 0.0;
    prm.keep_radius = 0.0;  prm.min_level = 0;  prm.max_elems = 400000;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto later_p = [](double x, double y) { return -1.1 + 0.4 * x + 3.3 * y; };
    double max_same = 0.0, max_lin = 0.0;
    for (int cycle = 0; cycle < 4; ++cycle) {
        const Mesh2D& m = f.mesh;
        for (int i = 0; i < m.num_nodes; ++i) {   // smooth non-linear state
            f.u[i] = std::sin(3.0 * m.x[i]) * m.y[i];  f.v[i] = std::cos(2.0 * m.y[i]);
            f.uo[i] = 0.9 * f.u[i];  f.vo[i] = 0.9 * f.v[i];  f.p[i] = m.x[i] * m.x[i] - m.y[i];
        }
        std::vector<double> su = f.u, sv = f.v, sp = f.p, suo = f.uo, svo = f.vo;
        std::vector<double> lp(m.num_nodes);
        for (int i = 0; i < m.num_nodes; ++i) lp[i] = later_p(m.x[i], m.y[i]);
        std::vector<int> old_vids = f.amr.node_vertices();
        const int L = m.num_elems;
        std::vector<double> eta(2 * L);
        double bx = 0.3 + 1.5 * uni(rng), by = 0.41 * uni(rng);
        for (int e = 0; e < L; ++e) {
            const Triangle& t = m.elems[e];
            double xc = (m.x[t.v[0]] + m.x[t.v[1]] + m.x[t.v[2]]) / 3.0;
            double yc = (m.y[t.v[0]] + m.y[t.v[1]] + m.y[t.v[2]]) / 3.0;
            double d2 = (xc - bx) * (xc - bx) + (yc - by) * (yc - by);
            eta[e] = 8.0 * std::exp(-d2 / 0.01);
            eta[L + e] = 0.05 * uni(rng);
        }
        f.amr.scatter_solution(f.u, f.v, f.p, f.uo, f.vo);
        f.amr.adapt(eta, prm);
        f.refresh();   // extract: adapt()'s transfer of the state above
        std::vector<double> ru, rv, rp, ruo, rvo;
        f.amr.retransfer(old_vids, su, sv, sp, suo, svo, ru, rv, rp, ruo, rvo);
        for (int i = 0; i < f.mesh.num_nodes; ++i)
            max_same = std::max({max_same, std::fabs(ru[i] - f.u[i]), std::fabs(rv[i] - f.v[i]), std::fabs(rp[i] - f.p[i]),
                                 std::fabs(ruo[i] - f.uo[i]), std::fabs(rvo[i] - f.vo[i])});
        f.amr.retransfer(old_vids, su, sv, lp, suo, svo, ru, rv, rp, ruo, rvo);
        for (int i = 0; i < f.mesh.num_nodes; ++i) {
            double r = std::hypot(f.mesh.x[i] - SimConfig::X_CYL, f.mesh.y[i] - SimConfig::Y_CYL);
            if (r < SimConfig::R_CYL + 0.03) continue;   // arc midpoints, as in test_random_adaptation
            max_lin = std::max(max_lin, std::fabs(rp[i] - later_p(f.mesh.x[i], f.mesh.y[i])));
        }
    }
    CHECK(max_same == 0.0, "retransfer of the same state differs from adapt()'s transfer: %.3e", max_same);
    CHECK(max_lin < 1.0e-12, "retransfer of a later linear field not exact: %.3e", max_lin);
    printf("  retransfer: 4 cycles, %d elements at the end\n", f.mesh.num_elems);
}

int main() {
    printf("cylinder_fem_amr unit tests\n");
    test_random_adaptation();
    test_round_trip();
    test_min_level();
    test_force_at_rest();
    test_mms_solution();
    test_momentum_ilu();
    test_oss_assembled();
    test_amg_remap();
    test_retransfer();
    if (g_failures == 0) printf("ALL PASSED\n");
    else printf("%d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
