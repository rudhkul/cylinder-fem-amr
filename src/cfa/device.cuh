// Device buffers and GPU driver routines shared by time stepping and remeshing
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// Device Buffer Manager for Dynamic AMR Remeshing (capacity-based, grows geometrically)
// ============================================================================
template <typename T>
static void device_realloc(T*& ptr, size_t count) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
}

template <typename T>
static void device_free(T*& ptr) {
    if (ptr) cudaFree(ptr);
    ptr = nullptr;
}

struct DeviceMeshBuffers {
    int n_nodes = 0;
    int n_elems = 0;
    int nnz     = 0;

    size_t cap_nodes = 0;
    size_t cap_elems = 0;
    size_t cap_nnz   = 0;

    int *d_elem_v    = nullptr;
    int *d_cyl_edges = nullptr;   // [3 * n_cyl_edges]: (a, b, owning element)
    int  n_cyl_edges = 0;
    size_t cap_cyl_edges = 0;
    int *d_bc_tag    = nullptr;
    int *d_row_ptr   = nullptr;
    int *d_col_ind   = nullptr;

    double *d_x           = nullptr;
    double *d_y           = nullptr;
    double *d_u           = nullptr;
    double *d_v           = nullptr;
    double *d_u_old       = nullptr;
    double *d_v_old       = nullptr;
    double *d_u_prev      = nullptr; // u^{n-1} for BDF2
    double *d_v_prev      = nullptr;
    double *d_p           = nullptr;
    double *d_dp          = nullptr;
    double *d_vort        = nullptr;
    double *d_A_val       = nullptr;
    double *d_Kp_val      = nullptr;
    double *d_rhs_u       = nullptr;
    double *d_rhs_v       = nullptr;
    double *d_rhs_p       = nullptr;
    double *d_mass_lumped = nullptr;
    double *d_grad_dp_x   = nullptr;
    double *d_grad_dp_y   = nullptr;
    double *d_vort_num    = nullptr;
    double *d_elem_error  = nullptr; // [vorticity indicator | interpolation indicator], 2 * n_elems
    double *d_grad_rec    = nullptr; // Recovered velocity gradient [4 * n_nodes]
    double *d_div_num     = nullptr; // (N_i, div u*) for the rotational pressure update
    double *d_elem_s      = nullptr; // OSS weights s_K [n_elems]
    double *d_oss_gx      = nullptr; // scratch G(x) for the matrix-free OSS operator
    double *d_oss_gy      = nullptr;
    double *d_diag        = nullptr; // [F_x, F_y volume brackets, p_front, p_back, F_x, F_y surface]
    double *d_oss_val     = nullptr; // assembled OSS projection [Bx | By | BTx | BTy], 4 * nnz
    double *d_inv_m       = nullptr; // 1 / lumped mass for the assembled OSS projection
    double *d_phi         = nullptr; // --reproject-shift: potential of the last u^n re-projection

    PressureProbe probe_front, probe_back;

    void reserve(size_t nn, size_t ne, size_t nz) {
        if (nn > cap_nodes) {
            cap_nodes = nn + nn / 2 + 64;
            device_realloc(d_bc_tag,      cap_nodes);
            device_realloc(d_row_ptr,     cap_nodes + 1);
            device_realloc(d_x,           cap_nodes);
            device_realloc(d_y,           cap_nodes);
            device_realloc(d_u,           cap_nodes);
            device_realloc(d_v,           cap_nodes);
            device_realloc(d_u_old,       cap_nodes);
            device_realloc(d_v_old,       cap_nodes);
            device_realloc(d_u_prev,      cap_nodes);
            device_realloc(d_v_prev,      cap_nodes);
            device_realloc(d_p,           cap_nodes);
            device_realloc(d_dp,          cap_nodes);
            device_realloc(d_vort,        cap_nodes);
            device_realloc(d_rhs_u,       cap_nodes);
            device_realloc(d_rhs_v,       cap_nodes);
            device_realloc(d_rhs_p,       cap_nodes);
            device_realloc(d_mass_lumped, cap_nodes);
            device_realloc(d_grad_dp_x,   cap_nodes);
            device_realloc(d_grad_dp_y,   cap_nodes);
            device_realloc(d_vort_num,    cap_nodes);
            device_realloc(d_grad_rec,    4 * cap_nodes);
            device_realloc(d_div_num,     cap_nodes);
            device_realloc(d_oss_gx,      cap_nodes);
            device_realloc(d_oss_gy,      cap_nodes);
            device_realloc(d_inv_m,       cap_nodes);
            device_realloc(d_phi,         cap_nodes);
        }
        if (ne > cap_elems) {
            cap_elems = ne + ne / 2 + 64;
            device_realloc(d_elem_v,     3 * cap_elems);
            device_realloc(d_elem_error, 2 * cap_elems);
            device_realloc(d_elem_s,     cap_elems);
        }
        if (nz > cap_nnz) {
            cap_nnz = nz + nz / 2 + 256;
            device_realloc(d_col_ind, cap_nnz);
            device_realloc(d_A_val,   cap_nnz);
            device_realloc(d_Kp_val,  cap_nnz);
            device_realloc(d_oss_val, 4 * cap_nnz);
        }
        if (!d_diag) device_realloc(d_diag, 6);
    }

    void free_all() {
        device_free(d_cyl_edges);  cap_cyl_edges = 0;
        device_free(d_elem_v);  device_free(d_bc_tag);  device_free(d_row_ptr);  device_free(d_col_ind);
        device_free(d_x);       device_free(d_y);       device_free(d_u);        device_free(d_v);
        device_free(d_u_old);   device_free(d_v_old);   device_free(d_u_prev); device_free(d_v_prev);   device_free(d_p);        device_free(d_dp);
        device_free(d_vort);    device_free(d_A_val);   device_free(d_Kp_val);   device_free(d_rhs_u);
        device_free(d_rhs_v);   device_free(d_rhs_p);   device_free(d_mass_lumped);
        device_free(d_grad_dp_x); device_free(d_grad_dp_y); device_free(d_vort_num);
        device_free(d_elem_error); device_free(d_grad_rec); device_free(d_div_num); device_free(d_diag);
        device_free(d_elem_s);  device_free(d_oss_gx);  device_free(d_oss_gy);
        device_free(d_oss_val); device_free(d_inv_m); device_free(d_phi);
        cap_nodes = cap_elems = cap_nnz = 0;
    }

    void sync_from_host(
        const Mesh2D& mesh,
        const std::vector<double>& h_u,
        const std::vector<double>& h_v,
        const std::vector<double>& h_p,
        const std::vector<double>& h_uo,
        const std::vector<double>& h_vo
    ) {
        n_nodes = mesh.num_nodes;
        n_elems = mesh.num_elems;
        nnz     = mesh.nnz;
        reserve(n_nodes, n_elems, nnz);

        std::vector<int> flat_elems(3 * n_elems);
        for (int e = 0; e < n_elems; ++e) {
            flat_elems[3 * e + 0] = mesh.elems[e].v[0];
            flat_elems[3 * e + 1] = mesh.elems[e].v[1];
            flat_elems[3 * e + 2] = mesh.elems[e].v[2];
        }

        CUDA_CHECK(cudaMemcpy(d_elem_v,  flat_elems.data(),         3 * n_elems * sizeof(int),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bc_tag,  mesh.bc_tag.data(),        n_nodes * sizeof(int),       cudaMemcpyHostToDevice));
        if (g_outlet_dirichlet) {   // pin the pressure increment at the outlet node nearest mid-height
            int pin = -1;
            for (int i = 0; i < n_nodes; ++i)
                if (mesh.bc_tag[i] == BC_OUTLET
                    && (pin < 0 || std::fabs(mesh.y[i] - 0.5 * SimConfig::H_CHANNEL)
                                   < std::fabs(mesh.y[pin] - 0.5 * SimConfig::H_CHANNEL))) pin = i;
            if (pin < 0) { fprintf(stderr, "--outlet dirichlet: no outlet node\n"); exit(EXIT_FAILURE); }
            CUDA_CHECK(cudaMemcpyToSymbol(c_pressure_pin, &pin, sizeof(int)));
            // trapezoidal flux of the unit parabola through the inlet and outlet nodes (corners: u = 0)
            auto flux = [&](int tag) {
                std::vector<double> ys = {0.0, SimConfig::H_CHANNEL};
                for (int i = 0; i < n_nodes; ++i) if (mesh.bc_tag[i] == tag) ys.push_back(mesh.y[i]);
                std::sort(ys.begin(), ys.end());
                auto prof = [](double y) { return y * (SimConfig::H_CHANNEL - y); };
                double q = 0.0;
                for (size_t k = 1; k < ys.size(); ++k) q += 0.5 * (ys[k] - ys[k - 1]) * (prof(ys[k]) + prof(ys[k - 1]));
                return q;
            };
            double scale = flux(BC_INLET) / flux(BC_OUTLET);
            CUDA_CHECK(cudaMemcpyToSymbol(c_outlet_scale, &scale, sizeof(double)));
        }
        CUDA_CHECK(cudaMemcpy(d_row_ptr, mesh.csr_row_ptr.data(),   (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_ind, mesh.csr_col_ind.data(),   nnz * sizeof(int),           cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_x, mesh.x.data(), n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, mesh.y.data(), n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        if (g_outlet_dirichlet) {
            // Transferred outlet values (injection on coarsening) need not carry the inlet flux; scale
            // them so that both time levels enter the re-projection with a compatible flux
            auto balanced = [&](const std::vector<double>& f) {
                std::vector<double> g = f;
                auto flux = [&](int tag) {
                    std::vector<std::pair<double, double>> pts = {{0.0, 0.0}, {SimConfig::H_CHANNEL, 0.0}};
                    for (int i = 0; i < n_nodes; ++i) if (mesh.bc_tag[i] == tag) pts.push_back({mesh.y[i], f[i]});
                    std::sort(pts.begin(), pts.end());
                    double q = 0.0;
                    for (size_t k = 1; k < pts.size(); ++k)
                        q += 0.5 * (pts[k].first - pts[k - 1].first) * (pts[k].second + pts[k - 1].second);
                    return q;
                };
                double qi = flux(BC_INLET), qo = flux(BC_OUTLET);
                if (std::fabs(qo) > 1.0e-14)
                    for (int i = 0; i < n_nodes; ++i) if (mesh.bc_tag[i] == BC_OUTLET) g[i] *= qi / qo;
                return g;
            };
            std::vector<double> u_b = balanced(h_u), uo_b = balanced(h_uo);
            CUDA_CHECK(cudaMemcpy(d_u, u_b.data(),      n_nodes * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_u_old, uo_b.data(), n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(d_u, h_u.data(),      n_nodes * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_u_old, h_uo.data(), n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(),    n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p, h_p.data(),    n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_old, h_vo.data(), n_nodes * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u_prev, d_u_old, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_prev, d_v_old, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_dp,         0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_vort,       0, n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_elem_error, 0, 2 * n_elems * sizeof(double)));

        n_cyl_edges = (int)mesh.cylinder_edges.size();
        if ((size_t)n_cyl_edges > cap_cyl_edges) {
            cap_cyl_edges = 2 * n_cyl_edges + 16;
            device_realloc(d_cyl_edges, 3 * cap_cyl_edges);
        }
        std::vector<int> ce(3 * n_cyl_edges);
        for (int k = 0; k < n_cyl_edges; ++k) {
            ce[3 * k + 0] = mesh.cylinder_edges[k].first;
            ce[3 * k + 1] = mesh.cylinder_edges[k].second;
            ce[3 * k + 2] = mesh.cylinder_edge_elem[k];
        }
        if (n_cyl_edges) CUDA_CHECK(cudaMemcpy(d_cyl_edges, ce.data(), ce.size() * sizeof(int), cudaMemcpyHostToDevice));
        probe_front = find_cylinder_probe(mesh, SimConfig::PI); // (0.15, 0.20)
        probe_back  = find_cylinder_probe(mesh, 0.0);           // (0.25, 0.20)
    }

    void copy_solution_to_host(
        std::vector<double>& h_u,
        std::vector<double>& h_v,
        std::vector<double>& h_p,
        std::vector<double>& h_uo,
        std::vector<double>& h_vo,
        std::vector<double>& h_vort,
        std::vector<double>& h_elem_err
    ) {
        h_u.resize(n_nodes);
        h_v.resize(n_nodes);
        h_p.resize(n_nodes);
        h_uo.resize(n_nodes);
        h_vo.resize(n_nodes);
        h_vort.resize(n_nodes);
        h_elem_err.resize(2 * n_elems);

        CUDA_CHECK(cudaMemcpy(h_u.data(),        d_u,          n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_v.data(),        d_v,          n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_p.data(),        d_p,          n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_uo.data(),       d_u_old,      n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vo.data(),       d_v_old,      n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vort.data(),     d_vort,       n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_elem_err.data(), d_elem_error, 2 * n_elems * sizeof(double), cudaMemcpyDeviceToHost));
    }

    ~DeviceMeshBuffers() { free_all(); }
};

// ============================================================================
// GPU Driver Routines Shared by Time Stepping and Remeshing
// ============================================================================
/**
 * Lumped mass + gradient recovery + both element refinement indicators on the current mesh.
 */
void compute_error_indicator(DeviceMeshBuffers& g, double inflow_peak = SimConfig::U_MAX) {
    int threads = 256;
    int blocks_elems = (g.n_elems + threads - 1) / threads;

    CUDA_CHECK(cudaMemset(g.d_mass_lumped, 0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_grad_rec,    0, 4 * g.n_nodes * sizeof(double)));

    k_assemble_lumped_mass<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_mass_lumped);
    k_accumulate_recovered_gradient<<<blocks_elems, threads>>>(
        g.n_elems, g.n_nodes, g.d_elem_v, g.d_x, g.d_y, g.d_u, g.d_v,
        inflow_peak, SimConfig::H_CHANNEL, g.d_grad_rec);
    k_compute_amr_error_indicator<<<blocks_elems, threads>>>(
        g.n_elems, g.n_nodes, g.d_elem_v, g.d_x, g.d_y,
        g.d_grad_rec, g.d_mass_lumped, g.d_elem_error);
    CUDA_CHECK_LAUNCH();
}

/**
 * Matrix-free -G^T M_L^{-1} G part of the OSS operator, added into out (weights g.d_elem_s,
 * lumped mass g.d_mass_lumped). Reference for the assembled form below (unit test).
 */
void apply_oss_projection_matrix_free(DeviceMeshBuffers& g, const double* in, double* out) {
    int threads = 256;
    int blocks_elems = (g.n_elems + threads - 1) / threads;
    CUDA_CHECK(cudaMemset(g.d_oss_gx, 0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_oss_gy, 0, g.n_nodes * sizeof(double)));
    k_accumulate_scalar_gradient<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, in, g.d_elem_s, g.d_oss_gx, g.d_oss_gy);
    k_oss_subtract_projection<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_bc_tag, g.d_elem_s,
        g.d_oss_gx, g.d_oss_gy, g.d_mass_lumped, out);
}

/**
 * Assembles the OSS projection -B^T M_L^{-1} B (k_assemble_oss_operator) for the current weights
 * g.d_elem_s and lumped mass; call after both change. Per CG application it costs two row passes
 * instead of two element passes with atomics (A3: 314 us -> see HANDOVER 11.12).
 */
void assemble_oss_projection(DeviceMeshBuffers& g) {
    int threads = 256;
    CUDA_CHECK(cudaMemset(g.d_oss_val, 0, 4 * (size_t)g.nnz * sizeof(double)));
    k_assemble_oss_operator<<<(g.n_elems + threads - 1) / threads, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_elem_s, g.d_row_ptr, g.d_col_ind, g.nnz, g.d_oss_val);
    k_inverse_lumped_mass<<<(g.n_nodes + threads - 1) / threads, threads>>>(g.n_nodes, g.d_mass_lumped, g.d_inv_m);
    CUDA_CHECK_LAUNCH();
}

/**
 * -B^T M_L^{-1} B part of the OSS operator, added into out (assemble_oss_projection first)
 */
void apply_oss_projection(DeviceMeshBuffers& g, const double* in, double* out) {
    int threads = 256;
    int blocks_nodes = (g.n_nodes + threads - 1) / threads;
    k_oss_apply_B<<<blocks_nodes, threads>>>(g.n_nodes, g.d_row_ptr, g.d_col_ind, g.nnz, g.d_oss_val,
                                             g.d_inv_m, in, g.d_oss_gx, g.d_oss_gy);
    k_oss_apply_BT<<<blocks_nodes, threads>>>(g.n_nodes, g.d_row_ptr, g.d_col_ind, g.nnz, g.d_oss_val,
                                              g.d_bc_tag, g.d_oss_gx, g.d_oss_gy, out);
}

/**
 * Scheme-consistent re-projection of the transferred velocities after a remesh. The time step
 * leaves u^n with the divergence its own (OSS-stabilized) pressure step implies, not zero. This
 * applies that pressure step once more to each transferred field, without changing p:
 *   (K + S) phi = -(rho/dt') (N_i, div u) - S p^n,   u <- u - (dt'/rho) M_L^{-1} grad phi
 * with the PPE operator of the next step (weights from u^n, dt' = projection step). On an
 * unchanged mesh this is close to the identity (phi is the lumped/consistent mismatch of the
 * last increment), so a remesh only corrects what the transfer changed. The AMG hierarchy is
 * built for K + S and stays valid for the next pressure solve. Fields: u^{n-1}, then u^n
 * (u^n only if both_levels is false: an ablation). phi_out, if given, receives phi of u^n.
 * build_amg, if given, replaces amg->setup() once K + S is assembled (the driver's remap path).
 */
int project_scheme_consistent(DeviceMeshBuffers& g, GpuLinearSolver& solver, double dt_proj,
                              double stab, AmgPreconditioner* amg, int& nonconverged,
                              const std::function<void()>& build_amg = nullptr,
                              bool both_levels = true, double* phi_out = nullptr) {
    int threads = 256;
    int blocks_nodes = (g.n_nodes + threads - 1) / threads;
    int blocks_elems = (g.n_elems + threads - 1) / threads;
    const bool oss = stab > 0.0;

    CUDA_CHECK(cudaMemset(g.d_mass_lumped, 0, g.n_nodes * sizeof(double)));
    k_assemble_lumped_mass<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_mass_lumped);
    if (oss) {
        k_compute_oss_weights<<<blocks_elems, threads>>>(
            g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_u, g.d_v,
            dt_proj, SimConfig::RHO, SimConfig::NU, stab, g.d_elem_s);
        assemble_oss_projection(g);
    }
    std::function<void(const double*, double*)> oss_op = [&g](const double* in, double* out) {
        apply_oss_projection(g, in, out);
    };
    std::function<void(const double*, double*)> prec;
    if (amg) prec = [amg](const double* r, double* z) { amg->apply(r, z); };

    int iters = 0;
    double* fields[2][3] = { {g.d_u_old, g.d_v_old, g.d_oss_gx}, {g.d_u, g.d_v, g.d_vort} };
    const int first = both_levels ? 0 : 1;
    for (int f = first; f < 2; ++f) {
        double* u = fields[f][0];
        double* v = fields[f][1];
        CUDA_CHECK(cudaMemset(g.d_Kp_val,    0, g.nnz * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_rhs_p,     0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_dp,        0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_div_num,   0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_grad_dp_x, 0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_grad_dp_y, 0, g.n_nodes * sizeof(double)));
        if (oss) {   // G(p^n) into the grad_dp scratch, as in the time step
            k_accumulate_scalar_gradient<<<blocks_elems, threads>>>(
                g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_p, g.d_elem_s, g.d_grad_dp_x, g.d_grad_dp_y);
        }
        k_assemble_pressure_poisson<<<blocks_elems, threads>>>(
            g.n_elems, g.d_elem_v, g.d_x, g.d_y, u, v,
            g.d_p, g.d_grad_dp_x, g.d_grad_dp_y, g.d_mass_lumped, g.d_elem_s,
            dt_proj, SimConfig::RHO, stab,
            g.d_row_ptr, g.d_col_ind, g.d_Kp_val, g.d_rhs_p, g.d_div_num);
        apply_bc_pressure(g.n_nodes, g.d_bc_tag, g.d_row_ptr, g.d_col_ind, g.d_Kp_val, g.d_rhs_p, g.d_dp, g.d_mass_lumped);
        if (amg && f == first) {
            if (build_amg) build_amg();
            else amg->setup(g.n_nodes, g.nnz, g.d_row_ptr, g.d_col_ind, g.d_Kp_val);
        }
        iters += solver.solve_pcg(g.n_nodes, g.d_row_ptr, g.d_col_ind,
                                  g.d_Kp_val, g.d_rhs_p, g.d_dp, 4000, 1.0e-8,
                                  oss ? &oss_op : nullptr, amg ? &prec : nullptr);
        nonconverged += solver.converged ? 0 : 1;

        CUDA_CHECK(cudaMemset(g.d_grad_dp_x, 0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_grad_dp_y, 0, g.n_nodes * sizeof(double)));
        CUDA_CHECK(cudaMemset(g.d_vort_num,  0, g.n_nodes * sizeof(double)));
        k_accumulate_grad_dp_and_vorticity<<<blocks_elems, threads>>>(
            g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_dp, u, v,
            g.d_grad_dp_x, g.d_grad_dp_y, g.d_vort_num);
        k_velocity_pressure_correction<<<blocks_nodes, threads>>>(
            g.n_nodes, g.d_bc_tag, dt_proj, SimConfig::RHO, /*p_update=*/0.0, /*rot_coeff=*/0.0,
            g.d_div_num, g.d_mass_lumped, g.d_grad_dp_x, g.d_grad_dp_y, g.d_vort_num, g.d_dp,
            u, v, g.d_p, fields[f][2]);
        CUDA_CHECK_LAUNCH();
    }
    if (phi_out)   // potential of the u^n field (the last one), for --reproject-shift
        CUDA_CHECK(cudaMemcpy(phi_out, g.d_dp, g.n_nodes * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(g.d_dp, 0, g.n_nodes * sizeof(double)));
    return iters;
}

/**
 * Discrete divergence-free re-projection of a transferred velocity field (u, v):
 *   K phi = -(rho/dt) (N_i, div u),   u <- u - (dt/rho) M_L^{-1} grad phi   (p unchanged)
 * Interpolated/injected velocities are not discretely solenoidal on the new mesh; this
 * removes the spurious divergence before the next time step. The AMG hierarchy of K is built
 * when setup_amg is true and reused otherwise (K depends on the mesh only). `vort_out`
 * receives the nodal vorticity of the projected field.
 */
int project_divergence_free(DeviceMeshBuffers& g, GpuLinearSolver& solver, double dt,
                            AmgPreconditioner* amg, bool setup_amg,
                            double* u, double* v, double* vort_out) {
    int threads = 256;
    int blocks_nodes = (g.n_nodes + threads - 1) / threads;
    int blocks_elems = (g.n_elems + threads - 1) / threads;

    CUDA_CHECK(cudaMemset(g.d_mass_lumped, 0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_Kp_val,      0, g.nnz * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_rhs_p,       0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_dp,          0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_grad_dp_x,   0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_grad_dp_y,   0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_vort_num,    0, g.n_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(g.d_div_num,     0, g.n_nodes * sizeof(double)));

    k_assemble_lumped_mass<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_mass_lumped);
    k_assemble_pressure_poisson<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, u, v,
        g.d_p, g.d_grad_dp_x, g.d_grad_dp_y, g.d_mass_lumped, g.d_elem_s,
        dt, SimConfig::RHO, /*stab=*/0.0,
        g.d_row_ptr, g.d_col_ind, g.d_Kp_val, g.d_rhs_p, g.d_div_num);
    apply_bc_pressure(g.n_nodes, g.d_bc_tag, g.d_row_ptr, g.d_col_ind, g.d_Kp_val, g.d_rhs_p, g.d_dp, g.d_mass_lumped);

    std::function<void(const double*, double*)> prec;
    if (amg) {
        if (setup_amg) amg->setup(g.n_nodes, g.nnz, g.d_row_ptr, g.d_col_ind, g.d_Kp_val);
        prec = [amg](const double* r, double* z) { amg->apply(r, z); };
    }
    int iters = solver.solve_pcg(g.n_nodes, g.d_row_ptr, g.d_col_ind,
                                 g.d_Kp_val, g.d_rhs_p, g.d_dp, 4000, 1.0e-8,
                                 nullptr, amg ? &prec : nullptr);

    k_accumulate_grad_dp_and_vorticity<<<blocks_elems, threads>>>(
        g.n_elems, g.d_elem_v, g.d_x, g.d_y, g.d_dp, u, v,
        g.d_grad_dp_x, g.d_grad_dp_y, g.d_vort_num);
    k_velocity_pressure_correction<<<blocks_nodes, threads>>>(
        g.n_nodes, g.d_bc_tag, dt, SimConfig::RHO, /*p_update=*/0.0, /*rot_coeff=*/0.0,
        g.d_div_num, g.d_mass_lumped, g.d_grad_dp_x, g.d_grad_dp_y, g.d_vort_num, g.d_dp,
        u, v, g.d_p, vort_out);
    CUDA_CHECK_LAUNCH();

    // The cleanup potential is not a pressure increment: do not warm-start the PPE with it.
    CUDA_CHECK(cudaMemset(g.d_dp, 0, g.n_nodes * sizeof(double)));
    return iters;
}

void set_initial_condition(const Mesh2D& mesh,
                           std::vector<double>& h_u,
                           std::vector<double>& h_v,
                           std::vector<double>& h_p,
                           double inflow_peak = SimConfig::U_MAX) {
    h_u.assign(mesh.num_nodes, 0.0);
    h_v.assign(mesh.num_nodes, 0.0);
    h_p.assign(mesh.num_nodes, 0.0);
    for (int i = 0; i < mesh.num_nodes; ++i) {
        double yi = mesh.y[i];
        double profile = 4.0 * inflow_peak * yi * (SimConfig::H_CHANNEL - yi)
                       / (SimConfig::H_CHANNEL * SimConfig::H_CHANNEL);
        if (mesh.bc_tag[i] == BC_INLET) {
            h_u[i] = profile;
        } else if (mesh.bc_tag[i] == BC_INTERIOR || mesh.bc_tag[i] == BC_OUTLET) {
            double r = std::hypot(mesh.x[i] - SimConfig::X_CYL, yi - SimConfig::Y_CYL);
            double damp = 1.0 - std::exp(-std::max(0.0, r - SimConfig::R_CYL) / 0.03);
            h_u[i] = damp * profile;
        }
    }
}

/**
 * Rebuilds solver topology for the freshly extracted mesh and aborts on an invalid mesh.
 */
MeshQuality finalize_mesh(Mesh2D& mesh) {
    build_csr_and_boundary_topology(mesh);
    MeshQuality q;
    if (!validate_mesh(mesh, q)) {
        fprintf(stderr, "[AMR] mesh validation failed, aborting\n");
        exit(EXIT_FAILURE);
    }
    return q;
}

void print_amr_report(const char* tag, const AmrStats& st, const Mesh2D& mesh, const MeshQuality& q,
                      int proj_iters) {
    std::cout << "  --> [AMR " << tag << "] marked refine " << st.marked_refine
              << " / coarsen " << st.marked_coarsen
              << " | bisections " << st.bisections << " (+" << st.closure_bisections << " closure)"
              << " | removed vertices " << st.removed_vertices << "\n"
              << "      mesh: " << mesh.num_nodes << " nodes, " << mesh.num_elems << " elements"
              << " (" << st.leaves_before << " -> " << st.leaves_after << "), CSR nnz = " << mesh.nnz
              << " | max level " << q.max_level << " | min angle "
              << std::setprecision(2) << q.min_angle_deg << " deg"
              << " | adapt " << std::setprecision(1) << st.elapsed_ms << " ms";
    if (proj_iters >= 0) std::cout << " | div-projection PCG " << proj_iters << " it";
    std::cout << std::endl << std::setprecision(4);
}

/**
 * Accumulated wall time of the remeshing phases (host + device), reported at the end of a run.
 */
struct RemeshProfile {
    int    count = 0;
    double indicator = 0.0;     // GPU indicators + device -> host download + scatter to the forest
    double lookahead = 0.0, mark = 0.0, neighbours = 0.0, coarsen = 0.0, refine = 0.0;   // adapt() phases
    double extract = 0.0;       // leaf extraction, Morton ordering
    double csr = 0.0;           // CSR graph + cylinder boundary edges
    double validate = 0.0;      // validate_mesh
    double upload = 0.0;        // host -> device mesh and solution
    double colouring = 0.0;     // momentum ILU(0) colouring of the new pattern
    double amg_setup = 0.0;     // AMG setup of the re-projection operator
    double projection = 0.0;    // re-projection without its AMG setup
    // --amr-overlap: adapt/extract/CSR/validate run in a worker thread while the GPU steps; only the
    // wait at the join step and the second transfer are on the critical path
    double overlapped = 0.0;    // worker thread time (included in the adapt ... validate items)
    double join_wait = 0.0;     // main thread waiting for the worker at the join step
    double transfer = 0.0;      // download at the join step + AdaptiveMesh::retransfer

    void add(const AmrStats& st) {
        ++count;
        lookahead += st.ms_lookahead;  mark += st.ms_mark;  neighbours += st.ms_neighbours;
        coarsen += st.ms_coarsen;  refine += st.ms_refine;
    }

    void report(double total_ms) const {
        if (count == 0) return;
        auto item = [&](const char* name, double ms) {
            std::cout << "    " << std::left << std::setw(26) << name << std::right
                      << std::setw(9) << std::setprecision(2) << ms / 1000.0 << " s  "
                      << std::setw(6) << std::setprecision(1) << 100.0 * ms / total_ms << " %  "
                      << std::setw(8) << std::setprecision(1) << ms / count << " ms/remesh\n";
        };
        std::cout << "[Timing] remesh breakdown over " << count << " remeshes:\n" << std::fixed;
        if (overlapped > 0.0) {   // percentages of all remesh work, hidden or not
            total_ms = indicator + overlapped + join_wait + transfer + upload + colouring + amg_setup + projection;
            std::cout << "    (--amr-overlap: adapt ... validate ran in a worker thread, " << std::setprecision(2)
                      << overlapped / 1000.0 << " s, hidden behind stepping except the join wait)\n";
        }
        item("indicators + download", indicator);
        item("adapt: look-ahead", lookahead);
        item("adapt: marking", mark);
        item("adapt: neighbours/buffer", neighbours);
        item("adapt: coarsening", coarsen);
        item("adapt: refinement", refine);
        item("extract", extract);
        item("CSR build", csr);
        item("validate_mesh", validate);
        item("upload", upload);
        item("ILU colouring", colouring);
        item("re-projection AMG setup", amg_setup);
        item("re-projection solve", projection);
        if (overlapped > 0.0) {
            item("overlap: join wait", join_wait);
            item("overlap: transfer", transfer);
        }
    }
};
