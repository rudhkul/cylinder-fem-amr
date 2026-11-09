// GPU finite-element kernels (assembly, BCs, indicators, forces) and the manufactured solution
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// CUDA Device Kernels: Error Estimator, FEM Assembly, Boundary Conditions & Solvers
// ============================================================================

/**
 * Binary search helper inside a CSR row on the GPU
 */
__device__ __forceinline__ int find_csr_index(
    const int* __restrict__ col_ind,
    int row_start,
    int row_end,
    int target_col
) {
    int low = row_start;
    int high = row_end - 1;
    while (low <= high) {
        int mid = (low + high) >> 1;
        int c = col_ind[mid];
        if (c == target_col) return mid;
        if (c < target_col)  low = mid + 1;
        else                 high = mid - 1;
    }
    return -1;
}

/**
 * Element-wise FEM Assembly Kernel for Tentative Velocity (Advection-Diffusion-SUPG)
 * and Lumped Mass Vector, for a two-step time discretization
 *   du/dt ~ a0 u* - (a1 u^n + a2 u^{n-1}),   advecting velocity w = c1 u^n + c2 u^{n-1}:
 *   ( a0 M + nu K + C(w) + S_SUPG(w) ) u* = M (a1 u^n + a2 u^{n-1}) - (1/rho) grad(p^n)
 *   with C(w) the skew-symmetric (energy-conserving) Galerkin convection operator
 *   BDF2:           a0 = 3/(2dt), a1 = 2/dt, a2 = -1/(2dt), w = 2u^n - u^{n-1}
 *   backward Euler: a0 = a1 = 1/dt, a2 = 0,                 w = u^n
 * The SUPG test function weights the same discrete time derivative (consistent SUPG).
 */
__global__ void k_assemble_momentum_fem(
    int num_elems,
    const int* __restrict__ elem_v,      // [3 * num_elems]
    const double* __restrict__ x,        // [num_nodes]
    const double* __restrict__ y,        // [num_nodes]
    const double* __restrict__ u_old,    // u^n [num_nodes]
    const double* __restrict__ v_old,    // v^n [num_nodes]
    const double* __restrict__ u_prev,   // u^{n-1} [num_nodes]
    const double* __restrict__ v_prev,   // v^{n-1} [num_nodes]
    const double* __restrict__ p_old,    // [num_nodes]
    double dt,
    double nu,
    double rho,
    double a0,
    double a1,
    double a2,
    double c1,
    double c2,
    double tau_time,                     // time scale in the SUPG tau (normally dt)
    double skew_coef,                    // 0.5: skew-symmetric convection, 0: convective form
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    double* __restrict__ A_val,          // CSR matrix values [nnz]
    double* __restrict__ rhs_u,          // [num_nodes]
    double* __restrict__ rhs_v,          // [num_nodes]
    double* __restrict__ mass_lumped,    // [num_nodes]
    const double* __restrict__ force_x = nullptr,   // nodal body force at t^{n+1} (MMS), or null
    const double* __restrict__ force_y = nullptr
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];
    int nodes[3] = {n0, n1, n2};

    double x0 = x[n0], y0 = y[n0];
    double x1 = x[n1], y1 = y[n1];
    double x2 = x[n2], y2 = y[n2];

    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };

    // Time-history data a1 u^n + a2 u^{n-1}, element-averaged advecting velocity w and its
    // (element-constant) divergence for the skew-symmetric convection term
    double u_h[3], v_h[3], w_u = 0.0, w_v = 0.0, div_w = 0.0;
    for (int k = 0; k < 3; ++k) {
        double un = u_old[nodes[k]], vn = v_old[nodes[k]];
        double um = u_prev[nodes[k]], vm = v_prev[nodes[k]];
        u_h[k] = a1 * un + a2 * um;
        v_h[k] = a1 * vn + a2 * vm;
        double wu = c1 * un + c2 * um, wv = c1 * vn + c2 * vm;
        w_u += wu / 3.0;
        w_v += wv / 3.0;
        div_w += wu * dNdx[k] + wv * dNdy[k];
    }
    double p_e[3] = { p_old[n0], p_old[n1], p_old[n2] };

    double u_bar = w_u;
    double v_bar = w_v;
    double vel_mag = sqrt(u_bar * u_bar + v_bar * v_bar) + 1.0e-12;

    // Characteristic element length h_K and SUPG stabilization parameter tau_SUPG
    double h_K = sqrt(2.0 * area);
    // Transient tau (Tezduyar): a 2/T term is required for stability of this fractional-step
    // SUPG with lagged pressure (dropping it diverges on coarse cells). T is fixed (1 ms by
    // default, --supg-tau-time) so the spatial stabilization does not change with dt; T = dt
    // degrades BDF2 to an observed O(dt) because tau ~ dt/2 then perturbs the discretization.
    double term_t = 2.0 / tau_time;
    double term_c = 2.0 * vel_mag / h_K;
    double term_d = 4.0 * nu / (h_K * h_K);
    double tau_supg = 1.0 / sqrt(term_t * term_t + term_c * term_c + 9.0 * term_d * term_d);

    // Element (kinematic) pressure gradient
    double dpdx = (p_e[0] * dNdx[0] + p_e[1] * dNdx[1] + p_e[2] * dNdx[2]) / rho;
    double dpdy = (p_e[0] * dNdy[0] + p_e[1] * dNdy[1] + p_e[2] * dNdy[2]) / rho;

    // Streamline derivative of shape functions: u_bar * dN/dx + v_bar * dN/dy
    double u_gradN[3];
    for (int i = 0; i < 3; ++i) {
        u_gradN[i] = u_bar * dNdx[i] + v_bar * dNdy[i];
    }

    double m_lump = area / 3.0;

    for (int i = 0; i < 3; ++i) {
        int row = nodes[i];
        int r_start = row_ptr[row];
        int r_end   = row_ptr[row + 1];

        atomicAdd(&mass_lumped[row], m_lump);

        double f_u = 0.0;
        double f_v = 0.0;

        for (int j = 0; j < 3; ++j) {
            int col = nodes[j];
            // Consistent mass matrix M_ij = (area / 12) * (1 + delta_ij)
            double M_ij = (i == j) ? (area / 6.0) : (area / 12.0);
            // Viscous stiffness matrix K_ij = area * (dN_i/dx dN_j/dx + dN_i/dy dN_j/dy)
            double K_ij = area * (dNdx[i] * dNdx[j] + dNdy[i] * dNdy[j]);
            // Skew-symmetric convective matrix (w . grad N_j, N_i) + 1/2 ((div w) N_j, N_i):
            // energy-neutral even though the discrete w is not exactly divergence-free, so the
            // Galerkin convection cannot inject energy when tau ~ dt is small
            double C_ij = (area / 3.0) * u_gradN[j] + skew_coef * div_w * M_ij;
            // SUPG stabilization matrix S_ij = area * tau_supg * (u . grad N_i) * (u . grad N_j)
            double S_ij = area * tau_supg * u_gradN[i] * u_gradN[j];
            // SUPG mass-time coupling (tau (u . grad N_i), N_j)
            double M_supg_ij = (area / 3.0) * tau_supg * u_gradN[i];

            double A_ij = a0 * (M_ij + M_supg_ij) + nu * K_ij + C_ij + S_ij;

            int idx = find_csr_index(col_ind, r_start, r_end, col);
            if (idx >= 0) {
                atomicAdd(&A_val[idx], A_ij);
            }

            // Time-history contribution on RHS
            f_u += (M_ij + M_supg_ij) * u_h[j];
            f_v += (M_ij + M_supg_ij) * v_h[j];
            // Body force, Galerkin + SUPG weighted (consistent with the residual)
            if (force_x) {
                f_u += (M_ij + M_supg_ij) * force_x[col];
                f_v += (M_ij + M_supg_ij) * force_y[col];
            }
        }

        // Subtract pressure gradient contribution (with SUPG weight)
        double test_weight = (area / 3.0) + area * tau_supg * u_gradN[i];
        f_u -= test_weight * dpdx;
        f_v -= test_weight * dpdy;

        atomicAdd(&rhs_u[row], f_u);
        atomicAdd(&rhs_v[row], f_v);
    }
}

/**
 * Element-wise FEM Assembly Kernel for the (stabilized) Pressure Poisson Equation (PPE),
 * multiplied through by rho/dt:
 *   K dp + S (p^n + dp) = -(rho/dt) (N, div u*),
 *   S(x, q) = sum_K s_K |K| grad x . grad q  -  sum_i G_i(x) . G_i(q) / m_i,
 *   G_i(f)  = sum_{K ni i} sqrt(s_K) |K|/3 grad f|_K,     s_K = rho tau_K / dt.
 * S is the orthogonal-subscale (OSS) pressure stabilization (Codina 2001) with the gradient
 * projection treated implicitly: it penalizes only the part of sqrt(s) grad p orthogonal to
 * its lumped L2 projection onto continuous P1, so smooth pressures (and smooth increments dp)
 * are not penalized and incompressibility is enforced up to that consistent residual. S is
 * symmetric positive semi-definite (Cauchy-Schwarz on each nodal patch), so K + S is SPD.
 * It removes the equal-order P1-P1 instability of projection methods for dt < c h^2.
 * This kernel assembles K + sum_K s_K K_K into the CSR matrix and the S p^n part of the RHS;
 * the -G^T M^{-1} G part of S dp is applied inside CG (apply_oss_projection, assembled as -B^T M^{-1} B).
 * gp_num_x/y must hold G(p^n) and elem_s the per-element s_K (k_compute_oss_weights).
 * stab = 0 gives the plain PPE (used for the divergence-cleaning projection after remeshing).
 * Also accumulates (N_i, div u*) into div_num for the rotational pressure update.
 */
__global__ void k_assemble_pressure_poisson(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ u_star,
    const double* __restrict__ v_star,
    const double* __restrict__ p_old,       // p^n
    const double* __restrict__ gp_num_x,    // G_i(p^n), x component
    const double* __restrict__ gp_num_y,
    const double* __restrict__ mass_lumped,
    const double* __restrict__ elem_s,      // s_K = rho tau_K / dt
    double dt,
    double rho,
    double stab,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    double* __restrict__ Kp_val,
    double* __restrict__ rhs_p,
    double* __restrict__ div_num
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];
    int nodes[3] = {n0, n1, n2};

    double x0 = x[n0], y0 = y[n0];
    double x1 = x[n1], y1 = y[n1];
    double x2 = x[n2], y2 = y[n2];

    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };

    double div_u = u_star[n0] * dNdx[0] + u_star[n1] * dNdx[1] + u_star[n2] * dNdx[2]
                 + v_star[n0] * dNdy[0] + v_star[n1] * dNdy[1] + v_star[n2] * dNdy[2];

    double rhs_contrib = -(rho / dt) * (area / 3.0) * div_u;

    // OSS: S p^n contribution to the RHS, per local row i:
    //   s |K| grad N_i . grad p^n  -  sqrt(s) |K|/3 grad N_i . sum_{k in K} G_k(p^n) / m_k
    double s = 0.0, sx = 0.0, sy = 0.0, gx = 0.0, gy = 0.0;
    if (stab > 0.0) {
        s = elem_s[e];
        double rs = sqrt(s);
        gx = p_old[n0] * dNdx[0] + p_old[n1] * dNdx[1] + p_old[n2] * dNdx[2];
        gy = p_old[n0] * dNdy[0] + p_old[n1] * dNdy[1] + p_old[n2] * dNdy[2];
        for (int k = 0; k < 3; ++k) {
            double inv_m = 1.0 / fmax(mass_lumped[nodes[k]], 1.0e-14);
            sx += gp_num_x[nodes[k]] * inv_m;
            sy += gp_num_y[nodes[k]] * inv_m;
        }
        sx *= rs / 3.0;
        sy *= rs / 3.0;
    }

    for (int i = 0; i < 3; ++i) {
        int row = nodes[i];
        int r_start = row_ptr[row];
        int r_end   = row_ptr[row + 1];

        for (int j = 0; j < 3; ++j) {
            int col = nodes[j];
            double K_ij = area * (dNdx[i] * dNdx[j] + dNdy[i] * dNdy[j]);
            int idx = find_csr_index(col_ind, r_start, r_end, col);
            if (idx >= 0) {
                atomicAdd(&Kp_val[idx], (1.0 + s) * K_ij);
            }
        }
        double stab_i = s * area * (dNdx[i] * gx + dNdy[i] * gy)
                      - area * (dNdx[i] * sx + dNdy[i] * sy);
        atomicAdd(&rhs_p[row], rhs_contrib - stab_i);
        atomicAdd(&div_num[row], (area / 3.0) * div_u);
    }
}

// ============================================================================
// Manufactured Solution (code verification, --mms)
// ============================================================================
/**
 * Smooth, divergence-free exact solution on the benchmark domain (generated by
 * scripts/mms_generate.py with sympy; regenerate rather than edit):
 *   psi = A(t) f(x) g(y) / b,  u = psi_y,  v = -psi_x,  p = A(t) sin(k (L - x)) cos(b y) / 2
 *   f = cos(k (x - L)) - cos(2 k (x - L)) / 4,  g = sin(b y) + cos(b y) / 2,
 *   k = 2 pi / 0.82,  b = pi / H,  A = 1 + sin(10 pi t) / 2.
 * f'(L) = f''(L) = 0 and p(L, y) = 0, so the solution satisfies the code's outflow conditions
 * (nu du/dn = 0, p = 0) exactly; inflow, walls and cylinder get the exact Dirichlet values.
 * (fx, fy) = u_t + u.grad u - nu Lap u + grad p / rho is the body force.
 */
struct MmsValues { double u, v, p, fx, fy, ux, uy, vx, vy; };

__host__ __device__ inline MmsValues mms_exact(double x, double y, double t) {
    MmsValues r;
    const double c0 = 2.4390243902439024*x - 5.3658536585365854;
    const double c1 = 3.1415926535897932*c0;
    const double c2 = cos(c1);
    const double c3 = 15.324842212633138*x - 33.714652867792903;
    const double c4 = cos(c3);
    const double c5 = c2 - 0.25*c4;
    const double c6 = 7.6624211063165689;
    const double c7 = c6*y;
    const double c8 = sin(c7);
    const double c9 = cos(c7);
    const double c10 = -3.8312105531582844*c8 + 7.6624211063165689*c9;
    const double c11 = c10*c5;
    const double c12 = 31.415926535897932*t;
    const double c13 = sin(c12);
    const double c14 = 0.5*c13 + 1.0;
    const double c15 = 0.13050705333535418*c14;
    const double c16 = c8 + 0.5*c9;
    const double c17 = sin(c1);
    const double c18 = sin(c3);
    const double c19 = -c17*c6 + 3.8312105531582844*c18;
    const double c20 = c16*c19;
    const double c21 = 0.25*c13 + 0.5;
    const double c22 = -3.1415926535897932*c0;
    const double c23 = c21*sin(c22);
    const double c24 = cos(c12);
    const double c25 = 9.8696044010893586;
    const double c26 = c14*c25;
    const double c27 = 0.00297441998810232*c26*(c8 - 2.0*c9);
    const double c28 = 1.0/c25;
    const double c29 = pow(c14, 2);
    const double c30 = 5.9488399762046401*c25;
    const double c31 = c5*(-2.97441998810232*c25*c9 - c30*c8);
    const double c32 = 0.1681*c28*c29;
    const double c33 = c16*(-c2*c30 + 5.9488399762046401*c25*c4);
    const double c34 = c10*c15*c19;
    r.u = c11*c15;
    r.v = -c15*c20;
    r.p = c23*c9;
    r.fx = 0.1681*pow(c10, 2)*c19*c28*c29*c5 + 2.05*c10*c24*c5 - c20*c31*c32 - c21*c6*c9*cos(c22) - c27*c5 - c27*(c2 - c4);
    r.fy = c10*c16*pow(c19, 2)*c32 - c11*c32*c33 + 0.0059488399762046401*c16*c26*(c17 - 2.0*c18) - 2.05*c20*c24 - c23*c6*c8 + 0.00148720999405116*c26*(2.0*c17 - c18)*(2.0*c8 + c9);
    r.ux = c34;
    r.uy = c15*c31;
    r.vx = -c15*c33;
    r.vy = -c34;
    return r;
}

__global__ void k_mms_forcing(int n, const double* __restrict__ x, const double* __restrict__ y,
                              double t, double* __restrict__ fx, double* __restrict__ fy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    MmsValues r = mms_exact(x[i], y[i], t);
    fx[i] = r.fx;
    fy[i] = r.fy;
}

/**
 * Errors against the exact solution at time t, by a 7-point degree-5 quadrature per element:
 * out[0] = ||u_h - u||^2, out[1] = |u_h - u|_{H1}^2, out[2] = ||p_h - p||^2,
 * out[3] = ||u||^2, out[4] = |u|_{H1}^2, out[5] = ||p||^2.
 */
__global__ void k_mms_errors(int num_elems, const int* __restrict__ elem_v,
                             const double* __restrict__ x, const double* __restrict__ y,
                             const double* __restrict__ uh, const double* __restrict__ vh,
                             const double* __restrict__ ph, double t, double* __restrict__ out) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;
    int n[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double x0 = x[n[0]], y0 = y[n[0]], x1 = x[n[1]], y1 = y[n[1]], x2 = x[n[2]], y2 = y[n[2]];
    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    double inv2A = 1.0 / detJ;
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };
    double uhx = 0, uhy = 0, vhx = 0, vhy = 0;
    for (int k = 0; k < 3; ++k) {
        uhx += uh[n[k]] * dNdx[k];  uhy += uh[n[k]] * dNdy[k];
        vhx += vh[n[k]] * dNdx[k];  vhy += vh[n[k]] * dNdy[k];
    }
    const double a1 = 0.059715871789770, b1 = 0.470142064105115, w1 = 0.132394152788506;
    const double a2 = 0.797426985353087, b2 = 0.101286507323456, w2 = 0.125939180544827;
    const double L[7][3] = { {1.0 / 3, 1.0 / 3, 1.0 / 3},
                             {a1, b1, b1}, {b1, a1, b1}, {b1, b1, a1},
                             {a2, b2, b2}, {b2, a2, b2}, {b2, b2, a2} };
    const double W[7] = { 0.225, w1, w1, w1, w2, w2, w2 };
    double acc[6] = {0, 0, 0, 0, 0, 0};
    for (int q = 0; q < 7; ++q) {
        double xq = L[q][0] * x0 + L[q][1] * x1 + L[q][2] * x2;
        double yq = L[q][0] * y0 + L[q][1] * y1 + L[q][2] * y2;
        double uq = 0, vq = 0, pq = 0;
        for (int k = 0; k < 3; ++k) {
            uq += L[q][k] * uh[n[k]];  vq += L[q][k] * vh[n[k]];  pq += L[q][k] * ph[n[k]];
        }
        MmsValues r = mms_exact(xq, yq, t);
        double w = W[q] * area;
        acc[0] += w * ((uq - r.u) * (uq - r.u) + (vq - r.v) * (vq - r.v));
        acc[1] += w * ((uhx - r.ux) * (uhx - r.ux) + (uhy - r.uy) * (uhy - r.uy)
                     + (vhx - r.vx) * (vhx - r.vx) + (vhy - r.vy) * (vhy - r.vy));
        acc[2] += w * (pq - r.p) * (pq - r.p);
        acc[3] += w * (r.u * r.u + r.v * r.v);
        acc[4] += w * (r.ux * r.ux + r.uy * r.uy + r.vx * r.vx + r.vy * r.vy);
        acc[5] += w * r.p * r.p;
    }
    for (int k = 0; k < 6; ++k) atomicAdd(&out[k], acc[k]);
}

/**
 * Apply Dirichlet Boundary Conditions to Momentum System (Inlet, Wall, Cylinder).
 * mms_time >= 0 imposes the manufactured solution at that time instead.
 */
__global__ void k_apply_bc_momentum(
    int num_nodes,
    const int* __restrict__ bc_tag,
    const double* __restrict__ y,
    double u_max,
    double H,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    double* __restrict__ A_val,
    double* __restrict__ rhs_u,
    double* __restrict__ rhs_v,
    double* __restrict__ u_vec,
    double* __restrict__ v_vec,
    const double* __restrict__ x = nullptr,
    double mms_time = -1.0
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int tag = bc_tag[i];
    if (tag == BC_INLET || tag == BC_WALL || tag == BC_CYLINDER) {
        double u_bc = 0.0;
        double v_bc = 0.0;
        if (mms_time >= 0.0) {
            MmsValues r = mms_exact(x[i], y[i], mms_time);
            u_bc = r.u;
            v_bc = r.v;
        } else if (tag == BC_INLET) {
            double yi = y[i];
            u_bc = 4.0 * u_max * yi * (H - yi) / (H * H);
        }

        int r_start = row_ptr[i];
        int r_end   = row_ptr[i + 1];
        for (int idx = r_start; idx < r_end; ++idx) {
            A_val[idx] = (col_ind[idx] == i) ? 1.0 : 0.0;
        }
        rhs_u[i] = u_bc;
        rhs_v[i] = v_bc;
        u_vec[i] = u_bc;
        v_vec[i] = v_bc;
    }
}

/**
 * Apply Dirichlet Boundary Conditions to Pressure Poisson System (Outlet p = 0)
 * Note: For symmetric CG, we zero off-diagonals on outlet rows and also eliminate
 * columns corresponding to BC_OUTLET nodes so symmetry is preserved.
 */
__global__ void k_apply_bc_pressure(
    int num_nodes,
    const int* __restrict__ bc_tag,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    double* __restrict__ Kp_val,
    double* __restrict__ rhs_p,
    double* __restrict__ dp_vec
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int r_start = row_ptr[i];
    int r_end   = row_ptr[i + 1];

    if (bc_tag[i] == BC_OUTLET) {
        for (int idx = r_start; idx < r_end; ++idx) {
            Kp_val[idx] = (col_ind[idx] == i) ? 1.0 : 0.0;
        }
        rhs_p[i]  = 0.0;
        dp_vec[i] = 0.0;
    } else {
        // Symmetric elimination of homogeneous Dirichlet columns (dp_j = 0 on BC_OUTLET)
        for (int idx = r_start; idx < r_end; ++idx) {
            int col = col_ind[idx];
            if (col != i && bc_tag[col] == BC_OUTLET) {
                Kp_val[idx] = 0.0;
            }
        }
    }
}

/**
 * Nodal Pressure-Gradient & Vorticity Accumulation Kernel for Velocity Projection
 */
__global__ void k_accumulate_grad_dp_and_vorticity(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ dp,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ grad_dp_x,
    double* __restrict__ grad_dp_y,
    double* __restrict__ vort_num
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];
    int nodes[3] = {n0, n1, n2};

    double x0 = x[n0], y0 = y[n0];
    double x1 = x[n1], y1 = y[n1];
    double x2 = x[n2], y2 = y[n2];

    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };

    double dpdx = dp[n0] * dNdx[0] + dp[n1] * dNdx[1] + dp[n2] * dNdx[2];
    double dpdy = dp[n0] * dNdy[0] + dp[n1] * dNdy[1] + dp[n2] * dNdy[2];

    double dvdx = v[n0] * dNdx[0] + v[n1] * dNdx[1] + v[n2] * dNdx[2];
    double dudy = u[n0] * dNdy[0] + u[n1] * dNdy[1] + u[n2] * dNdy[2];
    double omega_e = dvdx - dudy;

    double w = area / 3.0;
    for (int i = 0; i < 3; ++i) {
        int node = nodes[i];
        atomicAdd(&grad_dp_x[node], w * dpdx);
        atomicAdd(&grad_dp_y[node], w * dpdy);
        atomicAdd(&vort_num[node],  w * omega_e);
    }
}

/**
 * Velocity Projection & Pressure Update Kernel:
 *   u^{n+1} = u* - (dt / rho) * M_L^{-1} grad(dp)
 *   p^{n+1} = p^n + p_update * (dp - rot_coeff * M_L^{-1} (N, div u*))
 * rot_coeff = rho * nu gives the rotational incremental scheme (Timmermans et al. 1996), which
 * removes the artificial pressure boundary layer of the standard scheme (rot_coeff = 0).
 * p_update = 0 gives a pure divergence-cleaning projection.
 */
__global__ void k_velocity_pressure_correction(
    int num_nodes,
    const int* __restrict__ bc_tag,
    double dt,
    double rho,
    double p_update,
    double rot_coeff,
    const double* __restrict__ div_num,
    const double* __restrict__ mass_lumped,
    const double* __restrict__ grad_dp_x,
    const double* __restrict__ grad_dp_y,
    const double* __restrict__ vort_num,
    const double* __restrict__ dp,
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ p,
    double* __restrict__ vort
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    double m_i = fmax(mass_lumped[i], 1.0e-14);
    vort[i] = vort_num[i] / m_i;
    p[i]   += p_update * (dp[i] - rot_coeff * div_num[i] / m_i);

    int tag = bc_tag[i];
    if (tag == BC_INTERIOR || tag == BC_OUTLET) {
        u[i] -= (dt / rho) * (grad_dp_x[i] / m_i);
        v[i] -= (dt / rho) * (grad_dp_y[i] / m_i);
    }
}

/**
 * out_i += c * (grad N_i, grad f): the P1 stiffness matrix applied to f, scaled
 */
__global__ void k_stiffness_apply(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ f,
    double c,
    double* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;
    int n[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double x0 = x[n[0]], y0 = y[n[0]], x1 = x[n[1]], y1 = y[n[1]], x2 = x[n[2]], y2 = y[n[2]];
    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;
    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };
    double gx = f[n[0]] * dNdx[0] + f[n[1]] * dNdx[1] + f[n[2]] * dNdx[2];
    double gy = f[n[0]] * dNdy[0] + f[n[1]] * dNdy[1] + f[n[2]] * dNdy[2];
    for (int i = 0; i < 3; ++i) atomicAdd(&out[n[i]], c * area * (dNdx[i] * gx + dNdy[i] * gy));
}

/**
 * Surface-traction force on the cylinder (diagnostic cross-check of the volume formula):
 *   F_i = sum_edges |E| [ nu (grad u_i)_K . eta - p_E eta_i ],  eta = outward normal of the circle
 * with the velocity gradient of the owning element K and the edge-mean pressure (exact for P1).
 * edges[3 k + 0..2] = (a, b, K). Accumulates (F_x, F_y) into out[0..1].
 */
__global__ void k_surface_forces(
    int num_edges,
    const int* __restrict__ edges,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ p,
    double nu,
    double rho,
    double xc,
    double yc,
    double* __restrict__ out
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_edges) return;
    int a = edges[3 * k + 0], b = edges[3 * k + 1], e = edges[3 * k + 2];
    int n[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double x0 = x[n[0]], y0 = y[n[0]], x1 = x[n[1]], y1 = y[n[1]], x2 = x[n[2]], y2 = y[n[2]];
    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double dNdx[3] = { (y1 - y2) / detJ, (y2 - y0) / detJ, (y0 - y1) / detJ };
    double dNdy[3] = { (x2 - x1) / detJ, (x0 - x2) / detJ, (x1 - x0) / detJ };
    double ux = 0, uy = 0, vx = 0, vy = 0;
    for (int i = 0; i < 3; ++i) {
        ux += u[n[i]] * dNdx[i];  uy += u[n[i]] * dNdy[i];
        vx += v[n[i]] * dNdx[i];  vy += v[n[i]] * dNdy[i];
    }
    double dx = x[b] - x[a], dy = y[b] - y[a];
    double len = sqrt(dx * dx + dy * dy);
    double ex = dy / len, ey = -dx / len;
    if (ex * (0.5 * (x[a] + x[b]) - xc) + ey * (0.5 * (y[a] + y[b]) - yc) < 0.0) { ex = -ex; ey = -ey; }
    double pm = 0.5 * (p[a] + p[b]) / rho;
    atomicAdd(&out[0], len * (nu * (ux * ex + uy * ey) - pm * ex));
    atomicAdd(&out[1], len * (nu * (vx * ex + vy * ey) - pm * ey));
}

/**
 * OSS weights s_K = rho tau_K / dt with tau_K = stab / (rho (4 nu / h^2 + 2 |u*| / h)).
 */
__global__ void k_compute_oss_weights(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ u_star,
    const double* __restrict__ v_star,
    double dt,
    double rho,
    double nu,
    double stab,
    double* __restrict__ elem_s
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;
    int n0 = elem_v[3 * e + 0], n1 = elem_v[3 * e + 1], n2 = elem_v[3 * e + 2];
    double area = 0.5 * fabs((x[n1] - x[n0]) * (y[n2] - y[n0]) - (x[n2] - x[n0]) * (y[n1] - y[n0]));
    double h  = sqrt(2.0 * area);
    double ub = (u_star[n0] + u_star[n1] + u_star[n2]) / 3.0;
    double vb = (v_star[n0] + v_star[n1] + v_star[n2]) / 3.0;
    double tau = stab / (rho * (4.0 * nu / (h * h) + 2.0 * sqrt(ub * ub + vb * vb) / h) + 1.0e-300);
    elem_s[e] = rho * tau / dt;
}

/**
 * Lumped-projection numerator of a weighted P1 scalar gradient:
 *   g_i = sum_K w_K |K|/3 grad f|_K,   w_K = sqrt(elem_s[K]) (or 1 if elem_s == nullptr)
 * (divide by the lumped mass to get the nodal L2-projected gradient)
 */
__global__ void k_accumulate_scalar_gradient(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ f,
    const double* __restrict__ elem_s,
    double* __restrict__ gx_num,
    double* __restrict__ gy_num
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];

    double x0 = x[n0], y0 = y[n0];
    double x1 = x[n1], y1 = y[n1];
    double x2 = x[n2], y2 = y[n2];
    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double gx = (f[n0] * (y1 - y2) + f[n1] * (y2 - y0) + f[n2] * (y0 - y1)) * inv2A;
    double gy = (f[n0] * (x2 - x1) + f[n1] * (x0 - x2) + f[n2] * (x1 - x0)) * inv2A;
    double w = (elem_s ? sqrt(elem_s[e]) : 1.0) * area / 3.0;
    atomicAdd(&gx_num[n0], w * gx);  atomicAdd(&gy_num[n0], w * gy);
    atomicAdd(&gx_num[n1], w * gx);  atomicAdd(&gy_num[n1], w * gy);
    atomicAdd(&gx_num[n2], w * gx);  atomicAdd(&gy_num[n2], w * gy);
}

/**
 * Matrix-free part of the OSS operator: y_i -= sum_K sqrt(s_K) |K|/3 grad N_i . sum_{k in K} g_k / m_k
 * with g = G(x) from k_accumulate_scalar_gradient. Rows of Dirichlet (outlet) nodes are skipped.
 */
__global__ void k_oss_subtract_projection(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ bc_tag,
    const double* __restrict__ elem_s,
    const double* __restrict__ gx_num,
    const double* __restrict__ gy_num,
    const double* __restrict__ mass_lumped,
    double* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;
    int nodes[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double x0 = x[nodes[0]], y0 = y[nodes[0]];
    double x1 = x[nodes[1]], y1 = y[nodes[1]];
    double x2 = x[nodes[2]], y2 = y[nodes[2]];
    double area = 0.5 * fabs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if (area < 1.0e-14) return;
    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };
    double sx = 0.0, sy = 0.0;
    for (int k = 0; k < 3; ++k) {
        double inv_m = 1.0 / fmax(mass_lumped[nodes[k]], 1.0e-14);
        sx += gx_num[nodes[k]] * inv_m;
        sy += gy_num[nodes[k]] * inv_m;
    }
    double w = sqrt(elem_s[e]) * area / 3.0;
    for (int i = 0; i < 3; ++i) {
        if (bc_tag[nodes[i]] == BC_OUTLET) continue;
        atomicAdd(&out[nodes[i]], -w * (dNdx[i] * sx + dNdy[i] * sy));
    }
}

/**
 * Assembled form of the same projection (2026-10-22): with w_K = sqrt(s_K) |K| / 3,
 *   G(x)_k = sum_{K ni k} w_K grad x|_K = (B x)_k,   B_kj = sum_{K ni k, j} w_K grad N_j|_K  (a 2-vector),
 *   and the subtracted term is sum_k B_ki . G_k / m_k, i.e. the operator is -B^T M_L^{-1} B.
 * B lives on the P1 pattern; B (rows k) and B^T (rows i, B^T_ik = B_ki) are both stored, so each
 * CG application is two row-wise passes without atomics. Assembled once per weight update.
 * oss_val = [Bx | By | BTx | BTy], each nnz long; inv_m = 1 / max(m, 1e-14) as in the matrix-free kernels.
 */
__global__ void k_assemble_oss_operator(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ elem_s,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    int nnz,
    double* __restrict__ oss_val
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;
    int nodes[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double x0 = x[nodes[0]], y0 = y[nodes[0]];
    double x1 = x[nodes[1]], y1 = y[nodes[1]];
    double x2 = x[nodes[2]], y2 = y[nodes[2]];
    double area = 0.5 * fabs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    if (area < 1.0e-14) return;
    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };
    double w = sqrt(elem_s[e]) * area / 3.0;
    double* Bx = oss_val;
    double* By = oss_val + nnz;
    double* BTx = oss_val + 2 * (size_t)nnz;
    double* BTy = oss_val + 3 * (size_t)nnz;
    for (int a = 0; a < 3; ++a) {
        const int k = nodes[a];
        for (int b = 0; b < 3; ++b) {
            const int j = nodes[b];
            int kj = find_csr_index(col_ind, row_ptr[k], row_ptr[k + 1], j);
            int jk = find_csr_index(col_ind, row_ptr[j], row_ptr[j + 1], k);
            if (kj < 0 || jk < 0) continue;
            atomicAdd(&Bx[kj],  w * dNdx[b]);
            atomicAdd(&By[kj],  w * dNdy[b]);
            atomicAdd(&BTx[jk], w * dNdx[b]);
            atomicAdd(&BTy[jk], w * dNdy[b]);
        }
    }
}

__global__ void k_inverse_lumped_mass(int n, const double* __restrict__ mass_lumped, double* __restrict__ inv_m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inv_m[i] = 1.0 / fmax(mass_lumped[i], 1.0e-14);
}

// Stage 1: (gx, gy)_k = M_L^{-1} (B x)_k
__global__ void k_oss_apply_B(int n, const int* __restrict__ row_ptr, const int* __restrict__ col_ind,
                              int nnz, const double* __restrict__ oss_val, const double* __restrict__ inv_m,
                              const double* __restrict__ in, double* __restrict__ gx, double* __restrict__ gy) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    double sx = 0.0, sy = 0.0;
    for (int a = row_ptr[k]; a < row_ptr[k + 1]; ++a) {
        double xj = in[col_ind[a]];
        sx += oss_val[a] * xj;
        sy += oss_val[nnz + a] * xj;
    }
    gx[k] = sx * inv_m[k];
    gy[k] = sy * inv_m[k];
}

// Stage 2: out_i -= (B^T g)_i, outlet (Dirichlet) rows untouched
__global__ void k_oss_apply_BT(int n, const int* __restrict__ row_ptr, const int* __restrict__ col_ind,
                               int nnz, const double* __restrict__ oss_val, const int* __restrict__ bc_tag,
                               const double* __restrict__ gx, const double* __restrict__ gy,
                               double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || bc_tag[i] == BC_OUTLET) return;
    const double* BTx = oss_val + 2 * (size_t)nnz;
    const double* BTy = oss_val + 3 * (size_t)nnz;
    double s = 0.0;
    for (int a = row_ptr[i]; a < row_ptr[i + 1]; ++a) {
        int k = col_ind[a];
        s += BTx[a] * gx[k] + BTy[a] * gy[k];
    }
    out[i] -= s;
}

/**
 * Lumped (row-sum) P1 mass vector: m_i = sum_K |K| / 3
 */
__global__ void k_assemble_lumped_mass(
    int num_elems,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ mass_lumped
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];

    double detJ = (x[n1] - x[n0]) * (y[n2] - y[n0]) - (x[n2] - x[n0]) * (y[n1] - y[n0]);
    double w = fabs(detJ) / 6.0;
    atomicAdd(&mass_lumped[n0], w);
    atomicAdd(&mass_lumped[n1], w);
    atomicAdd(&mass_lumped[n2], w);
}

/**
 * Gradient Recovery (Zienkiewicz-Zhu averaging) of the disturbance velocity
 *   u' = u_h - I_h u_P,  v' = v_h,   u_P(y) = 4 U_max y (H - y) / H^2 (inflow profile):
 * area-weighted accumulation of the piecewise-constant P1 gradient of u' onto the nodes.
 * Dividing by the lumped mass gives the recovered continuous gradient G'. Subtracting the
 * nodal interpolant of u_P BEFORE recovery makes undisturbed channel flow give exactly
 * zero, including at wall nodes, where one-sided averaging would otherwise leave a
 * spurious residual of order u_P'' h.
 *   grad_rec layout: [du'/dx | du'/dy | dv/dx | dv/dy], each block of length num_nodes
 */
__global__ void k_accumulate_recovered_gradient(
    int num_elems,
    int num_nodes,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double u_max,
    double H,
    double* __restrict__ grad_rec
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int n0 = elem_v[3 * e + 0];
    int n1 = elem_v[3 * e + 1];
    int n2 = elem_v[3 * e + 2];
    int nodes[3] = {n0, n1, n2};

    double x0 = x[n0], y0 = y[n0];
    double x1 = x[n1], y1 = y[n1];
    double x2 = x[n2], y2 = y[n2];

    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };

    double ud[3];
    for (int i = 0; i < 3; ++i) {
        double yi = y[nodes[i]];
        ud[i] = u[nodes[i]] - 4.0 * u_max * yi * (H - yi) / (H * H);
    }

    double g[4] = {
        ud[0] * dNdx[0] + ud[1] * dNdx[1] + ud[2] * dNdx[2],
        ud[0] * dNdy[0] + ud[1] * dNdy[1] + ud[2] * dNdy[2],
        v[n0] * dNdx[0] + v[n1] * dNdx[1] + v[n2] * dNdx[2],
        v[n0] * dNdy[0] + v[n1] * dNdy[1] + v[n2] * dNdy[2]
    };

    double w = area / 3.0;
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 4; ++c) {
            atomicAdd(&grad_rec[c * num_nodes + nodes[i]], w * g[c]);
        }
    }
}

/**
 * GPU Element Refinement Indicators for Adaptive Mesh Refinement (AMR), both in m/s.
 * Both criteria measure the flow relative to the undisturbed inflow (Poiseuille) profile
 *   u_P(y) = 4 U_max y (H - y) / H^2,   omega_P(y) = -du_P/dy,
 * so the parabolic channel flow, whose shear and curvature are the same at every x, does
 * not attract refinement. What remains is the disturbance created by the cylinder.
 *
 * elem_error[e]             disturbance vorticity times element size (eta ~ h),
 *     eta_v = h_K * max_vertices |omega_h - omega_P|,   h_K = sqrt(2 |K|):
 *   the velocity jump across the element induced by rotation and shear. Captures the
 *   boundary layer, the separated shear layers, the shed vortices and the wake.
 *
 * elem_error[num_elems + e] P1 interpolation error of the disturbance velocity (eta ~ h^2),
 *     eta_i = max_edges |((G'_u,j - G'_u,i) . e, (G'_v,j - G'_v,i) . e)| / 8,
 *   with G' the recovered gradient of the disturbance velocity (e^T H e ~ dG' . e):
 *   captures irrotational but strongly curved flow that vorticity cannot see, chiefly the
 *   stagnation region and the acceleration around the cylinder that set the surface pressure.
 *
 * Both use the recovered (area-averaged) gradient, which is smooth and free of
 * mesh-orientation patterns.
 */
__global__ void k_compute_amr_error_indicator(
    int num_elems,
    int num_nodes,
    const int* __restrict__ elem_v,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ grad_rec,
    const double* __restrict__ mass_lumped,
    double* __restrict__ elem_error
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int nodes[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };

    // Recovered disturbance gradient [du'/dx, du'/dy, dv/dx, dv/dy] at the vertices
    double G[3][4];
    for (int i = 0; i < 3; ++i) {
        double inv_m = 1.0 / fmax(mass_lumped[nodes[i]], 1.0e-14);
        for (int c = 0; c < 4; ++c) {
            G[i][c] = grad_rec[c * num_nodes + nodes[i]] * inv_m;
        }
    }

    double x0 = x[nodes[0]], y0 = y[nodes[0]];
    double area = 0.5 * fabs((x[nodes[1]] - x0) * (y[nodes[2]] - y0) -
                             (x[nodes[2]] - x0) * (y[nodes[1]] - y0));
    double omega = 0.0;
    double eta_i = 0.0;
    for (int i = 0; i < 3; ++i) {
        omega = fmax(omega, fabs(G[i][2] - G[i][1]));
        int j = (i + 1) % 3;
        double ex = x[nodes[j]] - x[nodes[i]];
        double ey = y[nodes[j]] - y[nodes[i]];
        double eu = (G[j][0] - G[i][0]) * ex + (G[j][1] - G[i][1]) * ey;
        double ev = (G[j][2] - G[i][2]) * ex + (G[j][3] - G[i][3]) * ey;
        eta_i = fmax(eta_i, 0.125 * sqrt(eu * eu + ev * ev));
    }
    elem_error[e]             = sqrt(2.0 * area) * omega;
    elem_error[num_elems + e] = eta_i;
}

/**
 * Volume-Integral (Variational) Drag & Lift Evaluation:
 *   out[0] = (du/dt, v_d) + (u.grad u, v_d) + nu (grad u, grad v_d) - (p/rho, div v_d),  v_d = (phi, 0)
 *   out[1] = same with v_l = (0, phi)
 * where phi is the P1 function equal to 1 on the cylinder nodes and 0 elsewhere, and du/dt is
 * the scheme's own discrete derivative b0 u^{n+1} + b1 u^n + b2 u^{n-1} (BDF2 or Euler). Only the
 * single layer of elements touching the cylinder contributes. Quadratic integrands are
 * integrated exactly with the edge-midpoint rule. C_D = -2 out[0] / (U^2 D), likewise C_L.
 */
__global__ void k_volume_forces(
    int num_elems,
    const int* __restrict__ elem_v,
    const int* __restrict__ bc_tag,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ u_old,
    const double* __restrict__ v_old,
    const double* __restrict__ u_prev,
    const double* __restrict__ v_prev,
    const double* __restrict__ p,
    double b0,
    double b1,
    double b2,
    double nu,
    double rho,
    double* __restrict__ out
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elems) return;

    int nodes[3] = { elem_v[3 * e + 0], elem_v[3 * e + 1], elem_v[3 * e + 2] };
    double phi[3];
    bool touches = false;
    for (int i = 0; i < 3; ++i) {
        phi[i] = (bc_tag[nodes[i]] == BC_CYLINDER) ? 1.0 : 0.0;
        touches = touches || (phi[i] > 0.0);
    }
    if (!touches) return;

    double x0 = x[nodes[0]], y0 = y[nodes[0]];
    double x1 = x[nodes[1]], y1 = y[nodes[1]];
    double x2 = x[nodes[2]], y2 = y[nodes[2]];
    double detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    double area = 0.5 * fabs(detJ);
    if (area < 1.0e-14) return;

    double inv2A = 1.0 / (2.0 * area);
    double dNdx[3] = { (y1 - y2) * inv2A, (y2 - y0) * inv2A, (y0 - y1) * inv2A };
    double dNdy[3] = { (x2 - x1) * inv2A, (x0 - x2) * inv2A, (x1 - x0) * inv2A };

    double ue[3], ve[3], ute[3], vte[3];
    double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
    double dphidx = 0.0, dphidy = 0.0, p_bar = 0.0;
    for (int i = 0; i < 3; ++i) {
        ue[i]  = u[nodes[i]];
        ve[i]  = v[nodes[i]];
        ute[i] = b0 * ue[i] + b1 * u_old[nodes[i]] + b2 * u_prev[nodes[i]];
        vte[i] = b0 * ve[i] + b1 * v_old[nodes[i]] + b2 * v_prev[nodes[i]];
        dudx += ue[i] * dNdx[i];  dudy += ue[i] * dNdy[i];
        dvdx += ve[i] * dNdx[i];  dvdy += ve[i] * dNdy[i];
        dphidx += phi[i] * dNdx[i];
        dphidy += phi[i] * dNdy[i];
        p_bar += p[nodes[i]] / 3.0;
    }

    double fx = 0.0, fy = 0.0;
    for (int i = 0; i < 3; ++i) {
        int j = (i + 1) % 3;
        double phi_m = 0.5 * (phi[i] + phi[j]);
        if (phi_m == 0.0) continue;
        double um = 0.5 * (ue[i] + ue[j]);
        double vm = 0.5 * (ve[i] + ve[j]);
        double conv_x = um * dudx + vm * dudy;
        double conv_y = um * dvdx + vm * dvdy;
        fx += (0.5 * (ute[i] + ute[j]) + conv_x) * phi_m;
        fy += (0.5 * (vte[i] + vte[j]) + conv_y) * phi_m;
    }
    fx *= area / 3.0;
    fy *= area / 3.0;
    fx += area * (nu * (dudx * dphidx + dudy * dphidy) - (p_bar / rho) * dphidx);
    fy += area * (nu * (dvdx * dphidx + dvdy * dphidy) - (p_bar / rho) * dphidy);

    atomicAdd(&out[0], fx);
    atomicAdd(&out[1], fy);
}

/**
 * Linear interpolation of two surface pressure probes along cylinder boundary edges:
 *   out[2] = p(front stagnation point), out[3] = p(rear point)
 */
__global__ void k_sample_pressure_probes(
    const double* __restrict__ p,
    int a0, int b0, double w0,
    int a1, int b1, double w1,
    double* __restrict__ out
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[2] = (1.0 - w0) * p[a0] + w0 * p[b0];
        out[3] = (1.0 - w1) * p[a1] + w1 * p[b1];
    }
}
