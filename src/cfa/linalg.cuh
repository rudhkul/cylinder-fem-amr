// GPU linear algebra: CSR kernels, PCG/BiCGSTAB, smoothed-aggregation AMG
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

// ============================================================================
// CUDA Linear Algebra Kernels (CSR SpMV, Dot Product, AXPY, Jacobi Precond)
// ============================================================================
__global__ void k_spmv_csr(
    int num_rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ val,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    double sum = 0.0;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        sum += val[idx] * x[col_ind[idx]];
    }
    y[row] = sum;
}

__global__ void k_extract_inv_diag(
    int num_rows,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ val,
    double* __restrict__ inv_diag
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    double d = 1.0;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];
    for (int idx = start; idx < end; ++idx) {
        if (col_ind[idx] == row) {
            d = val[idx];
            break;
        }
    }
    inv_diag[row] = (fabs(d) > 1.0e-14) ? (1.0 / d) : 1.0;
}

__global__ void k_vec_mul(
    int n,
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void k_axpy(
    int n,
    double alpha,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void k_xpay(
    int n,
    const double* __restrict__ x,
    double beta,
    double* __restrict__ y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + beta * y[i];
}

__global__ void k_bicgstab_p_update(
    int n,
    const double* __restrict__ r,
    const double* __restrict__ v,
    double beta,
    double omega,
    double* __restrict__ p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + beta * (p[i] - omega * v[i]);
}

// Dot product in two deterministic stages on the device: DOT_BLOCKS grid-stride partials, then one
// warp sums them in a fixed order; only the result goes to the host
constexpr int DOT_BLOCKS = 240;
__global__ void k_dot_grid(int n, const double* __restrict__ x, const double* __restrict__ y,
                           double* __restrict__ part) {
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += x[i] * y[i];
    __shared__ double sh[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) sh[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        double v = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) v += sh[w];
        part[blockIdx.x] = v;
    }
}
__global__ void k_dot_final(const double* __restrict__ part, double* __restrict__ out) {
    double v = 0.0;
    for (int b = threadIdx.x; b < DOT_BLOCKS; b += 32) v += part[b];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) *out = v;
}

// ============================================================================
// GPU Linear System Solvers (Jacobi-PCG & Jacobi-BiCGSTAB)
// ============================================================================
class GpuLinearSolver {
public:
    int max_n = 0;
    bool converged = true;   // Convergence flag of the most recent solve
    double last_rel_residual = 0.0;   // final residual relative to the convergence reference
    double *d_r = nullptr, *d_r0 = nullptr, *d_p = nullptr, *d_z = nullptr;
    double *d_q = nullptr, *d_s = nullptr, *d_t = nullptr, *d_v = nullptr;
    double *d_phat = nullptr, *d_shat = nullptr, *d_inv_diag = nullptr;
    double *d_block_sums = nullptr;   // DOT_BLOCKS partials + the result
    double *h_dot = nullptr;          // pinned

    // Grows geometrically so repeated AMR cycles do not reallocate on every remesh
    void allocate(int n) {
        if (n <= max_n) return;
        free_all();
        int cap = n + n / 2 + 256;
        max_n = cap;
        if (!h_dot) CUDA_CHECK(cudaMallocHost(&h_dot, sizeof(double)));

        CUDA_CHECK(cudaMalloc(&d_r,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r0,         cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_p,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_z,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_q,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_s,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_t,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_v,          cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_phat,       cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_shat,       cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inv_diag,   cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_sums, (DOT_BLOCKS + 1) * sizeof(double)));
    }

    void free_all() {
        if (d_r)          cudaFree(d_r);
        if (d_r0)         cudaFree(d_r0);
        if (d_p)          cudaFree(d_p);
        if (d_z)          cudaFree(d_z);
        if (d_q)          cudaFree(d_q);
        if (d_s)          cudaFree(d_s);
        if (d_t)          cudaFree(d_t);
        if (d_v)          cudaFree(d_v);
        if (d_phat)       cudaFree(d_phat);
        if (d_shat)       cudaFree(d_shat);
        if (d_inv_diag)   cudaFree(d_inv_diag);
        if (d_block_sums) cudaFree(d_block_sums);
        d_r = d_r0 = d_p = d_z = d_q = d_s = d_t = d_v = d_phat = d_shat = d_inv_diag = d_block_sums = nullptr;
        max_n = 0;
    }

    ~GpuLinearSolver() { free_all(); if (h_dot) cudaFreeHost(h_dot); }

    double dot(int n, const double* d_x, const double* d_y) {
        k_dot_grid<<<DOT_BLOCKS, 256>>>(n, d_x, d_y, d_block_sums);
        k_dot_final<<<1, 32>>>(d_block_sums, d_block_sums + DOT_BLOCKS);
        CUDA_CHECK(cudaMemcpy(h_dot, d_block_sums + DOT_BLOCKS, sizeof(double), cudaMemcpyDeviceToHost));
        return *h_dot;
    }

    /**
     * Jacobi-Preconditioned Conjugate Gradient (PCG) for Symmetric Pressure Poisson System.
     * d_x is used as the initial guess (warm start); convergence: ||r|| < tol * ||b||.
     */
    int solve_pcg(
        int n,
        const int* d_row_ptr,
        const int* d_col_ind,
        const double* d_val,
        const double* d_b,
        double* d_x,
        int max_iter = 400,
        double tol = 1.0e-7,
        const std::function<void(const double*, double*)>* extra_op = nullptr,
        const std::function<void(const double*, double*)>* precond = nullptr
    ) {
        allocate(n);
        converged = true;
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        // A x = CSR x (+ extra_op(x, y), which adds a matrix-free symmetric part into y)
        auto apply = [&](const double* in, double* out) {
            k_spmv_csr<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, in, out);
            if (extra_op) (*extra_op)(in, out);
        };

        double b_norm = std::sqrt(dot(n, d_b, d_b));
        if (b_norm < 1.0e-14) {
            CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
            return 0;
        }

        k_extract_inv_diag<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_inv_diag);
        // r = b - A * x
        apply(d_x, d_q);
        CUDA_CHECK(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        k_axpy<<<blocks, threads>>>(n, -1.0, d_q, d_r);

        double r_norm = std::sqrt(dot(n, d_r, d_r));
        if (r_norm < tol * b_norm) return 0;

        // z = M^{-1} r (AMG V-cycle if given, else Jacobi), p = z
        auto apply_prec = [&]() {
            if (precond) (*precond)(d_r, d_z);
            else k_vec_mul<<<blocks, threads>>>(n, d_inv_diag, d_r, d_z);
        };
        apply_prec();
        CUDA_CHECK(cudaMemcpy(d_p, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double rz_old = dot(n, d_r, d_z);

        int iter = 0;
        bool done = false;
        for (; iter < max_iter; ++iter) {
            apply(d_p, d_q);
            double pq = dot(n, d_p, d_q);
            if (std::fabs(pq) < 1.0e-30) break;

            double alpha = rz_old / pq;
            k_axpy<<<blocks, threads>>>(n,  alpha, d_p, d_x);
            k_axpy<<<blocks, threads>>>(n, -alpha, d_q, d_r);

            r_norm = std::sqrt(dot(n, d_r, d_r));
            last_rel_residual = r_norm / b_norm;
            if (r_norm < tol * b_norm) {
                ++iter;
                done = true;
                break;
            }

            apply_prec();
            double rz_new = dot(n, d_r, d_z);
            double beta   = rz_new / (rz_old + 1.0e-30);
            k_xpay<<<blocks, threads>>>(n, d_z, beta, d_p);
            rz_old = rz_new;
        }
        converged = done;
        return iter;
    }

    /**
     * Jacobi-Preconditioned BiCGSTAB for Non-Symmetric Advection-Diffusion-SUPG Momentum System
     */
    int solve_bicgstab(
        int n,
        const int* d_row_ptr,
        const int* d_col_ind,
        const double* d_val,
        const double* d_b,
        double* d_x,
        int max_iter = 200,
        double tol = 1.0e-7
    ) {
        allocate(n);
        converged = true;
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;

        k_extract_inv_diag<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_inv_diag);
        k_spmv_csr<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_x, d_v);
        CUDA_CHECK(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        k_axpy<<<blocks, threads>>>(n, -1.0, d_v, d_r);
        CUDA_CHECK(cudaMemcpy(d_r0, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_p,  d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_v, 0, n * sizeof(double)));

        double r_norm0 = std::sqrt(dot(n, d_r, d_r));
        if (r_norm0 < 1.0e-13) return 0;

        double rho_old = 1.0, alpha = 1.0, omega = 1.0;
        // Breakdown thresholds relative to the initial residual: with small flows (2D-3 near
        // t = 0, transverse velocity of a symmetric flow) absolute ones stopped converging solves
        const double breakdown = 1.0e-30 * r_norm0 * r_norm0 + 1.0e-300;

        int iter = 0;
        bool done = false;
        for (; iter < max_iter; ++iter) {
            double rho_new = dot(n, d_r0, d_r);
            if (std::fabs(rho_new) < breakdown) break;

            if (iter > 0) {
                double beta = (rho_new / rho_old) * (alpha / (omega + 1.0e-30));
                k_bicgstab_p_update<<<blocks, threads>>>(n, d_r, d_v, beta, omega, d_p);
            }

            // phat = M^{-1} p, v = A * phat
            k_vec_mul<<<blocks, threads>>>(n, d_inv_diag, d_p, d_phat);
            k_spmv_csr<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_phat, d_v);

            double r0_v = dot(n, d_r0, d_v);
            if (std::fabs(r0_v) < breakdown) break;
            alpha = rho_new / r0_v;

            // s = r - alpha * v
            CUDA_CHECK(cudaMemcpy(d_s, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
            k_axpy<<<blocks, threads>>>(n, -alpha, d_v, d_s);

            double s_norm = std::sqrt(dot(n, d_s, d_s));
            if (s_norm < tol * r_norm0 || s_norm < 1.0e-11) {
                k_axpy<<<blocks, threads>>>(n, alpha, d_phat, d_x);
                ++iter;
                done = true;
                break;
            }

            // shat = M^{-1} s, t = A * shat
            k_vec_mul<<<blocks, threads>>>(n, d_inv_diag, d_s, d_shat);
            k_spmv_csr<<<blocks, threads>>>(n, d_row_ptr, d_col_ind, d_val, d_shat, d_t);

            double tt = dot(n, d_t, d_t);
            omega = (tt > breakdown) ? (dot(n, d_t, d_s) / tt) : 0.0;

            k_axpy<<<blocks, threads>>>(n, alpha, d_phat, d_x);
            k_axpy<<<blocks, threads>>>(n, omega, d_shat, d_x);

            CUDA_CHECK(cudaMemcpy(d_r, d_s, n * sizeof(double), cudaMemcpyDeviceToDevice));
            k_axpy<<<blocks, threads>>>(n, -omega, d_t, d_r);

            double r_norm = std::sqrt(dot(n, d_r, d_r));
            last_rel_residual = r_norm / r_norm0;
            if (r_norm < tol * r_norm0 || r_norm < 1.0e-11) {
                ++iter;
                done = true;
                break;
            }
            rho_old = rho_new;
        }
        converged = done;
        return iter;
    }
};

// ============================================================================
// Multicolour ILU(0) Preconditioner and Two-Component BiCGSTAB for the Momentum System
// ============================================================================
// The nodes are coloured greedily (no two neighbours share a colour) and ILU(0) is taken in the
// order colour by colour. Rows of one colour do not couple, so the factorization and both
// triangular solves run one kernel per colour, every row of the colour in parallel. Deterministic
// (no atomics). Offline on dumped A3 systems (--dump-momentum): BiCGSTAB 72-78 iterations with
// Jacobi, 26-30 with this (6 colours); the natural-order ILU(0) gives 26-28 but has ~3800 levels.
// The factors are stored split, L and U rows in colour order, so a sweep reads only its own entries.
// u and v share the matrix, so both are solved together: every kernel covers both components
// (vectors [u | v], length 2n) and the dot products of both go to the host in one transfer.
constexpr int ILU_MAX_ROW = 32;   // P1 rows on NVB meshes have at most 9 entries

// Row i: for its L entries k in elimination order (colour by colour; entries of one colour do not
// couple), l_ik = a_ik / u_kk and a_ij -= l_ik u_kj over the U part of row k within the pattern.
// Row i is kept in registers/local memory; rows of earlier colours are final.
__global__ void k_ilu0_factor_colour(int n_rows, const int* __restrict__ rows,
                                     const int* __restrict__ row_ptr, const int* __restrict__ col_ind,
                                     const int* __restrict__ colour, const int* __restrict__ diag_pos,
                                     double* __restrict__ lu) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_rows) return;
    const int i = rows[t];
    const int s = row_ptr[i], len = row_ptr[i + 1] - s;
    const int ci = colour[i];
    int cols[ILU_MAX_ROW], ccol[ILU_MAX_ROW];
    double a[ILU_MAX_ROW];
    for (int q = 0; q < len; ++q) { cols[q] = col_ind[s + q]; ccol[q] = colour[cols[q]]; a[q] = lu[s + q]; }
    for (int c = 0; c < ci; ++c) {
        for (int q = 0; q < len; ++q) {
            if (ccol[q] != c) continue;
            const int k = cols[q];
            const double lik = a[q] / lu[diag_pos[k]];
            a[q] = lik;
            for (int m = row_ptr[k]; m < row_ptr[k + 1]; ++m) {
                const int j = col_ind[m];
                for (int b = 0; b < len; ++b)
                    if (cols[b] == j) { if (ccol[b] > c) a[b] -= lik * lu[m]; break; }
            }
        }
    }
    for (int q = 0; q < len; ++q) lu[s + q] = a[q];
}

// Split factors: lv[e] = lu[l_src[e]], uv[e] = lu[u_src[e]], dinv[t] = 1 / lu[d_src[t]]
__global__ void k_ilu0_gather(int nl, int nu, int n, const int* __restrict__ l_src, const int* __restrict__ u_src,
                              const int* __restrict__ d_src, const double* __restrict__ lu,
                              double* __restrict__ lv, double* __restrict__ uv, double* __restrict__ dinv) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < nl) lv[e] = lu[l_src[e]];
    if (e < nu) uv[e] = lu[u_src[e]];
    if (e < n)  dinv[e] = 1.0 / lu[d_src[e]];
}

// Forward solve L y = r (unit lower) for the rows of one colour; t covers nrhs * m (vectors of stride n).
// With last_colour the backward step of that colour (no U entries) is done too: z = y / u_ii.
__global__ void k_ilu0_lower_colour(int m, int nrhs, int n, const int* __restrict__ rows,
                                    const int* __restrict__ lp, const int* __restrict__ lc,
                                    const double* __restrict__ lv, const double* __restrict__ dinv,
                                    bool last_colour, const double* __restrict__ r, double* __restrict__ z) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrhs * m) return;
    const int comp = t / m, tr = t - comp * m;
    const size_t off = (size_t)comp * n;
    const int i = rows[tr];
    double acc = r[off + i];
    for (int e = lp[tr]; e < lp[tr + 1]; ++e) acc -= lv[e] * z[off + lc[e]];
    z[off + i] = last_colour ? acc * dinv[tr] : acc;
}

// Backward solve U z = y in place for the rows of one colour
__global__ void k_ilu0_upper_colour(int m, int nrhs, int n, const int* __restrict__ rows,
                                    const int* __restrict__ up, const int* __restrict__ uc,
                                    const double* __restrict__ uv, const double* __restrict__ dinv,
                                    double* __restrict__ z) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrhs * m) return;
    const int comp = t / m, tr = t - comp * m;
    const size_t off = (size_t)comp * n;
    const int i = rows[tr];
    double acc = z[off + i];
    for (int e = up[tr]; e < up[tr + 1]; ++e) acc -= uv[e] * z[off + uc[e]];
    z[off + i] = acc * dinv[tr];
}

class MulticolourIlu0 {
public:
    int n = 0, nnz = 0, n_colours = 0, nl = 0, nu = 0;
    double colour_ms = 0.0;               // host colouring, split pattern and upload, all pattern setups
    std::vector<int> colour_start;        // rows of colour c: positions colour_start[c] .. colour_start[c+1]
    // CSR-shaped working copy for the factorization
    int *d_rows = nullptr, *d_colour = nullptr, *d_diag_pos = nullptr;
    double* d_lu = nullptr;
    const int *d_row_ptr = nullptr, *d_col_ind = nullptr;
    // Split factors in colour-row order (position t = row rows[t])
    int *d_lp = nullptr, *d_up = nullptr, *d_lc = nullptr, *d_uc = nullptr;
    int *d_l_src = nullptr, *d_u_src = nullptr, *d_d_src = nullptr;   // positions in the CSR copy
    double *d_lv = nullptr, *d_uv = nullptr, *d_dinv = nullptr;
    size_t cap_n = 0, cap_nnz = 0;

    MulticolourIlu0() = default;
    MulticolourIlu0(const MulticolourIlu0&) = delete;
    MulticolourIlu0& operator=(const MulticolourIlu0&) = delete;
    ~MulticolourIlu0() { device_free_all(); }

    // New sparsity pattern (after every remesh): greedy colouring in node order on the host
    void set_pattern(int n_, const std::vector<int>& rp, const std::vector<int>& ci,
                     const int* d_rp, const int* d_ci) {
        auto t0 = std::chrono::steady_clock::now();
        n = n_;  nnz = rp[n];  d_row_ptr = d_rp;  d_col_ind = d_ci;
        std::vector<int> colour(n, -1), diag(n, -1);
        std::vector<int> mark;   // mark[c] == i: colour c is taken by a neighbour of i
        n_colours = 0;
        for (int i = 0; i < n; ++i) {
            if (rp[i + 1] - rp[i] > ILU_MAX_ROW) {
                fprintf(stderr, "[ILU] row %d has %d entries (limit %d)\n", i, rp[i + 1] - rp[i], ILU_MAX_ROW);
                exit(EXIT_FAILURE);
            }
            for (int a = rp[i]; a < rp[i + 1]; ++a) {
                int c = colour[ci[a]];
                if (c >= 0) mark[c] = i;
                if (ci[a] == i) diag[i] = a;
            }
            int c = 0;
            while (c < n_colours && mark[c] == i) ++c;
            if (c == n_colours) { ++n_colours; mark.push_back(-1); }
            colour[i] = c;
        }
        colour_start.assign(n_colours + 1, 0);
        for (int i = 0; i < n; ++i) colour_start[colour[i] + 1]++;
        for (int c = 0; c < n_colours; ++c) colour_start[c + 1] += colour_start[c];
        std::vector<int> rows(n), fill(colour_start.begin(), colour_start.end() - 1);
        for (int i = 0; i < n; ++i) rows[fill[colour[i]]++] = i;

        // Split pattern: counts, then fill (rows in colour order)
        std::vector<int> lp(n + 1, 0), up(n + 1, 0);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < n; ++t) {
            const int i = rows[t];
            int l = 0, u = 0;
            for (int a = rp[i]; a < rp[i + 1]; ++a) {
                if (ci[a] == i) continue;
                if (colour[ci[a]] < colour[i]) ++l; else ++u;
            }
            lp[t + 1] = l;  up[t + 1] = u;
        }
        for (int t = 0; t < n; ++t) { lp[t + 1] += lp[t];  up[t + 1] += up[t]; }
        nl = lp[n];  nu = up[n];
        std::vector<int> lc(nl), uc(nu), l_src(nl), u_src(nu), d_src(n);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < n; ++t) {
            const int i = rows[t];
            int l = lp[t], u = up[t];
            for (int a = rp[i]; a < rp[i + 1]; ++a) {
                const int j = ci[a];
                if (j == i) d_src[t] = a;
                else if (colour[j] < colour[i]) { lc[l] = j;  l_src[l++] = a; }
                else                            { uc[u] = j;  u_src[u++] = a; }
            }
        }

        if ((size_t)n > cap_n) {
            cap_n = n + n / 2 + 64;
            for (int** p : {&d_rows, &d_colour, &d_diag_pos, &d_lp, &d_up, &d_d_src}) realloc_dev(*p, cap_n + 1);
            realloc_dev(d_dinv, cap_n);
        }
        if ((size_t)nnz > cap_nnz) {
            cap_nnz = nnz + nnz / 2 + 256;
            for (int** p : {&d_lc, &d_uc, &d_l_src, &d_u_src}) realloc_dev(*p, cap_nnz);
            realloc_dev(d_lu, cap_nnz);  realloc_dev(d_lv, cap_nnz);  realloc_dev(d_uv, cap_nnz);
        }
        auto up_int = [](int* d, const std::vector<int>& h) {
            CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
        };
        up_int(d_rows, rows);  up_int(d_colour, colour);  up_int(d_diag_pos, diag);
        up_int(d_lp, lp);  up_int(d_up, up);  up_int(d_lc, lc);  up_int(d_uc, uc);
        up_int(d_l_src, l_src);  up_int(d_u_src, u_src);  up_int(d_d_src, d_src);
        colour_ms += ms_since(t0);
    }

    // Factor the current matrix values (once per time step; u and v share the matrix)
    void factor(const double* d_val) {
        CUDA_CHECK(cudaMemcpy(d_lu, d_val, nnz * sizeof(double), cudaMemcpyDeviceToDevice));
        for (int c = 1; c < n_colours; ++c) {   // colour 0 has no L entries
            int m = colour_start[c + 1] - colour_start[c];
            k_ilu0_factor_colour<<<(m + 127) / 128, 128>>>(m, d_rows + colour_start[c], d_row_ptr, d_col_ind,
                                                           d_colour, d_diag_pos, d_lu);
        }
        int len = std::max(n, std::max(nl, nu));
        k_ilu0_gather<<<(len + 255) / 256, 256>>>(nl, nu, n, d_l_src, d_u_src, d_d_src, d_lu, d_lv, d_uv, d_dinv);
        CUDA_CHECK_LAUNCH();
    }

    // z = (LU)^{-1} r for nrhs vectors of length n stored one after another
    void apply(const double* r, double* z, int nrhs = 1) const {
        for (int c = 0; c < n_colours; ++c) {
            int m = colour_start[c + 1] - colour_start[c], cs = colour_start[c];
            k_ilu0_lower_colour<<<(nrhs * m + 255) / 256, 256>>>(m, nrhs, n, d_rows + cs, d_lp + cs, d_lc, d_lv,
                                                                d_dinv + cs, c == n_colours - 1, r, z);
        }
        for (int c = n_colours - 2; c >= 0; --c) {
            int m = colour_start[c + 1] - colour_start[c], cs = colour_start[c];
            k_ilu0_upper_colour<<<(nrhs * m + 255) / 256, 256>>>(m, nrhs, n, d_rows + cs, d_up + cs, d_uc, d_uv,
                                                                d_dinv + cs, z);
        }
    }

private:
    template <typename T> static void realloc_dev(T*& p, size_t count) {
        if (p) CUDA_CHECK(cudaFree(p));
        CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    }
    void device_free_all() {
        for (void* p : {(void*)d_rows, (void*)d_colour, (void*)d_diag_pos, (void*)d_lu, (void*)d_lp, (void*)d_up,
                        (void*)d_lc, (void*)d_uc, (void*)d_l_src, (void*)d_u_src, (void*)d_lv, (void*)d_uv,
                        (void*)d_dinv, (void*)d_d_src})
            if (p) cudaFree(p);
    }
};

// ---- Two-component BiCGSTAB (u and v with the same matrix) ----
// SpMV for nrhs vectors of stride n (thread per row and component)
__global__ void k_spmv_csr_multi(int n, int nrhs, const int* __restrict__ row_ptr, const int* __restrict__ col_ind,
                                 const double* __restrict__ val, const double* __restrict__ x, double* __restrict__ y) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nrhs * n) return;
    const int comp = t / n, row = t - comp * n;
    const double* xc = x + (size_t)comp * n;
    double sum = 0.0;
    for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) sum += val[idx] * xc[col_ind[idx]];
    y[t] = sum;
}

// Up to three dot products for each of two components in one pass: block partials
// part[block * 6 + 2 q + comp] = sum over the block's indices of a_q * b_q (component comp)
struct DotSet { const double* a[3]; const double* b[3]; int nq; };
constexpr int MDOT_BLOCKS = 240, MDOT_THREADS = 256;
__global__ void k_multidot2_partial(int n, DotSet d, double* __restrict__ part) {
    double acc[6] = {0, 0, 0, 0, 0, 0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        for (int q = 0; q < d.nq; ++q) {
            acc[2 * q]     += d.a[q][i] * d.b[q][i];
            acc[2 * q + 1] += d.a[q][n + i] * d.b[q][n + i];
        }
    }
    __shared__ double sh[6][MDOT_THREADS / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int k = 0; k < 2 * d.nq; ++k) {
        double v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 2 * d.nq) {
        double v = 0.0;
        for (int w = 0; w < MDOT_THREADS / 32; ++w) v += sh[threadIdx.x][w];
        part[blockIdx.x * 6 + threadIdx.x] = v;
    }
}
// Sums the block partials in a fixed order (deterministic) into out[0 .. 2 nq): one warp per result
__global__ void k_multidot2_final(const double* __restrict__ part, double* __restrict__ out) {
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    double v = 0.0;
    for (int b = lane; b < MDOT_BLOCKS; b += 32) v += part[b * 6 + k];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) out[k] = v;
}

// Per-component coefficients for the vector updates (index i of the 2n-vector: component i >= n)
struct Coef2 { double c[2]; };
__global__ void k_bicg2_p(int n, const double* __restrict__ r, const double* __restrict__ v,
                          Coef2 beta, Coef2 omega, double* __restrict__ p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    const int c = i >= n;
    p[i] = r[i] + beta.c[c] * (p[i] - omega.c[c] * v[i]);
}
__global__ void k_bicg2_s(int n, const double* __restrict__ r, const double* __restrict__ v, Coef2 alpha,
                          double* __restrict__ s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    s[i] = r[i] - alpha.c[i >= n] * v[i];
}
__global__ void k_bicg2_xr(int n, const double* __restrict__ phat, const double* __restrict__ shat,
                           const double* __restrict__ s, const double* __restrict__ t,
                           Coef2 a1, Coef2 a2, Coef2 omega, double* __restrict__ x, double* __restrict__ r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n) return;
    const int c = i >= n;
    x[i] += a1.c[c] * phat[i] + a2.c[c] * shat[i];
    r[i] = s[i] - omega.c[c] * t[i];
}
class GpuBicgstab2 {
public:
    int iters[2] = {0, 0};
    bool converged[2] = {true, true};
    double last_rel_residual[2] = {0.0, 0.0};

    GpuBicgstab2() = default;
    GpuBicgstab2(const GpuBicgstab2&) = delete;
    GpuBicgstab2& operator=(const GpuBicgstab2&) = delete;
    ~GpuBicgstab2() {
        for (double* p : {d_x, d_b, d_r, d_r0, d_p, d_v, d_s, d_t, d_phat, d_shat, d_part}) if (p) cudaFree(p);
        if (h_out) cudaFreeHost(h_out);
    }

    // Solves A x_c = b_c for c = u, v (x used as initial guess); same algorithm, breakdown guards and
    // stopping rule per component as GpuLinearSolver::solve_bicgstab: |r| < tol |r0| or |r| < 1e-11.
    // A component that has stopped keeps its x (its coefficients are zero) while the other iterates.
    // Preconditioned by the factored multicolour ILU(0).
    void solve(int n, const int* d_row_ptr, const int* d_col_ind, const double* d_val,
               const double* b_u, const double* b_v, double* x_u, double* x_v,
               int max_iter, double tol, const MulticolourIlu0& ilu) {
        allocate(n);
        const int nb2 = (2 * n + 255) / 256;
        const size_t bytes = n * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_x,     x_u, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_x + n, x_v, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,     b_u, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_b + n, b_v, bytes, cudaMemcpyDeviceToDevice));
        auto spmv = [&](const double* in, double* out) {
            k_spmv_csr_multi<<<nb2, 256>>>(n, 2, d_row_ptr, d_col_ind, d_val, in, out);
        };
        auto prec = [&](const double* in, double* out) { ilu.apply(in, out, 2); };
        auto dots = [&](DotSet d) -> const double* {
            k_multidot2_partial<<<MDOT_BLOCKS, MDOT_THREADS>>>(n, d, d_part);
            k_multidot2_final<<<1, 64 * d.nq>>>(d_part, d_out);
            CUDA_CHECK(cudaMemcpy(h_out, d_out, 2 * d.nq * sizeof(double), cudaMemcpyDeviceToHost));
            return h_out;
        };

        // r = b - A x, r0 = p = r, v = 0
        spmv(d_x, d_v);
        k_bicg2_s<<<nb2, 256>>>(n, d_b, d_v, Coef2{{1.0, 1.0}}, d_r);
        CUDA_CHECK(cudaMemcpy(d_r0, d_r, 2 * bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_p,  d_r, 2 * bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_v, 0, 2 * bytes));

        const double* o = dots(DotSet{{d_r}, {d_r}, 1});
        double rn0[2], bd[2], rho[2], rho_old[2] = {1.0, 1.0}, alpha[2] = {1.0, 1.0}, omega[2] = {1.0, 1.0};
        bool active[2];
        for (int c = 0; c < 2; ++c) {
            rn0[c] = std::sqrt(o[c]);
            rho[c] = o[c];   // r0 . r with r0 = r
            bd[c] = 1.0e-30 * rn0[c] * rn0[c] + 1.0e-300;
            iters[c] = 0;  converged[c] = true;  last_rel_residual[c] = 0.0;
            active[c] = rn0[c] >= 1.0e-13;
        }
        for (int iter = 0; iter < max_iter && (active[0] || active[1]); ++iter) {
            Coef2 cb{{0.0, 0.0}}, co{{0.0, 0.0}};
            for (int c = 0; c < 2; ++c) {
                if (active[c] && std::fabs(rho[c]) < bd[c]) stop(c, iter, false, active);
                if (active[c] && iter > 0) {
                    cb.c[c] = (rho[c] / rho_old[c]) * (alpha[c] / (omega[c] + 1.0e-30));
                    co.c[c] = omega[c];
                }
            }
            if (!(active[0] || active[1])) break;
            if (iter > 0) k_bicg2_p<<<nb2, 256>>>(n, d_r, d_v, cb, co, d_p);

            // phat = M^{-1} p, v = A phat
            prec(d_p, d_phat);
            spmv(d_phat, d_v);
            o = dots(DotSet{{d_r0}, {d_v}, 1});
            Coef2 ca{{0.0, 0.0}};
            for (int c = 0; c < 2; ++c) {
                if (active[c] && std::fabs(o[c]) < bd[c]) stop(c, iter, false, active);
                if (active[c]) { alpha[c] = rho[c] / o[c];  ca.c[c] = alpha[c]; }
            }
            if (!(active[0] || active[1])) break;
            k_bicg2_s<<<nb2, 256>>>(n, d_r, d_v, ca, d_s);

            // shat = M^{-1} s, t = A shat (also when s has converged: then they are not used)
            prec(d_s, d_shat);
            spmv(d_shat, d_t);
            o = dots(DotSet{{d_s, d_t, d_t}, {d_s, d_t, d_s}, 3});
            Coef2 a1{{0.0, 0.0}}, a2{{0.0, 0.0}}, cw{{0.0, 0.0}};
            bool s_done[2] = {false, false};
            for (int c = 0; c < 2; ++c) {
                if (!active[c]) continue;
                a1.c[c] = alpha[c];
                const double s_norm = std::sqrt(o[c]);
                if (s_norm < tol * rn0[c] || s_norm < 1.0e-11) { s_done[c] = true; continue; }
                const double tt = o[2 + c];
                omega[c] = (tt > bd[c]) ? (o[4 + c] / tt) : 0.0;
                a2.c[c] = omega[c];  cw.c[c] = omega[c];
            }
            k_bicg2_xr<<<nb2, 256>>>(n, d_phat, d_shat, d_s, d_t, a1, a2, cw, d_x, d_r);
            for (int c = 0; c < 2; ++c) if (s_done[c]) stop(c, iter + 1, true, active);
            if (!(active[0] || active[1])) break;

            o = dots(DotSet{{d_r, d_r0}, {d_r, d_r}, 2});
            for (int c = 0; c < 2; ++c) {
                if (!active[c]) continue;
                const double r_norm = std::sqrt(o[c]);
                last_rel_residual[c] = r_norm / rn0[c];
                if (r_norm < tol * rn0[c] || r_norm < 1.0e-11) { stop(c, iter + 1, true, active); continue; }
                rho_old[c] = rho[c];
                rho[c] = o[2 + c];
                if (iter + 1 == max_iter) stop(c, max_iter, false, active);
            }
        }
        CUDA_CHECK(cudaMemcpy(x_u, d_x,     bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(x_v, d_x + n, bytes, cudaMemcpyDeviceToDevice));
    }

private:
    int cap = 0;
    double *d_x = nullptr, *d_b = nullptr, *d_r = nullptr, *d_r0 = nullptr, *d_p = nullptr, *d_v = nullptr;
    double *d_s = nullptr, *d_t = nullptr, *d_phat = nullptr, *d_shat = nullptr;
    double *d_part = nullptr, *d_out = nullptr;   // dot partials; d_out = d_part + 6 MDOT_BLOCKS
    double* h_out = nullptr;                      // pinned copy of the dot results

    void stop(int c, int it, bool ok, bool* active) {
        active[c] = false;  iters[c] = it;  converged[c] = ok;
    }
    void allocate(int n) {
        if (!h_out) {
            CUDA_CHECK(cudaMallocHost(&h_out, 8 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_part, (6 * MDOT_BLOCKS + 8) * sizeof(double)));   // block partials + results
            d_out = d_part + 6 * MDOT_BLOCKS;
        }
        if (n <= cap) return;
        cap = n + n / 2 + 256;
        for (double** p : {&d_x, &d_b, &d_r, &d_r0, &d_p, &d_v, &d_s, &d_t, &d_phat, &d_shat}) {
            if (*p) CUDA_CHECK(cudaFree(*p));
            CUDA_CHECK(cudaMalloc(p, 2 * (size_t)cap * sizeof(double)));
        }
    }
};

// ============================================================================
// Smoothed-Aggregation Algebraic Multigrid (SA-AMG) Preconditioner for the PPE
// ============================================================================
// Setup on the host (once per remesh / every few steps), V-cycle on the GPU.
//   - Strength of connection |a_ij| >= theta sqrt(|a_ii a_jj|), greedy 3-phase aggregation
//   - Tentative prolongator from the constant near-null space, smoothed by one damped
//     Jacobi step: P = (I - omega D^{-1} A) P0, omega = 4 / (3 rho(D^{-1} A))
//   - Galerkin coarse operators A_c = P^T A P, coarsest level solved with a dense inverse
//   - Symmetric V(nu, nu) cycle with damped Jacobi smoothing, so the preconditioner is SPD
//     and can be used inside CG. Dirichlet (identity) rows become singleton aggregates.
struct HostCsr {
    int rows = 0, cols = 0;
    std::vector<int> rp, ci;
    std::vector<double> v;
};

// C = A * B (row-wise sparse accumulation). Rows are independent and each entry is summed in
// the same order as a serial product, so the threaded result is bitwise reproducible.
static HostCsr csr_multiply(const HostCsr& A, const HostCsr& B) {
    HostCsr C;
    C.rows = A.rows;  C.cols = B.cols;
    C.rp.assign(A.rows + 1, 0);
    #pragma omp parallel
    {
        std::vector<int> marker(B.cols, -1);
        #pragma omp for schedule(static)
        for (int i = 0; i < A.rows; ++i) {   // symbolic: row lengths
            int count = 0;
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) {
                int k = A.ci[a];
                for (int b = B.rp[k]; b < B.rp[k + 1]; ++b) {
                    int j = B.ci[b];
                    if (marker[j] != i) { marker[j] = i; ++count; }
                }
            }
            C.rp[i + 1] = count;
        }
    }
    for (int i = 0; i < A.rows; ++i) C.rp[i + 1] += C.rp[i];
    C.ci.resize(C.rp[A.rows]);
    C.v.resize(C.rp[A.rows]);
    #pragma omp parallel
    {
        std::vector<int> marker(B.cols, -1);
        std::vector<double> acc(B.cols, 0.0);
        std::vector<int> cols;
        #pragma omp for schedule(static)
        for (int i = 0; i < A.rows; ++i) {   // numeric
            cols.clear();
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) {
                int k = A.ci[a];
                double av = A.v[a];
                for (int b = B.rp[k]; b < B.rp[k + 1]; ++b) {
                    int j = B.ci[b];
                    if (marker[j] != i) { marker[j] = i; acc[j] = 0.0; cols.push_back(j); }
                    acc[j] += av * B.v[b];
                }
            }
            std::sort(cols.begin(), cols.end());
            int out = C.rp[i];
            for (int j : cols) { C.ci[out] = j; C.v[out] = acc[j]; ++out; }
        }
    }
    return C;
}

static HostCsr csr_transpose(const HostCsr& A) {
    HostCsr T;
    T.rows = A.cols;  T.cols = A.rows;
    T.rp.assign(A.cols + 1, 0);
    for (int c : A.ci) T.rp[c + 1]++;
    for (int i = 0; i < A.cols; ++i) T.rp[i + 1] += T.rp[i];
    T.ci.resize(A.ci.size());  T.v.resize(A.v.size());
    std::vector<int> fill(T.rp.begin(), T.rp.end() - 1);
    for (int i = 0; i < A.rows; ++i) {
        for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) {
            int k = fill[A.ci[a]]++;
            T.ci[k] = i;  T.v[k] = A.v[a];
        }
    }
    return T;
}

__global__ void k_jacobi_init(int n, double omega, const double* __restrict__ dinv,
                              const double* __restrict__ b, double* __restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = omega * dinv[i] * b[i];
}

// dinv = 1 / diag(A), same rule as the host setup (1 for a zero diagonal)
__global__ void k_inverse_diagonal(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                                   const double* __restrict__ v, double* __restrict__ dinv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double d = 0.0;
    for (int a = rp[i]; a < rp[i + 1]; ++a) if (ci[a] == i) d = v[a];
    dinv[i] = (fabs(d) > 1.0e-300) ? 1.0 / d : 1.0;
}

// V-cycle operators with the SpMV fused in, TPR threads per row (the AMG levels range from 3 to
// ~90 entries per row; one thread per row made the small dense-ish coarse levels latency-bound):
//   CSR_SPMV y = A x,  CSR_ADD y += A x,  CSR_RESID y = b - A x,
//   CSR_JACOBI y = x + omega D^{-1} (b - A x)  (y must not alias x)
enum CsrOp { CSR_SPMV, CSR_ADD, CSR_RESID, CSR_JACOBI };
template <int TPR, int OP>
__global__ void k_csr_op(int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                         const double* __restrict__ val, const double* __restrict__ x,
                         const double* __restrict__ b, const double* __restrict__ dinv, double omega,
                         double* __restrict__ y) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = g / TPR, lane = g % TPR;
    double s = 0.0;
    if (row < rows)
        for (int k = rp[row] + lane; k < rp[row + 1]; k += TPR) s += val[k] * x[ci[k]];
    for (int o = TPR / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffff, s, o, TPR);
    if (row >= rows || lane != 0) return;
    if (OP == CSR_SPMV)        y[row] = s;
    else if (OP == CSR_ADD)    y[row] += s;
    else if (OP == CSR_RESID)  y[row] = b[row] - s;
    else                       y[row] = x[row] + omega * dinv[row] * (b[row] - s);
}

template <int OP>
static void csr_op_launch(int tpr, int rows, const int* rp, const int* ci, const double* val,
                          const double* x, const double* b, const double* dinv, double omega, double* y) {
    const int threads = 256;
    const int blocks = (int)(((long long)rows * tpr + threads - 1) / threads);
    switch (tpr) {
        case 1:  k_csr_op<1,  OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
        case 2:  k_csr_op<2,  OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
        case 4:  k_csr_op<4,  OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
        case 8:  k_csr_op<8,  OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
        case 16: k_csr_op<16, OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
        default: k_csr_op<32, OP><<<blocks, threads>>>(rows, rp, ci, val, x, b, dinv, omega, y); break;
    }
}

// Threads per row from the mean row length (measured on the A3 hierarchy, scripts/momentum_study/mb_amg.cu):
// about a quarter of the row length, half on small levels where the launch latency dominates
static int csr_threads_per_row(int rows, int nnz) {
    double per_row = rows > 0 ? (double)nnz / rows : 1.0;
    double target = per_row / (rows < 20000 ? 2.0 : 4.0);
    int tpr = 1;
    while (tpr < 32 && 2 * tpr <= target) tpr *= 2;
    return tpr;
}

// Dense coarse solve y = A^{-1} x with the stored inverse, one warp per row
__global__ void k_dense_matvec(int n, const double* __restrict__ A, const double* __restrict__ x,
                               double* __restrict__ y) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (i >= n) return;
    double s = 0.0;
    for (int j = lane; j < n; j += 32) s += A[(size_t)i * n + j] * x[j];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if (lane == 0) y[i] = s;
}

class AmgPreconditioner {
public:
    // Device buffers are kept between setups and only reallocated when a level grows (cudaMalloc and
    // cudaFree of the whole hierarchy cost about as much as copying it)
    struct DevCsr {
        int rows = 0, nnz = 0;
        int tpr = 1;                    // threads per row of the fused kernels (csr_threads_per_row)
        int *rp = nullptr, *ci = nullptr;
        double *v = nullptr;
        size_t cap_rp = 0, cap_nnz = 0, cap_v = 0;
    };
    struct Level {
        DevCsr A, P, R;                 // P: n x nc, R = P^T
        double* dinv = nullptr;
        double omega = 0.0;             // Jacobi smoothing weight
        double *x = nullptr, *b = nullptr, *r = nullptr, *Ax = nullptr;
        size_t cap_n = 0;
    };

    int    smooth_steps = 2;
    double theta        = 0.08;
    int    coarse_target = 300;
    int    max_levels    = 15;

    std::vector<Level> levels;
    std::vector<Level> spare;         // buffers of the previous hierarchy, reused by the next setup
    int     coarse_n = 0;
    double* d_coarse_inv = nullptr;   // dense inverse of the coarsest operator
    double *d_cx = nullptr, *d_cb = nullptr;
    size_t  cap_coarse_inv = 0, cap_coarse = 0;
    double  setup_ms = 0.0;
    // Accumulated setup phases [ms]: download, D^-1 + rho, aggregation, P smoothing, transpose,
    // Galerkin product, upload, coarse inverse
    enum { PH_DOWNLOAD, PH_RHO, PH_AGG, PH_SMOOTH, PH_TRANSPOSE, PH_GALERKIN, PH_UPLOAD, PH_COARSE, PH_REMAP, PH_COUNT };
    HostCsr P0_host;                  // level-0 prolongator of the current hierarchy, kept for remap()
    int     remaps = 0;               // remap() calls since the start
    mutable double phase_ms[PH_COUNT] = {0, 0, 0, 0, 0, 0, 0, 0};
    mutable std::chrono::steady_clock::time_point phase_t;
    void phase_start() const { phase_t = std::chrono::steady_clock::now(); }
    void phase_end(int ph) const {
        auto now = std::chrono::steady_clock::now();
        phase_ms[ph] += std::chrono::duration<double, std::milli>(now - phase_t).count();
        phase_t = now;
    }

    AmgPreconditioner() = default;
    AmgPreconditioner(const AmgPreconditioner&) = delete;
    AmgPreconditioner& operator=(const AmgPreconditioner&) = delete;
    ~AmgPreconditioner() {
        release();
        for (Level& L : spare) free_level(L);
        free_coarse();
    }

    int num_levels() const { return (int)levels.size() + 1; }

    void setup(int n, int nnz, const int* d_row_ptr, const int* d_col_ind, const double* d_val) {
        auto t0 = std::chrono::steady_clock::now();
        release();
        phase_start();
        HostCsr A;
        A.rows = A.cols = n;
        A.rp.resize(n + 1);  A.ci.resize(nnz);  A.v.resize(nnz);
        CUDA_CHECK(cudaMemcpy(A.rp.data(), d_row_ptr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(A.ci.data(), d_col_ind, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(A.v.data(),  d_val,     nnz * sizeof(double), cudaMemcpyDeviceToHost));
        phase_end(PH_DOWNLOAD);

        while (A.rows > coarse_target && (int)levels.size() < max_levels - 1) {
            std::vector<double> dinv(A.rows);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < A.rows; ++i) {
                double d = 0.0;
                for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) if (A.ci[a] == i) d = A.v[a];
                dinv[i] = (std::fabs(d) > 1.0e-300) ? 1.0 / d : 1.0;
            }
            double omega = 4.0 / (3.0 * spectral_radius_DinvA(A, dinv));
            phase_end(PH_RHO);

            HostCsr P = smoothed_prolongator(A, dinv, omega);
            if (P.cols >= (int)(0.85 * A.rows) || P.cols == 0) break;   // coarsening stalled
            phase_start();
            HostCsr R  = csr_transpose(P);
            phase_end(PH_TRANSPOSE);
            HostCsr Ac = csr_multiply(R, csr_multiply(A, P));
            phase_end(PH_GALERKIN);

            Level L;
            if (levels.size() < spare.size()) L = spare[levels.size()];
            if (levels.empty()) {   // level 0 is the device matrix itself: device-to-device copy
                upload_from_device(A.rows, nnz, d_row_ptr, d_col_ind, d_val, L.A);
            } else {
                upload(A, L.A);
            }
            upload(P, L.P);  upload(R, L.R);
            L.omega = omega;
            ensure_level_vectors(L, A.rows);
            CUDA_CHECK(cudaMemcpy(L.dinv, dinv.data(), A.rows * sizeof(double), cudaMemcpyHostToDevice));
            if (levels.empty()) P0_host = std::move(P);
            levels.push_back(L);
            A = std::move(Ac);
            phase_end(PH_UPLOAD);
        }

        // Coarsest level: dense Cholesky inverse
        phase_start();
        coarse_n = A.rows;
        std::vector<double> D((size_t)coarse_n * coarse_n, 0.0);
        for (int i = 0; i < coarse_n; ++i)
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) D[(size_t)i * coarse_n + A.ci[a]] = A.v[a];
        std::vector<double> inv = dense_spd_inverse(D, coarse_n);
        ensure_dev(d_coarse_inv, cap_coarse_inv, inv.size());
        CUDA_CHECK(cudaMemcpy(d_coarse_inv, inv.data(), inv.size() * sizeof(double), cudaMemcpyHostToDevice));
        size_t cap_before = cap_coarse;
        ensure_dev(d_cx, cap_coarse, coarse_n);
        if (cap_coarse != cap_before || !d_cb) {   // d_cx and d_cb share one capacity
            if (d_cb) cudaFree(d_cb);
            CUDA_CHECK(cudaMalloc(&d_cb, cap_coarse * sizeof(double)));
        }
        phase_end(PH_COARSE);
        setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    /**
     * Carries the hierarchy over to a new mesh instead of rebuilding it (AMR changes a small part
     * of the mesh per remesh). Level 0 gets the new operator, its diagonal and a prolongator whose
     * rows come from the old one: a node kept by the remesh (new_to_old[i] = old row, -1 for a new
     * node) keeps its row, a new node gets the mean of the rows of its graph neighbours that already
     * have one (passes outwards until all are set). Levels >= 1, omega and the coarse inverse stay
     * as they are, so the coarse operators are those of the old mesh (non-Galerkin, still SPD); the
     * V-cycle stays symmetric, so CG remains valid, only the iteration count can rise. h_rp/h_ci:
     * host CSR graph of the new operator. Falls back to setup() when there is nothing to reuse.
     * Measured on A3 (HANDOVER 11.14): PPE iterations 33 -> 35-42, net about -4% wall. Redoing the
     * level-1 Galerkin product as well was tried and gave nothing (fewer iterations, dearer remap).
     */
    void remap(int n, int nnz, const int* d_row_ptr, const int* d_col_ind, const double* d_val,
               const int* h_rp, const int* h_ci, const std::vector<int>& new_to_old) {
        if (levels.empty() || P0_host.rows == 0) { setup(n, nnz, d_row_ptr, d_col_ind, d_val); return; }
        auto t0 = std::chrono::steady_clock::now();
        phase_start();
        const HostCsr& Po = P0_host;
        HostCsr P;
        P.rows = n;  P.cols = Po.cols;
        P.rp.assign(n + 1, 0);
        // Rows of new nodes, built pass by pass from neighbours set in earlier passes (deterministic)
        std::vector<int> extra_of(n, -1);
        std::vector<std::vector<std::pair<int, double>>> extra;
        std::vector<int> pending;
        for (int i = 0; i < n; ++i) if (new_to_old[i] < 0) pending.push_back(i);
        std::vector<char> has_row(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) has_row[i] = new_to_old[i] >= 0;
        std::vector<std::pair<int, double>> acc;
        while (!pending.empty()) {
            std::vector<int> still, set_now;
            for (int i : pending) {
                acc.clear();
                int cnt = 0;
                for (int a = h_rp[i]; a < h_rp[i + 1]; ++a) {
                    int k = h_ci[a];
                    if (k == i || !has_row[k]) continue;
                    ++cnt;
                    if (new_to_old[k] >= 0) {
                        int j = new_to_old[k];
                        for (int b = Po.rp[j]; b < Po.rp[j + 1]; ++b) acc.push_back({Po.ci[b], Po.v[b]});
                    } else {
                        for (const auto& e : extra[extra_of[k]]) acc.push_back(e);
                    }
                }
                if (cnt == 0) { still.push_back(i); continue; }
                std::sort(acc.begin(), acc.end(),
                          [](const std::pair<int, double>& x, const std::pair<int, double>& y) { return x.first < y.first; });
                std::vector<std::pair<int, double>> row;
                for (const auto& e : acc) {
                    if (!row.empty() && row.back().first == e.first) row.back().second += e.second;
                    else row.push_back(e);
                }
                for (auto& e : row) e.second /= cnt;
                extra_of[i] = (int)extra.size();
                extra.push_back(std::move(row));
                set_now.push_back(i);
            }
            if (set_now.empty()) {   // isolated new nodes: no reuse possible
                setup(n, nnz, d_row_ptr, d_col_ind, d_val);
                return;
            }
            for (int i : set_now) has_row[i] = 1;
            pending.swap(still);
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int j = new_to_old[i];
            P.rp[i + 1] = j >= 0 ? Po.rp[j + 1] - Po.rp[j] : (int)extra[extra_of[i]].size();
        }
        for (int i = 0; i < n; ++i) P.rp[i + 1] += P.rp[i];
        P.ci.resize(P.rp[n]);  P.v.resize(P.rp[n]);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int j = new_to_old[i], o = P.rp[i];
            if (j >= 0) {
                std::copy(Po.ci.begin() + Po.rp[j], Po.ci.begin() + Po.rp[j + 1], P.ci.begin() + o);
                std::copy(Po.v.begin() + Po.rp[j], Po.v.begin() + Po.rp[j + 1], P.v.begin() + o);
            } else {
                for (const auto& e : extra[extra_of[i]]) { P.ci[o] = e.first;  P.v[o] = e.second;  ++o; }
            }
        }
        HostCsr R = csr_transpose(P);
        Level& L = levels[0];
        upload_from_device(n, nnz, d_row_ptr, d_col_ind, d_val, L.A);
        upload(P, L.P);  upload(R, L.R);
        ensure_level_vectors(L, n);
        k_inverse_diagonal<<<(n + 255) / 256, 256>>>(n, L.A.rp, L.A.ci, L.A.v, L.dinv);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
        P0_host = std::move(P);
        ++remaps;
        phase_end(PH_REMAP);
        setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // z = M^{-1} r, one symmetric V-cycle. (Replaying it as a CUDA graph was tried: no gain, the
    // kernels themselves dominate.)
    void apply(const double* d_r, double* d_z) {
        if (levels.empty()) {
            k_dense_matvec<<<(coarse_n * 32 + 255) / 256, 256>>>(coarse_n, d_coarse_inv, d_r, d_z);
            return;
        }
        vcycle(0, d_r);
        CUDA_CHECK(cudaMemcpy(d_z, levels[0].x, levels[0].A.rows * sizeof(double), cudaMemcpyDeviceToDevice));
    }

private:
    template <int OP>
    static void csr(const DevCsr& M, const double* x, double* y, const double* b = nullptr,
                    const double* dinv = nullptr, double omega = 0.0) {
        csr_op_launch<OP>(M.tpr, M.rows, M.rp, M.ci, M.v, x, b, dinv, omega, y);
    }

    // One damped Jacobi step x <- x + omega D^{-1} (b - A x) into the spare vector, then swap
    static void jacobi_step(Level& L, const double* b) {
        csr<CSR_JACOBI>(L.A, L.x, L.Ax, b, L.dinv, L.omega);
        std::swap(L.x, L.Ax);
    }

    // V-cycle on level l for the right-hand side b (level 0: the caller's vector, else L.b)
    void vcycle(int l, const double* b) {
        Level& L = levels[l];
        int n = L.A.rows, nb = (n + 255) / 256;
        // pre-smoothing from x = 0
        k_jacobi_init<<<nb, 256>>>(n, L.omega, L.dinv, b, L.x);
        for (int s = 1; s < smooth_steps; ++s) jacobi_step(L, b);
        // restrict residual
        csr<CSR_RESID>(L.A, L.x, L.r, b);
        double* cb = (l + 1 < (int)levels.size()) ? levels[l + 1].b : d_cb;
        csr<CSR_SPMV>(L.R, L.r, cb);
        // coarse correction x += P x_c
        double* cx;
        if (l + 1 < (int)levels.size()) {
            vcycle(l + 1, levels[l + 1].b);
            cx = levels[l + 1].x;
        } else {
            k_dense_matvec<<<(coarse_n * 32 + 255) / 256, 256>>>(coarse_n, d_coarse_inv, d_cb, d_cx);
            cx = d_cx;
        }
        csr<CSR_ADD>(L.P, cx, L.x);
        // post-smoothing (same count: symmetric cycle)
        for (int s = 0; s < smooth_steps; ++s) jacobi_step(L, b);
    }

    template <class T>
    static void ensure_dev(T*& p, size_t& cap, size_t n) {
        if (n <= cap && p) return;
        if (p) CUDA_CHECK(cudaFree(p));
        cap = n + n / 8 + 16;
        CUDA_CHECK(cudaMalloc(&p, cap * sizeof(T)));
    }

    static void ensure_level_vectors(Level& L, int n) {
        if (L.dinv && (size_t)n <= L.cap_n) return;
        for (double* w : {L.dinv, L.x, L.b, L.r, L.Ax}) if (w) CUDA_CHECK(cudaFree(w));
        L.cap_n = (size_t)n + n / 8 + 16;
        for (double** w : {&L.dinv, &L.x, &L.b, &L.r, &L.Ax}) CUDA_CHECK(cudaMalloc(w, L.cap_n * sizeof(double)));
    }

    static void upload(const HostCsr& M, DevCsr& D) {
        D.rows = M.rows;  D.nnz = (int)M.ci.size();
        D.tpr = csr_threads_per_row(D.rows, D.nnz);
        ensure_dev(D.rp, D.cap_rp, M.rows + 1);
        ensure_dev(D.ci, D.cap_nnz, std::max(1, D.nnz));
        ensure_dev(D.v, D.cap_v, std::max(1, D.nnz));
        CUDA_CHECK(cudaMemcpy(D.rp, M.rp.data(), (M.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        if (D.nnz) {
            CUDA_CHECK(cudaMemcpy(D.ci, M.ci.data(), D.nnz * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(D.v,  M.v.data(),  D.nnz * sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    static void upload_from_device(int rows, int nnz, const int* d_rp, const int* d_ci, const double* d_v, DevCsr& D) {
        D.rows = rows;  D.nnz = nnz;
        D.tpr = csr_threads_per_row(rows, nnz);
        ensure_dev(D.rp, D.cap_rp, rows + 1);
        ensure_dev(D.ci, D.cap_nnz, std::max(1, nnz));
        ensure_dev(D.v, D.cap_v, std::max(1, nnz));
        CUDA_CHECK(cudaMemcpy(D.rp, d_rp, (rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(D.ci, d_ci, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(D.v,  d_v,  nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    static void free_csr(DevCsr& D) {
        if (D.rp) cudaFree(D.rp);
        if (D.ci) cudaFree(D.ci);
        if (D.v)  cudaFree(D.v);
        D = DevCsr();
    }

    static void free_level(Level& L) {
        free_csr(L.A);  free_csr(L.P);  free_csr(L.R);
        for (double* w : {L.dinv, L.x, L.b, L.r, L.Ax}) if (w) cudaFree(w);
        L = Level();
    }

    void free_coarse() {
        for (double** w : {&d_coarse_inv, &d_cx, &d_cb}) { if (*w) cudaFree(*w); *w = nullptr; }
        cap_coarse_inv = cap_coarse = 0;
        coarse_n = 0;
    }

    // Keeps the device buffers for the next setup()
    void release() {
        if (spare.size() < levels.size()) spare.resize(levels.size());
        for (size_t l = 0; l < levels.size(); ++l) spare[l] = levels[l];
        levels.clear();
        coarse_n = 0;
    }

    // Power iteration for rho(D^{-1} A). The norms are summed over fixed 4096-row chunks and the
    // chunk sums in order, so the result does not depend on the thread count.
    static double spectral_radius_DinvA(const HostCsr& A, const std::vector<double>& dinv) {
        const int n = A.rows;
        const int chunk = 4096, nch = (n + chunk - 1) / chunk;
        std::vector<double> x(n), y(n), part_y(nch), part_x(nch);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * std::sin(1.0 + i);
        double lambda = 1.0;
        for (int it = 0; it < 20; ++it) {
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < nch; ++c) {
                double sy = 0.0, sx = 0.0;
                for (int i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) {
                    double s = 0.0;
                    for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) s += A.v[a] * x[A.ci[a]];
                    y[i] = dinv[i] * s;
                    sy += y[i] * y[i];
                    sx += x[i] * x[i];
                }
                part_y[c] = sy;  part_x[c] = sx;
            }
            double nrm = 0.0, xn = 0.0;
            for (int c = 0; c < nch; ++c) { nrm += part_y[c]; xn += part_x[c]; }
            nrm = std::sqrt(nrm);
            lambda = nrm / std::sqrt(xn);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) x[i] = y[i] / nrm;
        }
        return std::max(lambda, 1.0e-12) * 1.05;   // small safety margin
    }

    HostCsr smoothed_prolongator(const HostCsr& A, const std::vector<double>& dinv, double omega) const {
        int n = A.rows;
        phase_start();
        // Strong connections, evaluated once per entry (the aggregation passes below repeat the query)
        std::vector<double> diag(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) diag[i] = 1.0 / dinv[i];
        std::vector<char> is_strong(A.ci.size());
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) {
                int j = A.ci[a];
                is_strong[a] = j != i && std::fabs(A.v[a]) >= theta * std::sqrt(std::fabs(diag[i] * diag[j]));
            }
        auto strong = [&](int, int a) { return is_strong[a] != 0; };
        // Greedy 3-phase aggregation
        std::vector<int> agg(n, -1);
        int na = 0;
        for (int i = 0; i < n; ++i) {
            if (agg[i] >= 0) continue;
            bool free_nbhd = true;
            for (int a = A.rp[i]; a < A.rp[i + 1] && free_nbhd; ++a)
                if (strong(i, a) && agg[A.ci[a]] >= 0) free_nbhd = false;
            if (!free_nbhd) continue;
            agg[i] = na;
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a) if (strong(i, a)) agg[A.ci[a]] = na;
            ++na;
        }
        std::vector<int> agg2(agg);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            if (agg[i] >= 0) continue;
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a)
                if (strong(i, a) && agg[A.ci[a]] >= 0) { agg2[i] = agg[A.ci[a]]; break; }
        }
        agg.swap(agg2);
        for (int i = 0; i < n; ++i) {
            if (agg[i] >= 0) continue;
            agg[i] = na;
            for (int a = A.rp[i]; a < A.rp[i + 1]; ++a)
                if (strong(i, a) && agg[A.ci[a]] < 0) agg[A.ci[a]] = na;
            ++na;
        }
        phase_end(PH_AGG);
        // Tentative prolongator (normalized constant on each aggregate)
        std::vector<int> size(na, 0);
        for (int i = 0; i < n; ++i) size[agg[i]]++;
        HostCsr P0;
        P0.rows = n;  P0.cols = na;
        P0.rp.resize(n + 1);  P0.ci.resize(n);  P0.v.resize(n);
        for (int i = 0; i < n; ++i) {
            P0.rp[i] = i;  P0.ci[i] = agg[i];  P0.v[i] = 1.0 / std::sqrt((double)size[agg[i]]);
        }
        P0.rp[n] = n;
        // P = P0 - omega D^{-1} A P0. Rows of AP0 are sorted by column; the aggregate's own column is
        // merged in place, so the rows of P come out sorted. Exact zeros are dropped. Two passes
        // (row counts, then values) so the rows are built in parallel.
        HostCsr AP0 = csr_multiply(A, P0);
        HostCsr P;
        P.rows = n;  P.cols = na;
        P.rp.assign(n + 1, 0);
        auto build_row = [&](int i, int* cols, double* vals) {
            int out = 0;
            bool own = false;
            const int own_col = agg[i];
            for (int a = AP0.rp[i]; a < AP0.rp[i + 1]; ++a) {
                int j = AP0.ci[a];
                if (!own && j > own_col) {   // own column absent from AP0: it belongs before this entry
                    if (cols) { cols[out] = own_col; vals[out] = P0.v[i]; }
                    ++out;  own = true;
                }
                double val = -omega * dinv[i] * AP0.v[a];
                if (j == own_col) { val += P0.v[i]; own = true; }
                if (val != 0.0) { if (cols) { cols[out] = j; vals[out] = val; } ++out; }
            }
            if (!own) { if (cols) { cols[out] = own_col; vals[out] = P0.v[i]; } ++out; }
            return out;
        };
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) P.rp[i + 1] = build_row(i, nullptr, nullptr);
        for (int i = 0; i < n; ++i) P.rp[i + 1] += P.rp[i];
        P.ci.resize(P.rp[n]);  P.v.resize(P.rp[n]);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) build_row(i, P.ci.data() + P.rp[i], P.v.data() + P.rp[i]);
        phase_end(PH_SMOOTH);
        return P;
    }

    static std::vector<double> dense_spd_inverse(std::vector<double> M, int n) {
        // Cholesky M = L L^T, then invert column by column
        for (int j = 0; j < n; ++j) {
            double d = M[(size_t)j * n + j];
            for (int k = 0; k < j; ++k) d -= M[(size_t)j * n + k] * M[(size_t)j * n + k];
            if (!(d > 0.0)) {
                fprintf(stderr, "[AMG] coarse operator not SPD (pivot %.3e), aborting\n", d);
                exit(EXIT_FAILURE);
            }
            double ljj = std::sqrt(d);
            M[(size_t)j * n + j] = ljj;
            for (int i = j + 1; i < n; ++i) {
                double s = M[(size_t)i * n + j];
                for (int k = 0; k < j; ++k) s -= M[(size_t)i * n + k] * M[(size_t)j * n + k];
                M[(size_t)i * n + j] = s / ljj;
            }
        }
        std::vector<double> inv((size_t)n * n, 0.0);
        #pragma omp parallel
        {
        std::vector<double> col(n);
        #pragma omp for schedule(dynamic, 8)
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < n; ++i) {                    // L y = e_c
                double s = (i == c) ? 1.0 : 0.0;
                for (int k = 0; k < i; ++k) s -= M[(size_t)i * n + k] * col[k];
                col[i] = s / M[(size_t)i * n + i];
            }
            for (int i = n - 1; i >= 0; --i) {               // L^T x = y
                double s = col[i];
                for (int k = i + 1; k < n; ++k) s -= M[(size_t)k * n + i] * col[k];
                col[i] = s / M[(size_t)i * n + i];
            }
            for (int i = 0; i < n; ++i) inv[(size_t)i * n + c] = col[i];
        }
        }
        return inv;
    }
};
