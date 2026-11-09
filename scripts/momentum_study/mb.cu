// Microbenchmarks of SpMV and multicolour ILU(0) kernels on a dumped momentum system
#define CFA_NO_MAIN
// nvcc -O3 -arch=native -std=c++17 -Xcompiler -fopenmp scripts/momentum_study/mb.cu -o mb -lcusparse && ./mb <prefix>_momentum.bin
#include "../../src/cylinder_fem_amr.cu"
#include <cusparse.h>

template <int TPR>   // TPR threads per row
__global__ void k_spmv_vec(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                           const double* __restrict__ val, const double* __restrict__ x, double* __restrict__ y) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int row = g / TPR, lane = g % TPR;
    double s = 0.0;
    if (row < n) for (int k = rp[row] + lane; k < rp[row + 1]; k += TPR) s += val[k] * __ldg(&x[ci[k]]);
    for (int o = TPR / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffff, s, o, TPR);
    if (row < n && lane == 0) y[row] = s;
}
__global__ void k_spmv2(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                        const double* __restrict__ val, const double* __restrict__ x1, const double* __restrict__ x2,
                        double* __restrict__ y1, double* __restrict__ y2) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double s1 = 0, s2 = 0;
    for (int k = rp[row]; k < rp[row + 1]; ++k) { double a = val[k]; int c = ci[k]; s1 += a * x1[c]; s2 += a * x2[c]; }
    y1[row] = s1; y2[row] = s2;
}
// Split storage: L and U entries of the rows of each colour stored contiguously in colour-row order
__global__ void k_lower_split(int m, const int* __restrict__ rows, const int* __restrict__ lp, const int* __restrict__ lc,
                              const double* __restrict__ lv, const double* __restrict__ r, double* __restrict__ z) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= m) return;
    int i = rows[t]; double acc = r[i];
    for (int a = lp[t]; a < lp[t + 1]; ++a) acc -= lv[a] * z[lc[a]];
    z[i] = acc;
}
__global__ void k_upper_split(int m, const int* __restrict__ rows, const int* __restrict__ up, const int* __restrict__ uc,
                              const double* __restrict__ uv, const double* __restrict__ dinv, double* __restrict__ z) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= m) return;
    int i = rows[t]; double acc = z[i];
    for (int a = up[t]; a < up[t + 1]; ++a) acc -= uv[a] * z[uc[a]];
    z[i] = acc * dinv[t];
}

int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb"); int hdr[2]; fread(hdr, 4, 2, f);
    int n = hdr[0], nnz = hdr[1];
    std::vector<int> rp(n + 1), ci(nnz); std::vector<double> val(nnz), b(n);
    fread(rp.data(), 4, n + 1, f); fread(ci.data(), 4, nnz, f); fread(val.data(), 8, nnz, f); fread(b.data(), 8, n, f);
    fclose(f);
    int *d_rp, *d_ci; double *d_val, *d_x, *d_x2, *d_y, *d_y2;
    cudaMalloc(&d_rp, (n + 1) * 4); cudaMalloc(&d_ci, nnz * 4); cudaMalloc(&d_val, nnz * 8);
    cudaMalloc(&d_x, n * 8); cudaMalloc(&d_x2, n * 8); cudaMalloc(&d_y, n * 8); cudaMalloc(&d_y2, n * 8);
    cudaMemcpy(d_rp, rp.data(), (n + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, ci.data(), nnz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val.data(), nnz * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, b.data(), n * 8, cudaMemcpyHostToDevice); cudaMemcpy(d_x2, b.data(), n * 8, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time = [&](const char* name, int reps, auto fn, double bytes) {
        for (int i = 0; i < 5; ++i) fn();
        cudaEventRecord(e0); for (int i = 0; i < reps; ++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); double us = 1000.0 * ms / reps;
        printf("%-40s %8.1f us  %6.0f GB/s\n", name, us, bytes / us / 1e3);
    };
    printf("n %d nnz %d\n", n, nnz);
    double spmv_bytes = nnz * 12.0 + n * 20.0;
    int B = 256;
    time("spmv scalar (current)", 500, [&] { k_spmv_csr<<<(n + B - 1) / B, B>>>(n, d_rp, d_ci, d_val, d_x, d_y); }, spmv_bytes);
    time("spmv vector 2 thr/row", 500, [&] { k_spmv_vec<2><<<(2 * n + B - 1) / B, B>>>(n, d_rp, d_ci, d_val, d_x, d_y); }, spmv_bytes);
    time("spmv vector 4 thr/row", 500, [&] { k_spmv_vec<4><<<(4 * n + B - 1) / B, B>>>(n, d_rp, d_ci, d_val, d_x, d_y); }, spmv_bytes);
    time("spmv vector 8 thr/row", 500, [&] { k_spmv_vec<8><<<(8 * n + B - 1) / B, B>>>(n, d_rp, d_ci, d_val, d_x, d_y); }, spmv_bytes);
    time("spmv scalar, 2 rhs", 500, [&] { k_spmv2<<<(n + B - 1) / B, B>>>(n, d_rp, d_ci, d_val, d_x, d_x2, d_y, d_y2); }, nnz * 12.0 + n * 36.0);
    {
        cusparseHandle_t h; cusparseCreate(&h);
        cusparseSpMatDescr_t A; cusparseDnVecDescr_t X, Y;
        cusparseCreateCsr(&A, n, n, nnz, d_rp, d_ci, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateDnVec(&X, n, d_x, CUDA_R_64F); cusparseCreateDnVec(&Y, n, d_y, CUDA_R_64F);
        double one = 1, zero = 0; size_t bs; void* buf;
        for (auto alg : {CUSPARSE_SPMV_CSR_ALG1, CUSPARSE_SPMV_CSR_ALG2}) {
            cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, &zero, Y, CUDA_R_64F, alg, &bs);
            cudaMalloc(&buf, bs + 8);
            cusparseSpMV_preprocess(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, &zero, Y, CUDA_R_64F, alg, buf);
            time(alg == CUSPARSE_SPMV_CSR_ALG1 ? "cusparse CSR ALG1" : "cusparse CSR ALG2", 500, [&] {
                cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, &zero, Y, CUDA_R_64F, alg, buf); }, spmv_bytes);
            cudaFree(buf);
        }
    }
    // Multicolour ILU(0)
    MulticolourIlu0 ilu;
    ilu.set_pattern(n, rp, ci, d_rp, d_ci);
    printf("colours %d, rows per colour:", ilu.n_colours);
    for (int c = 0; c < ilu.n_colours; ++c) printf(" %d", ilu.colour_start[c + 1] - ilu.colour_start[c]);
    printf("\n");
    time("ILU factor", 100, [&] { ilu.factor(d_val); }, nnz * 20.0);
    time("ILU apply (current)", 500, [&] { ilu.apply(d_x, d_y); }, 2 * spmv_bytes);
    // split L/U copy
    std::vector<double> lu(nnz); cudaMemcpy(lu.data(), ilu.d_lu, nnz * 8, cudaMemcpyDeviceToHost);
    std::vector<int> rows(n); cudaMemcpy(rows.data(), ilu.d_rows, n * 4, cudaMemcpyDeviceToHost);
    std::vector<int> col(n); cudaMemcpy(col.data(), ilu.d_colour, n * 4, cudaMemcpyDeviceToHost);
    std::vector<int> lp(n + 1, 0), up(n + 1, 0), lc, uc; std::vector<double> lv, uv, dinv(n);
    for (int t = 0; t < n; ++t) { int i = rows[t];
        for (int a = rp[i]; a < rp[i + 1]; ++a) { int j = ci[a];
            if (j == i) dinv[t] = 1.0 / lu[a];
            else if (col[j] < col[i]) { lc.push_back(j); lv.push_back(lu[a]); }
            else { uc.push_back(j); uv.push_back(lu[a]); } }
        lp[t + 1] = lc.size(); up[t + 1] = uc.size(); }
    int *d_lp, *d_up, *d_lc, *d_uc; double *d_lv, *d_uv, *d_dinv;
    cudaMalloc(&d_lp, (n + 1) * 4); cudaMalloc(&d_up, (n + 1) * 4); cudaMalloc(&d_lc, lc.size() * 4); cudaMalloc(&d_uc, uc.size() * 4);
    cudaMalloc(&d_lv, lv.size() * 8); cudaMalloc(&d_uv, uv.size() * 8); cudaMalloc(&d_dinv, n * 8);
    cudaMemcpy(d_lp, lp.data(), (n + 1) * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_up, up.data(), (n + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lc, lc.data(), lc.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_uc, uc.data(), uc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lv, lv.data(), lv.size() * 8, cudaMemcpyHostToDevice); cudaMemcpy(d_uv, uv.data(), uv.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dinv, dinv.data(), n * 8, cudaMemcpyHostToDevice);
    std::vector<double> z_ref(n), z_new(n);
    ilu.apply(d_x, d_y); cudaMemcpy(z_ref.data(), d_y, n * 8, cudaMemcpyDeviceToHost);
    auto split_apply = [&] {
        const auto& cs = ilu.colour_start; int* dr = ilu.d_rows;
        for (int c = 0; c < ilu.n_colours; ++c) { int m = cs[c + 1] - cs[c];
            k_lower_split<<<(m + 255) / 256, 256>>>(m, dr + cs[c], d_lp + cs[c], d_lc, d_lv, d_x, d_y); }
        for (int c = ilu.n_colours - 1; c >= 0; --c) { int m = cs[c + 1] - cs[c];
            k_upper_split<<<(m + 255) / 256, 256>>>(m, dr + cs[c], d_up + cs[c], d_uc, d_uv, d_dinv + cs[c], d_y); }
    };
    split_apply(); cudaMemcpy(z_new.data(), d_y, n * 8, cudaMemcpyDeviceToHost);
    double md = 0; for (int i = 0; i < n; ++i) md = std::max(md, std::fabs(z_new[i] - z_ref[i]) / (std::fabs(z_ref[i]) + 1e-30));
    printf("split apply max rel diff %.2e\n", md);
    time("ILU apply (split L/U)", 500, split_apply, 2 * spmv_bytes);
    // dot as in the solver
    GpuLinearSolver sv; sv.allocate(n);
    time("dot (kernel + D2H + host sum)", 500, [&] { volatile double d = sv.dot(n, d_x, d_x2); (void)d; }, n * 16.0);
    time("empty kernel launch", 2000, [&] { k_axpy<<<1, 32>>>(0, 0.0, d_x, d_y); }, 1.0);
    time("axpy", 2000, [&] { k_axpy<<<(n + B - 1) / B, B>>>(n, 0.5, d_x, d_y); }, n * 24.0);
    return 0;
}
