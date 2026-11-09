// AMG V-cycle kernel microbenchmark on a dumped pressure matrix (--dump-systems)
#define CFA_NO_MAIN
// nvcc -O3 -arch=native -std=c++17 -Xcompiler -fopenmp scripts/momentum_study/mb_amg.cu -o mb_amg && ./mb_amg <prefix>_pressure.bin [smooth_steps]
#include "../../src/cylinder_fem_amr.cu"

template <int TPR>
__global__ void k_spmv_vec(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                           const double* __restrict__ val, const double* __restrict__ x, double* __restrict__ y) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int row = g / TPR, lane = g % TPR;
    double s = 0.0;
    if (row < n) for (int k = rp[row] + lane; k < rp[row + 1]; k += TPR) s += val[k] * x[ci[k]];
    for (int o = TPR / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffff, s, o, TPR);
    if (row < n && lane == 0) y[row] = s;
}

int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb"); int hdr[2]; if (fread(hdr, 4, 2, f) != 2) return 1;
    int n = hdr[0], nnz = hdr[1];
    std::vector<int> rp(n + 1), ci(nnz); std::vector<double> val(nnz), b(n);
    if (fread(rp.data(), 4, n + 1, f) + fread(ci.data(), 4, nnz, f) + fread(val.data(), 8, nnz, f) + fread(b.data(), 8, n, f) == 0) return 1;
    fclose(f);
    int *d_rp, *d_ci; double *d_val, *d_b, *d_z;
    cudaMalloc(&d_rp, (n + 1) * 4); cudaMalloc(&d_ci, nnz * 4); cudaMalloc(&d_val, nnz * 8);
    cudaMalloc(&d_b, n * 8); cudaMalloc(&d_z, n * 8);
    cudaMemcpy(d_rp, rp.data(), (n + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, ci.data(), nnz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val.data(), nnz * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * 8, cudaMemcpyHostToDevice);
    AmgPreconditioner amg;
    if (argc > 2) amg.smooth_steps = atoi(argv[2]);
    amg.setup(n, nnz, d_rp, d_ci, d_val);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time = [&](int reps, auto fn) {
        for (int i = 0; i < 5; ++i) fn();
        cudaEventRecord(e0); for (int i = 0; i < reps; ++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); return 1000.0 * ms / reps;
    };
    double *d_x, *d_y; cudaMalloc(&d_x, n * 8); cudaMalloc(&d_y, n * 8); cudaMemcpy(d_x, d_b, n * 8, cudaMemcpyDeviceToDevice);
    printf("levels %d, coarse %d\n", amg.num_levels(), amg.coarse_n);
    const char* nm[3] = {"A", "R", "P"};
    for (size_t l = 0; l < amg.levels.size(); ++l) {
        auto& L = amg.levels[l];
        const AmgPreconditioner::DevCsr* M[3] = {&L.A, &L.R, &L.P};
        for (int k = 0; k < 3; ++k) {
            const auto& A = *M[k];
            printf("L%zu %s rows %7d nnz/row %6.1f :", l, nm[k], A.rows, (double)A.nnz / A.rows);
            int B = 256;
            printf(" 1:%6.1f", time(300, [&] { k_spmv_csr<<<(A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
            printf(" 2:%6.1f", time(300, [&] { k_spmv_vec<2><<<(2 * A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
            printf(" 4:%6.1f", time(300, [&] { k_spmv_vec<4><<<(4 * A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
            printf(" 8:%6.1f", time(300, [&] { k_spmv_vec<8><<<(8 * A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
            printf(" 16:%6.1f", time(300, [&] { k_spmv_vec<16><<<(16 * A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
            printf(" 32:%6.1f us\n", time(300, [&] { k_spmv_vec<32><<<(32 * A.rows + B - 1) / B, B>>>(A.rows, A.rp, A.ci, A.v, d_x, d_y); }));
        }
    }
    int cn = amg.coarse_n;
    printf("dense coarse %d: %.1f us\n", cn, time(300, [&] { k_dense_matvec<<<(cn * 32 + 255) / 256, 256>>>(cn, amg.d_coarse_inv, d_x, d_y); }));
    printf("V-cycle: %.1f us\n", time(300, [&] { amg.apply(d_b, d_z); }));
    GpuLinearSolver sv;
    cudaMemset(d_z, 0, n * 8);
    std::function<void(const double*, double*)> pc = [&](const double* r, double* z) { amg.apply(r, z); };
    sv.solve_pcg(n, d_rp, d_ci, d_val, d_b, d_z, 400, 1e-7, nullptr, &pc);   // warm-up (graph capture)
    cudaMemset(d_z, 0, n * 8);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    int it = sv.solve_pcg(n, d_rp, d_ci, d_val, d_b, d_z, 400, 1e-7, nullptr, &pc);
    printf("PCG (K_p only, from 0): %d its, %.2f ms, %.1f us/it\n", it, ms_since(t0), 1000 * ms_since(t0) / it);
    printf("dot: %.1f us\n", time(300, [&] { volatile double d = sv.dot(n, d_b, d_z); (void)d; }));
    return 0;
}
