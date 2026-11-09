// Preconditioner kernels for the momentum solver study (CSR, sorted columns assumed)
#include <stdlib.h>
#include <string.h>

// ILU(0) in place on a copy of val; diag[i] = position of a_ii
void ilu0(int n, const int* rp, const int* ci, double* a, int* diag) {
    int* pos = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) pos[i] = -1;
    for (int i = 0; i < n; ++i)
        for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] == i) diag[i] = k;
    for (int i = 0; i < n; ++i) {
        for (int k = rp[i]; k < rp[i + 1]; ++k) pos[ci[k]] = k;
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            int j = ci[k];
            if (j >= i) continue;
            a[k] /= a[diag[j]];
            double lij = a[k];
            for (int m = diag[j] + 1; m < rp[j + 1]; ++m) {
                int p = pos[ci[m]];
                if (p >= 0) a[p] -= lij * a[m];
            }
        }
        for (int k = rp[i]; k < rp[i + 1]; ++k) pos[ci[k]] = -1;
    }
    free(pos);
}

// z = (LU)^{-1} r
void ilu0_apply(int n, const int* rp, const int* ci, const double* a, const int* diag,
                const double* r, double* z) {
    for (int i = 0; i < n; ++i) {
        double s = r[i];
        for (int k = rp[i]; k < diag[i]; ++k) s -= a[k] * z[ci[k]];
        z[i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = z[i];
        for (int k = diag[i] + 1; k < rp[i + 1]; ++k) s -= a[k] * z[ci[k]];
        z[i] = s / a[diag[i]];
    }
}

// Symmetric Gauss-Seidel, nsweep forward+backward pairs, z0 = 0, damping w (SSOR)
void sgs_apply(int n, const int* rp, const int* ci, const double* a, const int* diag,
               const double* r, double* z, int nsweep, double w) {
    memset(z, 0, n * sizeof(double));
    for (int s = 0; s < nsweep; ++s) {
        for (int i = 0; i < n; ++i) {
            double acc = r[i];
            for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] != i) acc -= a[k] * z[ci[k]];
            z[i] = (1 - w) * z[i] + w * acc / a[diag[i]];
        }
        for (int i = n - 1; i >= 0; --i) {
            double acc = r[i];
            for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] != i) acc -= a[k] * z[ci[k]];
            z[i] = (1 - w) * z[i] + w * acc / a[diag[i]];
        }
    }
}

// Dependency depth of the lower (lower=1) or upper triangle: number of levels of a level-scheduled solve
int levels(int n, const int* rp, const int* ci, int lower) {
    int* lev = (int*)calloc(n, sizeof(int));
    int mx = 0;
    if (lower) {
        for (int i = 0; i < n; ++i) { int l = 0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] < i && lev[ci[k]] + 1 > l) l = lev[ci[k]] + 1;
            lev[i] = l; if (l > mx) mx = l; }
    } else {
        for (int i = n - 1; i >= 0; --i) { int l = 0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] > i && lev[ci[k]] + 1 > l) l = lev[ci[k]] + 1;
            lev[i] = l; if (l > mx) mx = l; }
    }
    free(lev);
    return mx + 1;
}
