"""Momentum preconditioner study on a dumped system (--dump-systems).
usage: study.py <prefix>_momentum.bin [methods...]"""
import sys, time, ctypes as C
import numpy as np, scipy.sparse as sp

# build: gcc -O3 -march=native -shared -fPIC prec.c -o libprec.so (next to this script); needs numpy, scipy (pyamg for 'amg')
lib = C.CDLL(__file__.rsplit('/', 1)[0] + '/libprec.so')
P = np.ctypeslib.ndpointer
I32 = P(np.int32, flags='C'); F64 = P(np.float64, flags='C')
lib.ilu0.argtypes = [C.c_int, I32, I32, F64, I32]
lib.ilu0_apply.argtypes = [C.c_int, I32, I32, F64, I32, F64, F64]
lib.sgs_apply.argtypes = [C.c_int, I32, I32, F64, I32, F64, F64, C.c_int, C.c_double]
lib.levels.argtypes = [C.c_int, I32, I32, C.c_int]; lib.levels.restype = C.c_int

def load(fn):
    b = open(fn, 'rb').read(); o = 0
    def take(dt, k):
        nonlocal o; a = np.frombuffer(b, dt, k, o); o += a.nbytes; return a.copy()
    n, nnz = take(np.int32, 2)
    d = dict(rp=take(np.int32, n + 1), ci=take(np.int32, nnz), val=take(np.float64, nnz))
    for k in ('bu', 'bv', 'u0', 'v0', 'x', 'y'): d[k] = take(np.float64, n)
    d['tag'] = take(np.int32, n)
    if o < len(b): d['u'] = take(np.float64, n); d['v'] = take(np.float64, n)
    d['A'] = sp.csr_matrix((d['val'], d['ci'], d['rp']), shape=(n, n)); d['A'].sort_indices()
    return d

def bicgstab(A, b, x, prec, tol=1e-7, maxit=1000):
    """Same algorithm and stopping rule as GpuLinearSolver::solve_bicgstab (right preconditioning)."""
    x = x.copy(); r = b - A @ x; r0 = r.copy(); p = r.copy(); v = np.zeros_like(b)
    rn0 = np.linalg.norm(r)
    if rn0 < 1e-13: return x, 0
    rho_old = alpha = omega = 1.0; bd = 1e-30 * rn0**2 + 1e-300
    for it in range(maxit):
        rho = r0 @ r
        if abs(rho) < bd: break
        if it > 0:
            beta = (rho / rho_old) * (alpha / (omega + 1e-30)); p = r + beta * (p - omega * v)
        ph = prec(p); v = A @ ph; alpha = rho / (r0 @ v); s = r - alpha * v
        sn = np.linalg.norm(s)
        if sn < tol * rn0 or sn < 1e-11: return x + alpha * ph, it + 1
        sh = prec(s); t = A @ sh; tt = t @ t; omega = (t @ s) / tt if tt > bd else 0.0
        x = x + alpha * ph + omega * sh; r = s - omega * t
        if np.linalg.norm(r) < tol * rn0 or np.linalg.norm(r) < 1e-11: return x, it + 1
        rho_old = rho
    return x, -1

def permuted(A, perm):
    B = A[perm][:, perm].tocsr(); B.sort_indices(); return B

def greedy_colors(A):
    n = A.shape[0]; col = -np.ones(n, np.int64); rp, ci = A.indptr, A.indices
    for i in range(n):
        used = set(col[ci[rp[i]:rp[i + 1]]]); c = 0
        while c in used: c += 1
        col[i] = c
    return col

def make_prec(A, kind, **kw):
    n = A.shape[0]; rp = A.indptr.astype(np.int32); ci = A.indices.astype(np.int32)
    val = A.data.astype(np.float64).copy(); diag = np.zeros(n, np.int32)
    if kind == 'jacobi':
        dinv = 1.0 / A.diagonal(); return lambda r: dinv * r
    if kind == 'ilu0':
        lib.ilu0(n, rp, ci, val, diag)
        def f(r):
            z = np.empty(n); lib.ilu0_apply(n, rp, ci, val, diag, np.ascontiguousarray(r), z); return z
        return f
    if kind == 'ilu0_sweep':   # exact ILU(0) factors, triangular solves by k Jacobi sweeps
        lib.ilu0(n, rp, ci, val, diag)
        F = sp.csr_matrix((val, ci, rp), shape=(n, n))
        Ls = sp.tril(F, -1).tocsr(); Us = sp.triu(F, 1).tocsr(); dU = F.diagonal(); k = kw['k']
        def f(r):
            y = r.copy()
            for _ in range(k): y = r - Ls @ y
            z = y / dU
            for _ in range(k): z = (y - Us @ z) / dU
            return z
        return f
    if kind == 'sgs':
        lib.ilu0(n, rp, ci, val.copy(), diag)   # only to fill diag positions
        ns, w = kw.get('nsweep', 1), kw.get('w', 1.0)
        def f(r):
            z = np.empty(n); lib.sgs_apply(n, rp, ci, val, diag, np.ascontiguousarray(r), z, ns, w); return z
        return f
    if kind == 'amg':
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A, symmetry=kw.get('symmetry', 'nonsymmetric'),
                 strength=('symmetric', {'theta': kw.get('theta', 0.08)}),
                 smooth=('jacobi', {'omega': 4.0 / 3.0}),
                 presmoother=('jacobi', {'omega': 2.0 / 3.0, 'iterations': kw.get('nu', 1)}),
                 postsmoother=('jacobi', {'omega': 2.0 / 3.0, 'iterations': kw.get('nu', 1)}),
                 max_coarse=500)
        print('   ', str(ml).splitlines()[1:4])
        return lambda r: ml.solve(r, x0=np.zeros_like(r), maxiter=1, cycle='V', tol=1e-30)
    raise ValueError(kind)

def run(A, d, label, prec_kind, perm=None, **kw):
    t0 = time.time()
    if perm is not None:
        A = permuted(A, perm); bu, bv, u0, v0 = (d[k][perm] for k in ('bu', 'bv', 'u0', 'v0'))
    else:
        bu, bv, u0, v0 = d['bu'], d['bv'], d['u0'], d['v0']
    M = make_prec(A, prec_kind, **kw); ts = time.time() - t0
    xu, iu = bicgstab(A, bu, u0, M); xv, iv = bicgstab(A, bv, v0, M)
    err = ''
    if 'u' in d:
        ru = xu if perm is None else np.empty_like(xu); 
        if perm is not None: ru[perm] = xu
        err = f'| max|u-u_gpu| {np.abs(ru - d["u"]).max():.2e}'
    lv = ''
    if prec_kind == 'ilu0':
        rp, ci = A.indptr.astype(np.int32), A.indices.astype(np.int32)
        lv = f'| levels L {lib.levels(A.shape[0], rp, ci, 1)} U {lib.levels(A.shape[0], rp, ci, 0)}'
    print(f'{label:28s} its u {iu:4d} v {iv:4d} {lv} {err} ({time.time()-t0:.1f} s)', flush=True)

if __name__ == '__main__':
    d = load(sys.argv[1]); A = d['A']; n = A.shape[0]
    dg = A.diagonal(); offs = abs(A).sum(1).A1 - abs(dg)
    print(f'n {n}, nnz {A.nnz}, rows with |a_ii| < sum|a_ij|: {(abs(dg) < offs).sum()}, '
          f'asym ||A-A^T||/||A|| {sp.linalg.norm(A - A.T) / sp.linalg.norm(A):.3f}')
    which = sys.argv[2:] or ['jacobi', 'ilu0', 'sgs', 'mc']
    if 'jacobi' in which: run(A, d, 'Jacobi', 'jacobi')
    if 'ilu0' in which: run(A, d, 'ILU(0) natural', 'ilu0')
    if 'sgs' in which:
        run(A, d, 'SGS natural', 'sgs')
        run(A, d, 'SGS x2 natural', 'sgs', nsweep=2)
    if 'xsort' in which:
        perm = np.lexsort((d['y'], d['x']))
        run(A, d, 'ILU(0) x-sorted', 'ilu0', perm=perm)
        run(A, d, 'SGS x-sorted', 'sgs', perm=perm)
    if 'sweep' in which:
        perm = np.lexsort((d['y'], d['x']))
        for k in (2, 3, 5):
            run(A, d, f'ILU(0) x-sorted, {k} sweeps', 'ilu0_sweep', perm=perm, k=k)
            run(A, d, f'ILU(0) natural, {k} sweeps', 'ilu0_sweep', k=k)
    if 'mcx' in which:   # greedy colouring visited in x order
        px = np.lexsort((d['y'], d['x'])); Ax = permuted(A, px); col = greedy_colors(Ax)
        print(f'greedy colours (x order): {col.max() + 1}')
        perm = px[np.argsort(col, kind='stable')]
        run(A, d, 'ILU(0) multicolour, x order', 'ilu0', perm=perm)
    if 'amg' in which:
        run(A, d, 'SA-AMG V(1,1) Jacobi', 'amg')
        run(A, d, 'SA-AMG V(2,2) Jacobi', 'amg', nu=2)
    if 'mc' in which:
        col = greedy_colors(A); print(f'greedy colours: {col.max() + 1}')
        perm = np.argsort(col, kind='stable')
        run(A, d, 'ILU(0) multicolour', 'ilu0', perm=perm)
        run(A, d, 'SGS multicolour', 'sgs', perm=perm)
        run(A, d, 'SGS x2 multicolour', 'sgs', perm=perm, nsweep=2)
