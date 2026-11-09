import sympy as sp
x, y, t = sp.symbols('x y t', real=True)
L, H, nu, rho = sp.Rational(11, 5), sp.Rational(41, 100), sp.Rational(1, 1000), 1
k = 2 * sp.pi / sp.Rational(82, 100)       # wavelength 0.82 m in x
b = sp.pi / H                               # half wavelength over the channel height
xi = x - L
f = sp.cos(k * xi) - sp.Rational(1, 4) * sp.cos(2 * k * xi)    # f'(L) = f''(L) = 0
g = sp.sin(b * y) + sp.Rational(1, 2) * sp.cos(b * y)
A = 1 + sp.Rational(1, 2) * sp.sin(2 * sp.pi * t / sp.Rational(1, 5))   # period 0.2 s
psi = A * f * g / b
u = sp.diff(psi, y)
v = -sp.diff(psi, x)
p = A * sp.Rational(1, 2) * sp.sin(k * (L - x)) * sp.cos(b * y)     # p = 0 on the outlet
assert sp.simplify(sp.diff(u, x) + sp.diff(v, y)) == 0
# outlet natural BC of the code: nu du/dn = 0 (pressure gradient not integrated by parts)
assert sp.simplify(sp.diff(u, x).subs(x, L)) == 0 and sp.simplify(sp.diff(v, x).subs(x, L)) == 0
fx = sp.diff(u, t) + u * sp.diff(u, x) + v * sp.diff(u, y) - nu * (sp.diff(u, x, 2) + sp.diff(u, y, 2)) + sp.diff(p, x) / rho
fy = sp.diff(v, t) + u * sp.diff(v, x) + v * sp.diff(v, y) - nu * (sp.diff(v, x, 2) + sp.diff(v, y, 2)) + sp.diff(p, y) / rho
exprs = [u, v, p, fx, fy, sp.diff(u, x), sp.diff(u, y), sp.diff(v, x), sp.diff(v, y)]
names = ['u', 'v', 'p', 'fx', 'fy', 'ux', 'uy', 'vx', 'vy']
repl, red = sp.cse(exprs, symbols=sp.numbered_symbols('c'))
lines = []
for s_, e in repl:
    lines.append(f"    const double {s_} = {sp.ccode(sp.N(e, 17))};")
for n, e in zip(names, red):
    lines.append(f"    r.{n} = {sp.ccode(sp.N(e, 17))};")
print("\n".join(lines))
