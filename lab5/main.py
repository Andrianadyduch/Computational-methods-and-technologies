import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi, log2

def f(x):
    return 50 + 20*np.sin(pi*x/12) + 5*np.exp(-0.2*(x-12)**2)

def simpson(n):
    if n < 2 or n % 2:
        raise ValueError('N має бути додатним парним числом')
    y = f(np.linspace(0, 24, n+1))
    return 24/(3*n)*(y[0]+y[-1]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2]))

def adaptive(eps):
    cache = {}

    def value(x):
        if x not in cache:
            cache[x] = float(f(x))
        return cache[x]

    def integrate(a, b, fa, fm, fb, whole, tol, depth):
        m = (a+b)/2
        lm, rm = (a+m)/2, (m+b)/2
        fl, fr = value(lm), value(rm)
        left = (m-a)*(fa+4*fl+fm)/6
        right = (b-m)*(fm+4*fr+fb)/6
        delta = left+right-whole

        if abs(delta)/15 <= tol or depth >= 30:
            return left+right+delta/15

        return (integrate(a, m, fa, fl, fm, left, tol/2, depth+1)
                + integrate(m, b, fm, fr, fb, right, tol/2, depth+1))

    a, b, m = 0., 24., 12.
    fa, fm, fb = value(a), value(m), value(b)
    whole = (b-a)*(fa+4*fm+fb)/6
    return integrate(a, b, fa, fm, fb, whole, eps, 0), len(cache)

I0 = 1200 + 5*sqrt(pi/0.2)*erf(12*sqrt(0.2))

ns = np.arange(10, 1001, 2)
errors = np.array([abs(simpson(int(n))-I0) for n in ns])
valid = ns[errors <= 1e-12]
nopt = int(valid[0]) if len(valid) else None

if nopt is None:
    raise RuntimeError('Для N від 10 до 1000 точність 1e-12 не досягнута')

# Вибір N0, кратного 8
n0 = max(8, 8*int(nopt/80 + 0.5))

# Методи Сімпсона, Рунге-Ромберга та Ейткена
a, b, c = (simpson(n) for n in (n0, 2*n0, 4*n0))
rr = b+(b-a)/15

p = log2(abs((a-b)/(b-c))) if a != b and b != c else float('nan')
aitken = c+(c-b)/(2**p-1) if np.isfinite(p) and abs(2**p-1)>1e-14 else float('nan')

print(f'Точний інтеграл I0 = {I0:.12f}')
print(f'Nopt = {nopt}, похибка = {abs(simpson(nopt)-I0):.3e}')
print(f'N0 = {n0} (кратне 8, близьке до Nopt/10)')
print(f'Порядок за Ейткеном p = {p:.6f}')

print('\nПорівняння методів:')
print(f'{"Метод":<22}{"Інтеграл":>20}{"Абс. похибка":>19}')

for name, val in [('Сімпсон (N0)', a),
                  ('Рунге–Ромберг', rr),
                  ('Ейткен', aitken)]:
    print(f'{name:<22}{val:>20.12f}{abs(val-I0):>19.3e}')


print('\nАдаптивний алгоритм:')
print(f'{"eps":<12}{"Інтеграл":>20}{"Абс. похибка":>19}{"Виклики f":>12}')

for eps in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    val, calls = adaptive(eps)
    print(f'{eps:<12.0e}{val:>20.12f}{abs(val-I0):>19.3e}{calls:>12}')

# Графік функції
x = np.linspace(0, 24, 500)
plt.figure()
plt.plot(x, f(x))
plt.title('Навантаження сервера')
plt.xlabel('Час (год)')
plt.ylabel('Навантаження (%)')
plt.grid()
plt.savefig('function.png', dpi=180, bbox_inches='tight')

# Графік похибки
plt.figure()
plt.semilogy(ns, np.maximum(errors, np.finfo(float).eps))
plt.title('Похибка методу Сімпсона')
plt.xlabel('N')
plt.ylabel('Абсолютна похибка')
plt.grid()
plt.savefig('errors.png', dpi=180, bbox_inches='tight')
plt.show()
