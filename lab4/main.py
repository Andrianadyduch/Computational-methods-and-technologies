import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Аналітичне розв'язання
def M(t):
    return 50.0 * np.exp(-0.1 * t) + 5.0 * np.sin(t)


def dM_exact(t):
    return -5.0 * np.exp(-0.1 * t) + 5.0 * np.cos(t)


t0 = 1.0
exact_val = dM_exact(t0)

print("=" * 70)
print("Аналітичний розрахунок:")
print(f"Точка диференціювання: t0 = {t0}")
print("M'(t) = -5*exp(-0.1*t) + 5*cos(t)")
print(f"Точне значення похідної M'(1): {exact_val:.12f}")
print("=" * 70)


# 2. Дослідження залежності похибки від кроку h
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


h_steps = np.logspace(-20, 3, 400)

errors = np.array([
    abs(central_diff(M, t0, h) - exact_val)
    for h in h_steps
])

min_err_idx = np.argmin(errors)
h0_opt = h_steps[min_err_idx]
R0_opt = errors[min_err_idx]

print("\nРегуляризація за кроком h:")
print(f"Оптимальний серед перевірених крок h0: {h0_opt:.2e}")
print(f"Найменша отримана похибка R0: {R0_opt:.4e}")


# Графік залежності похибки від кроку h
plt.figure(figsize=(9, 5))

positive = errors > 0

plt.loglog(
    h_steps[positive],
    errors[positive],
    label="Абсолютна похибка R(h)",
    color="tab:blue",
    linewidth=2
)

if R0_opt > 0:
    plt.scatter(
        [h0_opt],
        [R0_opt],
        color="red",
        s=80,
        zorder=5,
        label=f"h0 = {h0_opt:.2e}, R0 = {R0_opt:.2e}"
    )

plt.axvline(
    h0_opt,
    color="red",
    linestyle="--",
    alpha=0.5
)

plt.xlabel("Крок диференціювання h (логарифмічна шкала)")
plt.ylabel("Абсолютна похибка R (логарифмічна шкала)")
plt.title("Залежність похибки чисельного диференціювання від кроку h")

plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()


# 3–5. Обчислення з кроками h та 2h і похибки R1
h_base = 1e-3

D_h = central_diff(M, t0, h_base)
D_2h = central_diff(M, t0, 2.0 * h_base)

R1 = abs(D_h - exact_val)
R_2h = abs(D_2h - exact_val)

print("\n" + "=" * 70)
print(f"Центральна різниця (h = {h_base}):")
print(f"D(h)  = {D_h:.12f} | R1 = {R1:.4e}")
print(f"D(2h) = {D_2h:.12f} | R(2h) = {R_2h:.4e}")


# 6. Метод Рунге-Ромберга
D_RR = D_h + (D_h - D_2h) / 3.0
R2 = abs(D_RR - exact_val)

print("\nМетод Рунге-Ромберга:")
print(f"D_RR = {D_RR:.12f} | R2 = {R2:.4e}")

if R2 > 0:
    print(f"Зменшення похибки: {R1 / R2:.2f} разів")
else:
    print("Похибка R2 дорівнює нулю в межах машинної точності")


# 7. Метод Ейткена та оцінка порядку точності
D_4h = central_diff(M, t0, 4.0 * h_base)
R_4h = abs(D_4h - exact_val)

denom_aitken = 2.0 * D_2h - D_4h - D_h

if abs(denom_aitken) > 1e-14:
    D_E = (D_2h**2 - D_4h * D_h) / denom_aitken
    R3 = abs(D_E - exact_val)
else:
    D_E = np.nan
    R3 = np.nan

if D_2h != D_h:
    ratio = (D_4h - D_2h) / (D_2h - D_h)
else:
    ratio = np.nan

if np.isfinite(ratio) and ratio != 0:
    p_aitken = np.log2(abs(ratio))
else:
    p_aitken = np.nan

print("\n" + "=" * 70)
print("Метод Ейткена та оцінка порядку точності (h, 2h, 4h):")
print(f"D(4h) = {D_4h:.12f} | R(4h) = {R_4h:.4e}")
print(f"D_E   = {D_E:.12f} | R3 = {R3:.4e}")
print(f"Оцінений порядок точності p = {p_aitken:.4f} (теоретичний p = 2)")
print("=" * 70)


# Порівняння похибок
print("\nПорівняння похибок:")
print(f"Центральна різниця: {R1:.4e}")
print(f"Рунге-Ромберг:     {R2:.4e}")
print(f"Ейткен:            {R3:.4e}")

if R2 > 0:
    print(f"Зменшення похибки методом Рунге-Ромберга: {R1 / R2:.2f} разів")

if np.isfinite(R3) and R3 > 0:
    print(f"Зменшення похибки методом Ейткена: {R1 / R3:.2f} разів")


# 8. Підсумкова таблиця результатів
results_data = [
    {
        "Метод": "Точне значення",
        "Крок": "-",
        "Значення похідної": f"{exact_val:.12f}",
        "Абсолютна похибка": "0"
    },
    {
        "Метод": "Центральна різниця D(h)",
        "Крок": f"h = {h_base:g}",
        "Значення похідної": f"{D_h:.12f}",
        "Абсолютна похибка": f"{R1:.4e}"
    },
    {
        "Метод": "Центральна різниця D(2h)",
        "Крок": f"2h = {2*h_base:g}",
        "Значення похідної": f"{D_2h:.12f}",
        "Абсолютна похибка": f"{R_2h:.4e}"
    },
    {
        "Метод": "Центральна різниця D(4h)",
        "Крок": f"4h = {4*h_base:g}",
        "Значення похідної": f"{D_4h:.12f}",
        "Абсолютна похибка": f"{R_4h:.4e}"
    },
    {
        "Метод": "Рунге-Ромберг",
        "Крок": "h, 2h",
        "Значення похідної": f"{D_RR:.12f}",
        "Абсолютна похибка": f"{R2:.4e}"
    },
    {
        "Метод": "Ейткен",
        "Крок": "h, 2h, 4h",
        "Значення похідної": f"{D_E:.12f}",
        "Абсолютна похибка": f"{R3:.4e}"
    },
    {
        "Метод": "Оптимальний крок h0",
        "Крок": f"h0 = {h0_opt:.2e}",
        "Значення похідної": f"{central_diff(M, t0, h0_opt):.12f}",
        "Абсолютна похибка": f"{R0_opt:.4e}"
    }
]

df_results = pd.DataFrame(results_data)

print("\nЗведена таблиця точності чисельного диференціювання:")
print(df_results.to_string(index=False))
