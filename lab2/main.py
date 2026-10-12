import matplotlib
matplotlib.use("Agg")
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    x = []
    y = []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["tasks"]))
            y.append(float(row["cost"]))
    return np.array(x), np.array(y)


def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table


def newton(x, table, value):
    result = np.zeros_like(np.asarray(value, dtype=float)) + table[0, 0]
    term = np.ones_like(result)

    for j in range(1, len(x)):
        term = term * (value - x[j - 1])
        result = result + table[0, j] * term

    return result


def factorial_interpolation(x, y, value):
    steps = np.diff(x)

    if not np.allclose(steps, steps[0]):
        raise ValueError("Факторіальна формула потребує рівновіддалених вузлів")

    t = (np.asarray(value, dtype=float) - x[0]) / steps[0]
    differences = np.array(y, dtype=float)
    result = np.zeros_like(t) + differences[0]
    factor = np.ones_like(t)

    for k in range(1, len(x)):
        differences = np.diff(differences)
        factor = factor * (t - (k - 1)) / k
        result = result + factor * differences[0]

    return result


def model_cost(x):
    z = np.asarray(x, dtype=float) / 1000
    return 0.15 + 0.27 * z + 0.004 * z ** 2 + 0.12 * np.sin(z / 2.5)


folder = Path(__file__).resolve().parent
x, y = read_data(folder / "data.csv")
print("Кількість завдань:", x)
print("Вартість:", y)

table = divided_differences(x, y)

print("ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ")

for i in range(len(x)):
    print(f"x = {x[i]:8.0f}:", "  ".join(f"{table[i, j]:.10g}" for j in range(len(x) - i)))

forecast = float(newton(x, table, 15000))

print(f"\nПрогноз Ньютона для 15000 завдань: ${forecast:.6f}")

nodes = np.linspace(1000, 20000, 5)
values = model_cost(nodes)
factorial_result = float(factorial_interpolation(nodes, values, 15000))

print(f"Результат факторіальної інтерполяції для 15000 завдань: ${factorial_result:.6f}")

xx = np.linspace(1000, 20000, 600)
yy = newton(x, table, xx)

plt.figure(figsize=(9, 5))
plt.scatter(x, y, label="Експериментальні дані", zorder=3)
plt.plot(xx, yy, label="Многочлен Ньютона")
plt.scatter([15000], [forecast], marker="x", s=100, label=f"Прогноз: ${forecast:.3f}")
plt.xlabel("Кількість завдань")
plt.ylabel("Вартість, $")
plt.title("Інтерполяція вартості обчислень")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder / "graph_1_original.png", dpi=180)
plt.close()

nodes = np.linspace(1000, 20000, 5)
values = model_cost(nodes)

xx_factorial = np.linspace(1000, 20000, 600)
yy_factorial = factorial_interpolation(nodes, values, xx_factorial)

factorial_result = float(factorial_interpolation(nodes, values, 15000))

plt.figure(figsize=(9, 5))
plt.scatter(nodes, values, label="Модельні дані", zorder=3)
plt.plot(xx_factorial, yy_factorial, label="Факторіальний многочлен")
plt.scatter([15000], [factorial_result], marker="x", s=100,
            label=f"Значення: ${factorial_result:.3f}")

plt.xlabel("Кількість завдань")
plt.ylabel("Вартість, $")
plt.title("Факторіальна інтерполяція вартості обчислень")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder / "graph_5_factorial.png", dpi=180)
plt.close()

print("\nДОСЛІДЖЕННЯ НА МОДЕЛЬНИХ ДАНИХ")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for n in [5, 10, 20]:
    nodes = np.linspace(1000, 20000, n)
    values = model_cost(nodes)

    dd = divided_differences(nodes, values)
    prediction_newton = newton(nodes, dd, xx)
    prediction_factorial = factorial_interpolation(nodes, values, xx)
    print(f"Факторіальна інтерполяція ({n} вузлів): ${float(factorial_interpolation(nodes, values, 15000)):.6f}")

    error = np.abs(prediction_newton - model_cost(xx))
    difference = np.max(np.abs(prediction_newton - prediction_factorial))

    print(f"{n} вузлів: прогноз(15000) = ${float(newton(nodes, dd, 15000)):.6f}, "
          f"макс. похибка = {np.max(error):.3e}, різниця методів = {difference:.3e}")

    axes[0].plot(xx, prediction_newton, label=f"{n} вузлів")
    axes[1].plot(xx, error, label=f"{n} вузлів")

axes[0].plot(xx, model_cost(xx), "--", label="Модельна функція")
axes[0].set_title("Вплив кількості вузлів, сталий інтервал")
axes[1].set_title("Абсолютна похибка, сталий інтервал")

for ax in axes:
    ax.set_xlabel("Кількість завдань")
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel("Вартість, $")
axes[1].set_ylabel("Абсолютна похибка, $")

fig.tight_layout()
fig.savefig(folder / "graph_2_nodes.png", dpi=180)
plt.close(fig)

plt.figure(figsize=(9, 5))

for n in [5, 10, 20]:
    step = 1000
    nodes = 1000 + step * np.arange(n)
    values = model_cost(nodes)

    dd = divided_differences(nodes, values)
    grid = np.linspace(nodes[0], nodes[-1], 300)
    error = np.abs(newton(nodes, dd, grid) - model_cost(grid))

    plt.plot(grid, error, label=f"{n} вузлів, h={step}")

plt.title("Фіксований крок, змінний інтервал: похибки")
plt.xlabel("Кількість завдань")
plt.ylabel("Абсолютна похибка, $")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder / "graph_3_fixed_step.png", dpi=180)
plt.close()


def runge_function(z):
    return 1 / (1 + 25 * z ** 2)


zgrid = np.linspace(-1, 1, 600)

plt.figure(figsize=(9, 5))
plt.plot(zgrid, runge_function(zgrid), "--", label="Функція Рунге")

for n in [5, 10, 20]:
    nodes = np.linspace(-1, 1, n)
    dd = divided_differences(nodes, runge_function(nodes))
    values = newton(nodes, dd, zgrid)

    plt.plot(zgrid, values, label=f"{n} вузлів")

    print(f"Ефект Рунге, {n} вузлів: макс. похибка = "
          f"{np.max(np.abs(values - runge_function(zgrid))):.6f}")

plt.title("Ефект Рунге на рівномірних вузлах")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.ylim(-2, 3)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder / "graph_4_runge.png", dpi=180)
plt.close()

print("\nЗбережено 4 графіки.")
