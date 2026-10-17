import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = sum(x ** (i + j))

    return A


def form_vector(x, y, m):
    b = np.zeros(m + 1)

    for i in range(m + 1):
        b[i] = sum(y * x ** i)

    return b


def gauss_solve(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)

    for k in range(n):
        max_row = k

        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[max_row, k]):
                max_row = i

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        if abs(A[k, k]) < 1e-12:
            raise ValueError("Система не має єдиного розв'язку")

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]

            for j in range(k, n):
                A[i, j] = A[i, j] - factor * A[k, j]

            b[i] = b[i] - factor * b[k]

    result = np.zeros(n)

    for i in range(n - 1, -1, -1):
        s = 0

        for j in range(i + 1, n):
            s = s + A[i, j] * result[j]

        result[i] = (b[i] - s) / A[i, i]

    return result


def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)

    for i in range(len(coef)):
        y_poly = y_poly + coef[i] * x ** i

    return y_poly


def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)


def error_function(y_true, y_approx):
    return y_true - y_approx


data = pd.read_csv("temperatures.csv")

x = data["Month"].to_numpy(dtype=float)
y = data["Temp"].to_numpy(dtype=float)

print("Вхідні дані:")

for i in range(len(x)):
    print("Місяць:", int(x[i]), "Температура:", y[i])

degrees = list(range(1, 11))
variances = []
coefficients = []

print("\nДисперсія для різних степенів многочлена:")

for m in degrees:
    A = form_matrix(x, m)
    b = form_vector(x, y, m)

    coef = gauss_solve(A, b)
    y_approx = polynomial(x, coef)

    var = variance(y, y_approx)

    variances.append(var)
    coefficients.append(coef)

    print("Степінь:", m, "Дисперсія:", round(var, 6))

optimal_index = np.argmin(variances)
optimal_m = degrees[optimal_index]
optimal_coef = coefficients[optimal_index]

print("\nОптимальний степінь многочлена:", optimal_m)

print("\nКоефіцієнти многочлена:")

for i in range(len(optimal_coef)):
    print("a" + str(i) + " =", optimal_coef[i])

y_optimal = polynomial(x, optimal_coef)

errors = error_function(y, y_optimal)

print("\nПохибки апроксимації:")

for i in range(len(x)):
    print("Місяць:", int(x[i]), "Похибка:", round(errors[i], 6))

x_future = np.array([25, 26, 27], dtype=float)
y_future = polynomial(x_future, optimal_coef)

print("\nПрогноз температури на наступні 3 місяці:")

for i in range(len(x_future)):
    print("Місяць:", int(x_future[i]),
          "Температура:", round(y_future[i], 2))

plt.figure(figsize=(9, 5))

plt.plot(degrees, variances, marker="o")
plt.xlabel("Степінь многочлена")
plt.ylabel("Дисперсія")
plt.title("Залежність дисперсії від степеня многочлена")
plt.grid(True)
plt.show()

x_smooth = np.linspace(min(x), max(x), 300)
y_smooth = polynomial(x_smooth, optimal_coef)

plt.figure(figsize=(9, 5))

plt.scatter(x, y, label="Фактичні температури")
plt.plot(x_smooth, y_smooth, color="red",
         label="Апроксимація")
plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.title("Апроксимація температур методом найменших квадратів")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 5))

plt.plot(x, errors, marker="o")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.title("Похибка апроксимації")
plt.grid(True)
plt.show()

x_error = np.linspace(min(x), max(x), 481)

plt.figure(figsize=(10, 6))

for m in degrees:
    coef = coefficients[m - 1]
    y_nodes = polynomial(x, coef)
    errors_nodes = y - y_nodes
    errors_smooth = np.interp(x_error, x, errors_nodes)

    plt.plot(x_error, errors_smooth, label="m = " + str(m))

plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.title("Похибки апроксимації для різних степенів")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 5))

plt.scatter(x, y, label="Фактичні дані")
plt.plot(x_smooth, y_smooth, label="Апроксимація")
plt.scatter(x_future, y_future, color="red",
            label="Прогноз")

plt.plot(np.concatenate((x_smooth, x_future)),
         polynomial(np.concatenate((x_smooth, x_future)),
                    optimal_coef),
         linestyle="--")

plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.title("Прогноз температури на наступні 3 місяці")
plt.legend()
plt.grid(True)
plt.show()
