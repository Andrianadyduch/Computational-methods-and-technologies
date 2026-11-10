import random
import math

N = 100
EPS = 1e-10
MAX_ITER = 100


def matrix_vector(A, x):
    return [
        sum(A[i][j] * x[j] for j in range(len(x)))
        for i in range(len(A))
    ]

def vector_norm(v):
    return math.sqrt(sum(value ** 2 for value in v))

def save_matrix(filename, A):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(f"{v:.15g}" for v in row) + "\n")

def save_vector(filename, v):
    with open(filename, "w") as f:
        for value in v:
            f.write(f"{value:.15g}\n")

def read_matrix(filename):
    with open(filename) as f:
        return [
            [float(v) for v in line.split()]
            for line in f if line.strip()
        ]

def read_vector(filename):
    with open(filename) as f:
        return [float(v) for v in f.read().split()]

def generate_system(n):
    random.seed(42)

    A = [
        [random.randint(-10, 10) for _ in range(n)]
        for _ in range(n)
    ]

    for i in range(n):
        A[i][i] = sum(abs(A[i][j]) for j in range(n)
                      if j != i) + 10

    x_true = [1.0] * n
    b = matrix_vector(A, x_true)

    save_matrix("matrix.txt", A)
    save_vector("vector_b.txt", b)

    return x_true

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            L[i][k] = A[i][k] - sum(
                L[i][j] * U[j][k] for j in range(k)
            )

        if abs(L[k][k]) < 1e-14:
            raise ValueError("Нульовий опорний елемент")

        for j in range(k + 1, n):
            U[k][j] = (
                A[k][j] -
                sum(L[k][s] * U[s][j] for s in range(k))
            ) / L[k][k]

    return L, U

def solve_lu(L, U, b):
    n = len(b)
    y = [0.0] * n
    x = [0.0] * n

    for i in range(n):
        y[i] = (
            b[i] - sum(L[i][j] * y[j] for j in range(i))
        ) / L[i][i]

    for i in range(n - 1, -1, -1):
        x[i] = y[i] - sum(
            U[i][j] * x[j] for j in range(i + 1, n)
        )

    return x

def refinement(A, b, L, U, x):
    for iteration in range(1, MAX_ITER + 1):
        Ax = matrix_vector(A, x)
        residual = [b[i] - Ax[i] for i in range(len(b))]

        delta = solve_lu(L, U, residual)
        x = [x[i] + delta[i] for i in range(len(x))]

        new_residual = [
            b[i] - value
            for i, value in enumerate(matrix_vector(A, x))
        ]

        if vector_norm(delta) < EPS and \
           vector_norm(new_residual) < EPS:
            return x, iteration

    return x, MAX_ITER

def main():
    print("LU-розклад та ітераційне уточнення\n")

    x_true = generate_system(N)

    A = read_matrix("matrix.txt")
    b = read_vector("vector_b.txt")

    print("Розмір матриці:", N, "x", N)
    print("Матрицю та вектор записано у файли")

    L, U = lu_decomposition(A)

    save_matrix("matrix_L.txt", L)
    save_matrix("matrix_U.txt", U)

    print("LU-розклад виконано")

    x = solve_lu(L, U, b)

    print("\nПерші 10 значень розв'язку:")
    for i in range(10):
        print(f"x[{i + 1}] = {x[i]:.12f}")

    error = vector_norm([
        x[i] - x_true[i] for i in range(N)
    ])

    residual = vector_norm([
        b[i] - value
        for i, value in enumerate(matrix_vector(A, x))
    ])

    print("\nПохибка:", f"{error:.12e}")
    print("Норма нев'язки:", f"{residual:.12e}")

    x_refined, iterations = refinement(A, b, L, U, x)

    refined_error = vector_norm([
        x_refined[i] - x_true[i] for i in range(N)
    ])

    print("\nІтераційне уточнення:")
    print("Кількість ітерацій:", iterations)
    print("Похибка після уточнення:",
          f"{refined_error:.12e}")

    save_vector("solution.txt", x_refined)
    print("\nРозв'язок записано у solution.txt")

if __name__ == "__main__":
    main()
