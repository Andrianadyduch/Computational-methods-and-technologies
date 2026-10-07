import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

print("Кількість вузлів:", n)

print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")

for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

with open("results.txt", "w") as file:
    file.write("№ | Latitude | Longitude | Elevation (m)\n")
    for i, point in enumerate(results):
        file.write(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}\n")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = []
elevations = []

for point in results:
    coords.append((point["latitude"], point["longitude"]))
    elevations.append(point["elevation"])

distances = [0]

for i in range(1, n):
    d = haversine(
        coords[i - 1][0],
        coords[i - 1][1],
        coords[i][0],
        coords[i][1]
    )

    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")

for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

x = np.array(distances)
y = np.array(elevations)

plt.figure()
plt.plot(x, y, "o-")
plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Висота, м")
plt.title("Профіль маршруту Заросляк - Говерла")
plt.grid()
plt.show()


def progonka(lower, main, upper, right):
    n = len(main)

    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -upper[0] / main[0]
    beta[0] = right[0] / main[0]

    for i in range(1, n):
        denominator = main[i] + lower[i] * alpha[i - 1]

        if i < n - 1:
            alpha[i] = -upper[i] / denominator

        beta[i] = (right[i] - lower[i] * beta[i - 1]) / denominator

    result = np.zeros(n)
    result[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        result[i] = alpha[i] * result[i + 1] + beta[i]

    return result


def spline_coefficients(x, y):
    n = len(x)

    h = np.zeros(n - 1)

    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]

    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    right = np.zeros(n)

    main[0] = 1
    right[0] = 0

    for i in range(1, n - 1):
        lower[i] = h[i - 1]
        main[i] = 2 * (h[i - 1] + h[i])
        upper[i] = h[i]

        right[i] = 3 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

    main[n - 1] = 1
    right[n - 1] = 0

    c = progonka(lower, main, upper, right)

    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        a[i] = y[i]

        b[i] = (
            (y[i + 1] - y[i]) / h[i]
            - h[i] * (2 * c[i] + c[i + 1]) / 3
        )

        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def spline_value(x_value, x, a, b, c, d):
    n = len(x)

    if x_value <= x[0]:
        i = 0
    elif x_value >= x[n - 1]:
        i = n - 2
    else:
        i = 0

        for j in range(n - 1):
            if x[j] <= x_value <= x[j + 1]:
                i = j
                break

    dx = x_value - x[i]

    value = (
        a[i]
        + b[i] * dx
        + c[i] * dx ** 2
        + d[i] * dx ** 3
    )

    return value


a, b, c, d = spline_coefficients(x, y)

print("\nКоефіцієнти c:")

for i in range(len(c)):
    print(f"c[{i}] = {c[i]:.8f}")

print("\nКоефіцієнти кубічних сплайнів:")
print("i | a | b | c | d")

for i in range(len(a)):
    print(f"{i:2d} | {a[i]:10.6f} | {b[i]:10.6f} | {c[i]:10.6f} | {d[i]:10.8f}")


xx = np.linspace(x[0], x[-1], 1000)
yy = np.zeros(len(xx))

for i in range(len(xx)):
    yy[i] = spline_value(xx[i], x, a, b, c, d)


def select_nodes(x, y, count):
    indexes = np.linspace(0, len(x) - 1, count)
    indexes = np.round(indexes)
    indexes = indexes.astype(int)

    new_x = x[indexes]
    new_y = y[indexes]

    return new_x, new_y


node_counts = [10, 15, 20]
splines = {}

for count in node_counts:
    x_nodes, y_nodes = select_nodes(x, y, count)

    a_nodes, b_nodes, c_nodes, d_nodes = spline_coefficients(
        x_nodes,
        y_nodes
    )

    yy_nodes = np.zeros(len(xx))

    for i in range(len(xx)):
        yy_nodes[i] = spline_value(
            xx[i],
            x_nodes,
            a_nodes,
            b_nodes,
            c_nodes,
            d_nodes
        )

    splines[count] = yy_nodes


plt.figure()

plt.plot(x, y, "o", label="Початкові вузли")

for count in node_counts:
    plt.plot(xx, splines[count], label=f"{count} вузлів")

plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Висота, м")
plt.title("Сплайни для 10, 15 і 20 вузлів")
plt.grid()
plt.legend()
plt.show()


plt.figure()

plt.plot(xx, yy, label="Сплайн для 21 вузла")

for count in node_counts:
    plt.plot(xx, splines[count], label=f"{count} вузлів")

plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Висота, м")
plt.title("Порівняння наближених значень")
plt.grid()
plt.legend()
plt.show()


errors = {}

for count in node_counts:
    errors[count] = np.abs(yy - splines[count])

    print(
        f"\nМаксимальна похибка для {count} вузлів:",
        np.max(errors[count])
    )

    print(
        f"Середня похибка для {count} вузлів:",
        np.mean(errors[count])
    )


plt.figure()

for count in node_counts:
    plt.plot(xx, errors[count], label=f"{count} вузлів")

plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Похибка, м")
plt.title("Похибка сплайн-інтерполяції")
plt.grid()
plt.legend()
plt.show()