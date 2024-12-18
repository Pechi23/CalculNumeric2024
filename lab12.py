import numpy as np
import matplotlib.pyplot as plt

def natural_cubic_spline(x, y, num_points=1000):
    """
    Compute natural cubic spline interpolation.

    Parameters:
    x (list or np.array): x-coordinates of the data points
    y (list or np.array): y-coordinates of the data points
    num_points (int): Number of interpolated points

    Returns:
    np.array: Interpolated x-values
    np.array: Interpolated y-values
    """
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]

    # Calculate the coefficients a, b, c, and d
    a = y[:-1]

    # Set up the tridiagonal system for M
    alpha = [0] + [
        6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i - 1]) for i in range(1, n)
    ]

    l = [2] + [h[i] + h[i - 1] for i in range(1, n)] + [2]
    mu = [0] + [h[i] / (h[i] + h[i - 1]) for i in range(1, n)]
    z = [0] * (n + 1)

    for i in range(1, n):
        l[i] -= mu[i] * mu[i - 1] / l[i - 1]
        alpha[i] -= alpha[i - 1] * mu[i - 1] / l[i - 1]

    # Solve for M (second derivatives)
    M = [0] * (n + 1)
    for i in range(n - 1, 0, -1):
        M[i] = (alpha[i] - mu[i] * M[i + 1]) / l[i]

    # Interpolate values
    interpolated_x = np.linspace(x[0], x[-1], num_points)
    interpolated_y = []

    for xi in interpolated_x:
        for i in range(n):
            if x[i] <= xi <= x[i + 1]:
                hi = h[i]
                term1 = M[i] * (x[i + 1] - xi) ** 3 / (6 * hi)
                term2 = M[i + 1] * (xi - x[i]) ** 3 / (6 * hi)
                term3 = (y[i] / hi - M[i] * hi / 6) * (x[i + 1] - xi)
                term4 = (y[i + 1] / hi - M[i + 1] * hi / 6) * (xi - x[i])
                interpolated_y.append(term1 + term2 + term3 + term4)
                break

    return interpolated_x, np.array(interpolated_y)

def akima_interpolation(x, y, num_points=1000):
    """
    Compute Akima spline interpolation.

    Parameters:
    x (list or np.array): x-coordinates of the data points
    y (list or np.array): y-coordinates of the data points
    num_points (int): Number of interpolated points

    Returns:
    np.array: Interpolated x-values
    np.array: Interpolated y-values
    """
    n = len(x)
    m = [(y[i + 1] - y[i]) / (x[i + 1] - x[i]) for i in range(n - 1)]

    def slope(i):
        if i == 0:
            return m[0]
        elif i == n - 1:
            return m[-1]
        else:
            p0 = m[i - 1]
            p1 = m[i]
            p2 = m[i]
            w1 = abs(p2 - p1)
            w2 = abs(p1 - p0)
            if w1 + w2 == 0:
                return (p0 + p1) / 2
            return (w1 * p0 + w2 * p1) / (w1 + w2)

    slopes = [slope(i) for i in range(n)]

    interpolated_x = np.linspace(x[0], x[-1], num_points)
    interpolated_y = []

    for xi in interpolated_x:
        for i in range(n - 1):
            if x[i] <= xi <= x[i + 1]:
                hi = x[i + 1] - x[i]
                t = (xi - x[i]) / hi
                term1 = slopes[i] * (x[i + 1] - xi) ** 2 / hi
                term2 = slopes[i + 1] * (xi - x[i]) ** 2 / hi
                term3 = y[i] * (x[i + 1] - xi) / hi
                term4 = y[i + 1] * (xi - x[i]) / hi
                interpolated_y.append(term1 + term2 + term3 + term4)
                break

    return interpolated_x, np.array(interpolated_y)

# Example usage:
x = [7.5, 10.5, 13, 15.5, 18, 21, 24, 27]
y = [130, 121, 128, 96, 122, 138, 114, 90]

# Natural Cubic Spline
interp_x, interp_y = natural_cubic_spline(x, y)

# Akima Interpolation
akima_x, akima_y = akima_interpolation(x, y)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data Points')
plt.plot(interp_x, interp_y, label='Natural Cubic Spline')
plt.plot(akima_x, akima_y, label='Akima Interpolation', linestyle='--')
plt.legend()
plt.title('Interpolation Methods')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()