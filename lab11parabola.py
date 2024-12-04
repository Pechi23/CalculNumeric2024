def parabola_regression(x, y):
    """
    Calculate the coefficients a, b, c for the parabola regression equation: y = ax^2 + bx + c.
    :param x: List of x values
    :param y: List of y values
    :return: Tuple (a, b, c) where a, b, c are the coefficients of the parabola
    """
    n = len(x)
    S1 = S2 = S3 = S4 = T1 = T2 = T3 = 0

    # Step 1: Calculate sums
    for i in range(n):
        S1 += x[i]
        S2 += x[i] ** 2
        S3 += x[i] ** 3
        S4 += x[i] ** 4
        T1 += y[i]
        T2 += x[i] * y[i]
        T3 += x[i] ** 2 * y[i]

    # Step 2: Calculate d, d1, d2, d3
    d = (n * S2 * S4 + 2 * S1 * S2 * S3 - S1 ** 2 * S4 - n * S3 ** 2 - S2 ** 3)
    d1 = (T1 * S2 * S4 + S1 * S3 * T2 + S1 * S2 * T3 - T1 * S3 ** 2 - S2 ** 2 * T3 - S1 ** 2 * T2)
    d2 = (n * T2 * S4 + S1 * S3 * T1 + S2 * T3 * n - S1 ** 2 * T3 - S2 * T2 * S3 - T1 * S3 ** 2)
    d3 = (n * S2 * T3 + S1 * T2 * S3 + S1 * S2 * T1 - T1 * S2 ** 2 - S1 ** 2 * T3 - S3 ** 2 * T2)

    # Step 3: Calculate coefficients
    a = d1 / d
    b = d2 / d
    c = d3 / d

    return a, b, c


def plot_parabola_regression(x, y, a, b, c):
    """
    Plot the data points and the parabola regression curve.
    :param x: List of x values
    :param y: List of y values
    :param a: Coefficient of x^2
    :param b: Coefficient of x
    :param c: Constant term
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate parabola points
    x_parabola = np.linspace(min(x), max(x), 500)
    y_parabola = a * x_parabola ** 2 + b * x_parabola + c

    # Plot data points
    plt.scatter(x, y, color='blue', label='Data Points')

    # Plot regression parabola
    plt.plot(x_parabola, y_parabola, color='red', label=f'Parabola: y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Parabola Regression')
    plt.grid(True)
    plt.show()


# Example input
x = [-3, -2, -1, 0, 1, 2, 3]
y = [6, 4, 1, 2, 3, 8, 11]

# Perform parabola regression
a, b, c = parabola_regression(x, y)

# Print coefficients
print(f"Coefficient a: {a}")
print(f"Coefficient b: {b}")
print(f"Coefficient c: {c}")

# Plot the parabola
plot_parabola_regression(x, y, a, b, c)
