def least_squares_regression(x, y):
    """
    Calculate the coefficients a (slope) and b (intercept) for the least squares regression line.
    :param x: List of x values
    :param y: List of y values
    :return: Tuple (a, b) where a is the slope and b is the intercept
    """
    n = len(x)
    S1 = S2 = T1 = T2 = 0

    # Step 1: Calculate sums
    for i in range(n):
        S1 += x[i]
        S2 += x[i] ** 2
        T1 += y[i]
        T2 += x[i] * y[i]

    # Step 2: Calculate d, d1, d2
    d = (n * S2) - (S1 ** 2)
    d1 = (n * T2) - (S1 * T1)
    d2 = (S2 * T1) - (S1 * T2)

    # Step 3: Calculate coefficients
    a = d1 / d
    b = d2 / d

    return a, b


def plot_regression_line(x, y, a, b):
    """
    Plot the data points and the regression line.
    :param x: List of x values
    :param y: List of y values
    :param a: Slope of the regression line
    :param b: Intercept of the regression line
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate line points
    x_line = np.linspace(min(x), max(x), 500)
    y_line = a * x_line + b

    # Plot data points
    plt.scatter(x, y, color='blue', label='Data Points')

    # Plot regression line
    plt.plot(x_line, y_line, color='red', label=f'Regression Line: y = {a:.2f}x + {b:.2f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Least Squares Regression Line')
    plt.grid(True)
    plt.show()


# Example input
x = [1, 3, 4, 6, 8, 9]
y = [1, 2, 4, 4, 5, 3]

# Perform regression
a, b = least_squares_regression(x, y)

# Print coefficients
print(f"Slope (a): {a}")
print(f"Intercept (b): {b}")

# Plot the regression line
plot_regression_line(x, y, a, b)
