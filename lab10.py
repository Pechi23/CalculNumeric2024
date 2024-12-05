import numpy as np
import matplotlib.pyplot as plt

def hermite_interpolation(a, b, fa, fb, dfa, dfb, x_values):
    """
    Calculates the Hermite interpolation polynomial for given nodes and derivatives.

    Parameters:
    a, b: float, nodes of interpolation
    fa, fb: float, function values at nodes a and b
    dfa, dfb: float, derivatives of the function at nodes a and b
    x_values: list or numpy array of x points for which to calculate P(x)

    Returns:
    y_values: numpy array of interpolated values
    """
    y_values = []
    for x in x_values:
        term1 = ((x - b) ** 2 * (x - a)) / ((b - a) ** 2) * dfa
        term2 = ((x - a) ** 2 * (x - b)) / ((b - a) ** 2) * dfb
        term3 = ((x - a) ** 2 * (2 * (x - b) + (b - a))) / ((b - a) ** 3) * fa
        term4 = ((x - b) ** 2 * (2 * (x - a) + (a - b))) / ((b - a) ** 3) * fb
        y = term1 + term2 + term3 + term4
        y_values.append(y)
    return np.array(y_values)

# Example 1
a = np.pi / 6
b = np.pi / 2
fa = 1 / 2
fb = 1
dfa = np.sqrt(3) / 2
dfb = 0

# Define x values for evaluation
x_vals = np.linspace(a, b, 1000)

# Compute Hermite interpolation
y_vals = hermite_interpolation(a, b, fa, fb, dfa, dfb, x_vals)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Hermite Polynomial P(x)", color='blue')
plt.scatter([a, b], [fa, fb], color='red', label="Nodes")
plt.title("Hermite Interpolation Polynomial")
plt.xlabel("x")
plt.ylabel("P(x)")
plt.legend()
plt.grid(True)
plt.show()
