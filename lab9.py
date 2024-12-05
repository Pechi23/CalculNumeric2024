import numpy as np
import matplotlib.pyplot as plt

def divided_differences(x, y):
    """
    Compute the divided difference table.
    
    Parameters:
        x (list): Interpolation nodes.
        y (list): Function values at nodes.
        
    Returns:
        np.ndarray: Divided differences table.
    """
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])

    return table


def newton_polynomial(x, y, x_eval):
    """
    Compute the Newton interpolation polynomial at given points.
    
    Parameters:
        x (list): Interpolation nodes.
        y (list): Function values at nodes.
        x_eval (array): Points where the polynomial is evaluated.
        
    Returns:
        array: Evaluated polynomial values.
    """
    table = divided_differences(x, y)
    n = len(x)
    coeffs = table[0, :]  # Extract coefficients from the first row

    def newton_term(k, x_val):
        term = 1
        for i in range(k):
            term *= (x_val - x[i])
        return term

    # Compute polynomial values
    p_values = []
    for x_val in x_eval:
        p_val = coeffs[0]
        for k in range(1, n):
            p_val += coeffs[k] * newton_term(k, x_val)
        p_values.append(p_val)

    return np.array(p_values)


def visualize_newton(x, y, x_eval, y_func=None):
    """
    Visualize the Newton interpolation polynomial.
    
    Parameters:
        x (list): Interpolation nodes.
        y (list): Function values at nodes.
        x_eval (array): Points where the polynomial is evaluated.
        y_func (callable, optional): Original function for comparison.
    """
    y_poly = newton_polynomial(x, y, x_eval)

    plt.figure(figsize=(30, 10))

    # Plot original function if available
    if y_func:
        y_original = y_func(x_eval)
        plt.plot(x_eval, y_original, label="Original Function", color="blue", linestyle="--")

    # Plot Newton polynomial
    plt.plot(x_eval, y_poly, label="Newton Polynomial", color="red")

    # Plot interpolation points
    plt.scatter(x, y, color="black", label="Data Points")

    plt.title("Newton Interpolation Polynomial")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example 1 from the problem
    x = [1, 1.1, 1.3, 1.5, 1.6]  # Nodes
    y = [1, 1.032, 1.091, 1.145, 1.17]  # Function values (f(x))
    x_eval = np.linspace(1.0, 1.6, 500)  # Points to evaluate the polynomial
    visualize_newton(x, y, x_eval)

    # Example 2 from the problem
    x = [1.00, 1.08, 1.13, 1.20, 1.27, 1.31, 1.38]
    y = [1.17520, 1.30254, 1.38631, 1.50916, 1.21730, 1.22361, 1.23740]
    x_eval = np.linspace(1.0, 1.4, 500)
    visualize_newton(x, y, x_eval)
