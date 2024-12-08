import numpy as np

def successive_approximation(F, G, x0, y0, epsilon, max_iterations):
    """
    Solves a nonlinear system of equations using the Successive Approximation Method.

    Parameters:
        F (callable): Function representing x = F(y).
        G (callable): Function representing y = G(x).
        x0 (float): Initial guess for x.
        y0 (float): Initial guess for y.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (float, float, int): Solution (x, y) and number of iterations.
    """
    x, y = x0, y0
    for iteration in range(1, max_iterations + 1):
        x_new = F(x, y)
        y_new = G(x_new)
        
        # Check for convergence
        if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon:
            return x_new, y_new, iteration
        
        x, y = x_new, y_new
    
    raise ValueError("Method did not converge within the maximum number of iterations")

# Example 1: Solving the system
# x = sqrt((x * (y + 5) -1 ) / 2)
# y = sqrt(x + 3 * log10(x))

def example_1():
    F = lambda x,y: np.sqrt(((x * (y + 5) -1 ) / 2))  # Define F(y)
    G = lambda x: np.sqrt(x + 3 * np.log10(x))  # Define G(x)
    
    x0, y0 = 3.5, 2.2  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        x, y, iterations = successive_approximation(F, G, x0, y0, epsilon, max_iterations)
        print(f"Example 1: Solution: x = {x}, y = {y}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 1:", e)


# Example 2: Symmetric system
# x = sqrt(5 - y^2)
# y = 2 / x

def example_2():
    F = lambda x,y: np.sqrt(5 - y**2)  # Define F(y)
    G = lambda x: 2 / x  # Define G(x)
    
    x0, y0 = 2, 1  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        x, y, iterations = successive_approximation(F, G, x0, y0, epsilon, max_iterations)
        print(f"Example 2: Solution: x = {x}, y = {y}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 2:", e)


if __name__ == "__main__":
    example_1()
    example_2()

import matplotlib.pyplot as plt

def visualize_convergence(F, G, x0, y0, epsilon, max_iterations):
    x_vals, y_vals = [x0], [y0]
    x, y = x0, y0
    
    for _ in range(max_iterations):
        x_new = F(x,y)
        y_new = G(x_new)
        
        x_vals.append(x_new)
        y_vals.append(y_new)
        
        if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon:
            break
        
        x, y = x_new, y_new
    
    iterations = range(len(x_vals))
    
    # Plot the convergence of x and y
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, x_vals, label="x values", marker="o")
    plt.plot(iterations, y_vals, label="y values", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Convergence of x and y")
    plt.legend()
    plt.grid()
    plt.show()


# Visualize Example 1
visualize_convergence(
    lambda x,y: np.sqrt(((x * (y + 5) -1 ) / 2)),
    lambda x: np.sqrt(x + 3 * np.log10(x)),
    x0=3.5, y0=2.2, epsilon=1e-4, max_iterations=100
)

# Visualize Example 2
visualize_convergence(
    lambda x,y: np.sqrt(5 - y**2),
    lambda x: 2 / x,
    x0=2, y0=1, epsilon=1e-4, max_iterations=100
)
