import numpy as np

def tangent_method(f, df, x0, epsilon, max_iterations):
    """
    Solves a nonlinear equation f(x) = 0 using the tangent (Newton's) method.

    Parameters:
        f (callable): The function f(x).
        df (callable): The derivative f'(x).
        x0 (float): Initial guess.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (float, int): Solution and number of iterations.
    """
    x = x0
    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Method fails.")
        
        x_new = x - fx / dfx
        
        # Check for convergence
        if abs(x_new - x) < epsilon:
            return x_new, iteration
        
        x = x_new
    
    raise ValueError("Method did not converge within the maximum number of iterations.")

# Example 1: Solving x^3 - x - 1 = 0
def example_1():
    f = lambda x: x**3 - x - 1  # Define f(x)
    df = lambda x: 3 * x**2 - 1  # Define f'(x)
    x0 = 1.5  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = tangent_method(f, df, x0, epsilon, max_iterations)
        print(f"Example 1: Solution: x = {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 1:", e)


# Example 2: Solving 5x^2 - 5x + 1 = 0
def example_2():
    f = lambda x: 5 * x**2 - 5 * x + 1  # Define f(x)
    df = lambda x: 10 * x - 5  # Define f'(x)
    x0 = 0.2  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = tangent_method(f, df, x0, epsilon, max_iterations)
        print(f"Example 2: Solution: x = {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 2:", e)


if __name__ == "__main__":
    example_1()
    example_2()

def modified_tangent_method(f, df, d0, x0, epsilon, max_iterations):
    """
    Solves a nonlinear equation using the modified tangent method.

    Parameters:
        f (callable): The function f(x).
        df (float): The derivative f'(x) at the initial point.
        x0 (float): Initial guess.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (float, int): Solution and number of iterations.
    """
    x = x0
    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        if d0 == 0:
            raise ValueError("Derivative is zero. Method fails.")
        
        x_new = x - fx / d0
        
        # Check for convergence
        if abs(x_new - x) < epsilon:
            return x_new, iteration
        
        x = x_new
    
    raise ValueError("Method did not converge within the maximum number of iterations.")

# Example: Solving x^3 - x - 1 = 0 using the modified tangent method
def modified_example():
    f = lambda x: x**3 - x - 1  # Define f(x)
    d0 = 3 * 1.5**2 - 1  # Derivative f'(x

import matplotlib.pyplot as plt

def visualize_tangent_convergence(f, df, x0, epsilon, max_iterations):
    x = x0
    iterates = [x]

    for _ in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            break
        
        x_new = x - fx / dfx
        iterates.append(x_new)

        if abs(x_new - x) < epsilon:
            break

        x = x_new

    # Plot convergence
    plt.plot(iterates, marker="o")
    plt.title("Convergence of Tangent Method")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

# Example visualization for x^3 - x - 1 = 0
visualize_tangent_convergence(
    lambda x: x**3 - x - 1,
    lambda x: 3 * x**2 - 1,
    x0=1.5, epsilon=1e-4, max_iterations=100
)
