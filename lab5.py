import numpy as np

def successive_approximation(phi, x0, epsilon, max_iterations):
    """
    Solves a nonlinear equation using the successive approximation method.

    Parameters:
        phi (callable): The phi(x) function.
        x0 (float): Initial guess.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (float, int): Solution and number of iterations.
    """
    x = x0
    for iteration in range(1, max_iterations + 1):
        x_new = phi(x)
        
        # Check for convergence
        if abs(x_new - x) < epsilon:
            return x_new, iteration
        
        x = x_new
    
    raise ValueError("Method did not converge within the maximum number of iterations.")

# Example 1: Solving x^2 - x - 1 = 0
def example_1():
    phi = lambda x: np.sqrt(1 + x)  # Define phi(x)
    x0 = 1.2  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = successive_approximation(phi, x0, epsilon, max_iterations)
        print(f"Example 1: Solution: x = {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 1:", e)

# Example 2: Solving x^3 - x - 1 = 0
def example_2():
    phi = lambda x: np.cbrt(1 + x)  # Define phi(x)
    x0 = 1.0  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = successive_approximation(phi, x0, epsilon, max_iterations)
        print(f"Example 2: Solution: x = {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 2:", e)


if __name__ == "__main__":
    example_1()
    example_2()

def heron_sqrt(a, x0, epsilon, max_iterations):
    """
    Approximates the square root of a number using Heron's method.

    Parameters:
        a (float): Number to find the square root of.
        x0 (float): Initial guess.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (float, int): Approximated square root and number of iterations.
    """
    x = x0
    for iteration in range(1, max_iterations + 1):
        x_new = 0.5 * (x + a / x)
        
        # Check for convergence
        if abs(x_new - x) < epsilon:
            return x_new, iteration
        
        x = x_new
    
    raise ValueError("Heron's method did not converge within the maximum number of iterations.")

# Example 3: Approximating sqrt(2), sqrt(3), sqrt(5)
def heron_examples():
    numbers = [2, 3, 5]
    initial_guesses = [1, 1.5, 2]
    epsilon_values = [1e-4, 1e-8, 1e-12]
    max_iterations = 100

    for a, x0 in zip(numbers, initial_guesses):
        for epsilon in epsilon_values:
            try:
                solution, iterations = heron_sqrt(a, x0, epsilon, max_iterations)
                print(f"sqrt({a}), epsilon = {epsilon}: x = {solution}, Iterations = {iterations}")
            except ValueError as e:
                print(f"sqrt({a}), epsilon = {epsilon}:", e)

if __name__ == "__main__":
    heron_examples()
