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

# Example 1: Solving x^4 - x - 1 = 0
def example_1():
    phi = lambda x: (1 + x) ** 0.25  # Define phi(x) = (1 + x)^(1/4)
    x0 = 1.2  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = successive_approximation(phi, x0, epsilon, max_iterations)
        print(f"Example 1: Solution: x = {solution:.6f}, Iterations: {iterations}")
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

# Example 3: Solving x^5 - 5x + 1 = 0
def example_3():
    phi = lambda x: (1 / 5) * (x ** 5 + 1)  # Define phi(x) = (1/5) * (x^5 + 1)
    x0 = 0.5  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = successive_approximation(phi, x0, epsilon, max_iterations)
        print(f"Example 3: Solution: x = {solution:.6f}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 3:", e)
        
# Example 4: Solving x = sin(x) + 0.25
def example_4():
    phi = lambda x: np.sin(x) + 0.25  # Define phi(x) = sin(x) + 0.25
    x0 = 1.1781  # Initial guess
    epsilon = 1e-4  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    try:
        solution, iterations = successive_approximation(phi, x0, epsilon, max_iterations)
        print(f"Example 4: Solution: x = {solution:.6f}, Iterations: {iterations}")
    except ValueError as e:
        print("Example 4:", e)

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()

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

# Example 1: Approximating sqrt(2), sqrt(3), sqrt(5)
def heron_examples1():
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
                
def heron_example2():
    """
    Approximates sqrt(1/2) using Heron's method for different values of epsilon.
    """
    a = 1 / 2  # The value for which we approximate the square root
    x0 = 1  # Initial guess
    epsilon_values = [1e-4, 1e-8, 1e-12]  # Different convergence tolerances
    max_iterations = 100  # Maximum number of iterations

    for epsilon in epsilon_values:
        try:
            solution, iterations = heron_sqrt(a, x0, epsilon, max_iterations)
            print(f"sqrt(1/2), epsilon = {epsilon}: x = {solution:.12f}, Iterations = {iterations}")
        except ValueError as e:
            print(f"sqrt(1/2), epsilon = {epsilon}: {e}")
            
if __name__ == "__main__":
    heron_examples1()
    heron_example2()
