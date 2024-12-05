import numpy as np

def gauss_seidel_nonlinear(F, initial_guess, epsilon, max_iterations):
    """
    Solves a nonlinear system of equations using the Gauss-Seidel method.
    
    Parameters:
        F (list): A list of functions F1, F2, ..., Fn representing the system of equations.
        initial_guess (list): Initial guess for the solution.
        epsilon (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        list: The approximate solution.
        int: Number of iterations performed.
    """
    x = np.array(initial_guess, dtype=float)
    n = len(x)
    
    for iteration in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            # Update x[i] using the i-th function in F
            x[i] = F[i](*x)
        
        # Convergence check
        if np.all(np.abs(x - x_old) < epsilon):
            return x, iteration + 1
    
    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

# Example usage for the system:
# x = sqrt(0.5 * (y * z + 5z - 1))
# y = sqrt(2x + ln z)
# z = sqrt(x + 2y + 8)

def example_1():
    # Define the nonlinear system of equations as functions
    F1 = lambda x, y, z: np.sqrt(0.5 * (y * z + 5 * x - 1))
    F2 = lambda x, y, z: np.sqrt(2 * x + np.log(z))
    F3 = lambda x, y, z: np.sqrt(x * y + 2 * z + 8)
    
    F = [F1, F2, F3]
    initial_guess = [10, 10, 10]
    epsilon = 0.0000000000000001
    max_iterations = 100
    
    solution, iterations = gauss_seidel_nonlinear(F, initial_guess, epsilon, max_iterations)
    print("Solution:", solution)
    print("Iterations:", iterations)

example_1()
