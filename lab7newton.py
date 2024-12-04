import numpy as np

def newton_nonlinear(F, J, initial_guess, epsilon, max_iterations):
    """
    Solves a nonlinear system of equations using Newton's method.
    
    Parameters:
        F (list): A list of functions F1, F2, ..., Fn representing the system of equations.
        J (callable): A function returning the Jacobian matrix of the system.
        initial_guess (list): Initial guess for the solution.
        epsilon (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations.
    
    Returns:
        list: The approximate solution.
        int: Number of iterations performed.
    """
    x = np.array(initial_guess, dtype=float)
    
    for iteration in range(max_iterations):
        Fx = np.array([f(*x) for f in F])
        Jx = J(*x)
        
        # Solve J * delta_x = -F
        delta_x = np.linalg.solve(Jx, -Fx)
        x = x + delta_x
        
        # Convergence check
        if np.linalg.norm(delta_x, ord=np.inf) < epsilon:
            return x, iteration + 1
    
    raise ValueError("Newton's method did not converge within the maximum number of iterations.")

# Example usage for the system:
# F1(x, y) = x^2 + y^2 - 10
# F2(x, y) = sqrt(x) * y - 2
# Jacobian:
# J[0,0] = 2x, J[0,1] = 2y
# J[1,0] = y / (2*sqrt(x)), J[1,1] = sqrt(x)

def example_2():
    F1 = lambda x, y: x**2 + y**2 - 10
    F2 = lambda x, y: np.sqrt(x) * y - 2
    F = [F1, F2]
    
    def J(x, y):
        return np.array([
            [2 * x, 2 * y],
            [y / (2 * np.sqrt(x)), np.sqrt(x)]
        ])
    
    initial_guess = [3, 2]
    epsilon = 1e-4
    max_iterations = 100
    
    solution, iterations = newton_nonlinear(F, J, initial_guess, epsilon, max_iterations)
    print("Solution:", solution)
    print("Iterations:", iterations)

example_2()
