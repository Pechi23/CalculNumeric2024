import numpy as np

def jacobi_method(A, b, x0, epsilon, max_iterations):
    """
    Solves the system of linear equations Ax = b using the Jacobi method.

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Constant vector.
        x0 (ndarray): Initial guess for the solution.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (ndarray, int): Solution vector and number of iterations.
    """
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)

    for k in range(max_iterations):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            return x_new, k + 1
        
        x = x_new.copy()

    raise ValueError("Jacobi method did not converge within the maximum number of iterations.")

# Example: Solve the system using the Jacobi method
def example_jacobi():
    A = np.array([
        [5, 1, 1],
        [1, 6, 4],
        [1, 1, 10]
    ], dtype=float)
    b = np.array([10, 4, -7], dtype=float)
    x0 = np.zeros_like(b)
    epsilon = 1e-10
    max_iterations = 100

    try:
        solution, iterations = jacobi_method(A, b, x0, epsilon, max_iterations)
        print(f"Jacobi Method: Solution: {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Jacobi Method:", e)

example_jacobi()

def gauss_seidel_method(A, b, x0, epsilon, max_iterations):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Constant vector.
        x0 (ndarray): Initial guess for the solution.
        epsilon (float): Convergence tolerance.
        max_iterations (int): Maximum number of iterations.

    Returns:
        (ndarray, int): Solution vector and number of iterations.
    """
    n = len(b)
    x = x0.copy()

    for k in range(max_iterations):
        x_old = x.copy()

        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))  # Lower triangular part
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))  # Upper triangular part
            x[i] = (b[i] - s1 - s2) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x, k + 1

    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

# Example: Solve the system using the Gauss-Seidel method
def example_gauss_seidel():
    A = np.array([
        [5, 1, 1],
        [1, 6, 4],
        [1, 1, 10]
    ], dtype=float)
    b = np.array([10, 4, -7], dtype=float)
    x0 = np.zeros_like(b)
    epsilon = 1e-10
    max_iterations = 100

    try:
        solution, iterations = gauss_seidel_method(A, b, x0, epsilon, max_iterations)
        print(f"Gauss-Seidel Method: Solution: {solution}, Iterations: {iterations}")
    except ValueError as e:
        print("Gauss-Seidel Method:", e)

example_gauss_seidel()
