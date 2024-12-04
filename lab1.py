import numpy as np
import matplotlib.pyplot as plt

def jacobi_method(A, b, x0, tol=1e-10, max_iter=100):
    n = len(b)
    x = x0.copy()
    x_prev = np.zeros_like(x)
    errors = []

    for _ in range(max_iter):
        for i in range(n):
            s = sum(A[i, j] * x_prev[j] for j in range(n) if i != j)
            x[i] = (b[i] - s) / A[i, i]
        error = np.linalg.norm(x - x_prev, ord=np.inf)
        errors.append(error)
        if error < tol:
            break
        x_prev = x.copy()
    return x, errors

A = np.array([[5, 2, 1], [2, 6, 3], [1, 2, 10]], dtype=float)
b = np.array([10, 4, -7], dtype=float)
x0 = np.zeros_like(b)

x, errors = jacobi_method(A, b, x0)

print(f"Solution: {x}")
plt.plot(errors, label='Error at each iteration')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Convergence of Jacobi Method')
plt.legend()
plt.grid()
plt.show()

def gauss_seidel_method(A, b, x0, tol=1e-10, max_iter=100):
    n = len(b)
    x = x0.copy()
    errors = []

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        if error < tol:
            break
    return x, errors

x, errors = gauss_seidel_method(A, b, x0)

print(f"Solution: {x}")
plt.plot(errors, label='Error at each iteration')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Convergence of Gauss-Seidel Method')
plt.legend()
plt.grid()
plt.show()

def tridiagonal_solver(a, b, c, d):
    n = len(d)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        cp[i] = c[i] / (b[i] - a[i - 1] * cp[i - 1])
    for i in range(1, n):
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / (b[i] - a[i - 1] * cp[i - 1])

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x

a = [1, 1, 1]  # Lower diagonal
b = [4, 4, 4, 4]  # Main diagonal
c = [1, 1, 1]  # Upper diagonal
d = [2, 2, 2, 2]  # Right-hand side

solution = tridiagonal_solver(a, b, c, d)
print(f"Solution: {solution}")
