import numpy as np

def calculate_nodes(a, b, n):
    return np.array([a + i * (b - a) / n for i in range(n + 1)])

def rectangle_rule(f, a, b, n):
    x = calculate_nodes(a, b, n)
    S = sum(f(x[i]) for i in range(1, n + 1))
    T = ((b - a) / n) * S
    return T

def trapezoidal_rule(f, a, b, n):
    x = calculate_nodes(a, b, n)
    S = sum(f(x[i]) + f(x[i - 1]) for i in range(1, n + 1))
    T = ((b - a) / (2 * n)) * S
    return T

def example_1():
    f = lambda x: 1 / (x + 1)
    a, b, n = 0, 1, 100
    result = rectangle_rule(f, a, b, n)
    print(f"Example 1 \nResult: {result:.6f}, Exact: {np.log(2):.6f}")
    
def example_2():
    f = lambda x: 1 / (x + 1)
    a, b, n = 0, 1, 100
    result = trapezoidal_rule(f, a, b, n)
    print(f"Example 2 \nResult: {result:.6f}, Exact: {np.log(2):.6f}")

if __name__ == "__main__":
    example_1()
    example_2()