import numpy as np

def bisection_method(f, a, b, eps):
    if f(a) * f(b) >= 0:
        print("Invalid interval, the function must have different signs at a and b.")
        return None

    c = a
    while (b - a) / 2 > eps:
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c

# Example usage:
def f(x):
    return x**3 - x - 1

a, b = 1, 2
eps = 1e-4
root = bisection_method(f, a, b, eps)
print(f"Root found using Bisection Method: {root}")

def secant_method(f, a, b, eps):
    x0, x1 = a, b
    while abs(x1 - x0) > eps:
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:
            print("Division by zero in Secant Method.")
            return None
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
    return x1

# Example usage:
def f(x):
    return x**3 - x - 1

a, b = 1, 2
eps = 1e-4
root = secant_method(f, a, b, eps)
print(f"Root found using Secant Method: {root}")


