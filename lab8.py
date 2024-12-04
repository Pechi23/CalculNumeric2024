import math
import numpy as np
import matplotlib.pyplot as plt

def taylor_polynomial(f_derivatives, a, u, n):
    """
    Approximates the value of a function using Taylor's Polynomial.
    
    Parameters:
        f_derivatives (list): List of derivatives of the function at the point 'a', up to n-th order.
        a (float): The node of interpolation.
        u (float): The point where the function is approximated.
        n (int): The degree of the Taylor polynomial.

    Returns:
        float: The approximated value of the function at point u.
    """
    taylor_value = f_derivatives[0]  # Start with f(a)
    p = 1  # Factorial term
    for k in range(1, n + 1):
        p *= k
        taylor_value += ((u - a) ** k / p) * f_derivatives[k]
    return taylor_value


def visualize_taylor(f, f_derivatives, a, n, x_range):
    """
    Visualizes the Taylor polynomial approximation alongside the original function.

    Parameters:
        f (callable): The original function.
        f_derivatives (list): List of derivatives of the function at the point 'a', up to n-th order.
        a (float): The node of interpolation.
        n (int): The degree of the Taylor polynomial.
        x_range (tuple): Range of x values for the plot (start, end).
    """
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], 1000)
    # Calculate original function values
    y_original = [f(xi) for xi in x]
    # Calculate Taylor polynomial values
    y_taylor = [taylor_polynomial(f_derivatives, a, xi, n) for xi in x]

    # Plot original function
    plt.plot(x, y_original, label="Original Function", color="blue")
    # Plot Taylor polynomial
    plt.plot(x, y_taylor, label=f"Taylor Polynomial (n={n})", linestyle="--", color="red")
    # Highlight the interpolation point
    plt.scatter([a], [f(a)], color="black", label="Point of Approximation (a)")

    # Add labels, legend, and title
    plt.title("Taylor Polynomial Approximation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
    plt.axvline(a, color='gray', linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid()
    plt.show()


# Example 1: e^x visualization
def example_1():
    f = math.exp
    f_derivatives = [1, 1, 1, 1, 1, 1, 1, 1]  # Derivatives of e^x at x=0
    a = 0
    n = 7
    x_range = (-2, 2)
    visualize_taylor(f, f_derivatives, a, n, x_range)


# Example 2: sin(x) visualization
def example_2():
    f = math.sin
    f_derivatives = [0, 1, 0, -1, 0, 1, 0, -1]  # Derivatives of sin(x) at x=0
    a = 0
    n = 7
    x_range = (-2 * math.pi, 2 * math.pi)
    visualize_taylor(f, f_derivatives, a, n, x_range)


# Example 3: cos(x) visualization
def example_3():
    f = math.cos
    f_derivatives = [1, 0, -1, 0, 1, 0, -1, 0, 1]  # Derivatives of cos(x) at x=0
    a = 0
    n = 8
    x_range = (-2 * math.pi, 2 * math.pi)
    visualize_taylor(f, f_derivatives, a, n, x_range)


if __name__ == "__main__":
    # example_1()
    # example_2()
    example_3()
