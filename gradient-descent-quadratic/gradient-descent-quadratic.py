def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """

    x = float(x0)
    for _ in range(steps):
        f_prime_x= 2*a*x + b
        x= x - (lr * f_prime_x)

    return float(x)