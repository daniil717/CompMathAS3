import numpy as np

def power_method(A, x, tol=1e-5, max_iter=100):
    n = len(x)
    for iteration in range(max_iter):
        x_new = np.dot(A, x)
        x_new_norm = np.linalg.norm(x_new, ord=np.inf)
        x_new = x_new / x_new_norm
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            eigenvalue = x_new_norm
            return eigenvalue, x_new
        x = x_new
    raise ValueError("Power method did not converge.")

A = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
], dtype=float)

x = np.array([1, 0, 0], dtype=float)
eigenvalue, eigenvector = power_method(A, x)
print("\nLargest eigenvalue and corresponding eigenvector using power method:")
print("Eigenvalue:", eigenvalue)
print("Eigenvector:", eigenvector)
