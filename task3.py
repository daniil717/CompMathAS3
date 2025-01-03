import numpy as np

def iterative_matrix_inversion(A, B, tol=1e-5, max_iter=100):
    n = A.shape[0]
    X = B.copy()
    for iteration in range(max_iter):
        X_new = np.dot(2 * np.eye(n) - np.dot(A, X), X)
        if np.linalg.norm(X_new - X, ord=np.inf) < tol:
            return X_new
        X = X_new
    raise ValueError("Iterative method did not converge.")

A = np.array([
    [1, 10, 1],
    [2, 0, 1],
    [3, 3, 2]
], dtype=float)

B = np.array([
    [0.4, 2.4, -1.4],
    [0.14, 0.14, -0.14],
    [-0.85, -3.8, 2.8]
], dtype=float)

inverse_A_iterative = iterative_matrix_inversion(A, B)
print("\nInverse of matrix A using iterative method:")
print(inverse_A_iterative)
