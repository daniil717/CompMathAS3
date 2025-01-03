import numpy as np

def jacobi_eigenvalues(A, tol=1e-5, max_iter=100):
    n = A.shape[0]
    V = np.eye(n)
    for iteration in range(max_iter):
        max_val = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) > abs(max_val):
                    max_val = A[i, j]
                    p, q = i, j

        if abs(max_val) < tol:
            return np.diag(A), V

        theta = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])
        cos, sin = np.cos(theta), np.sin(theta)

        J = np.eye(n)
        J[p, p] = cos
        J[q, q] = cos
        J[p, q] = sin
        J[q, p] = -sin

        A = np.dot(J.T, np.dot(A, J))
        V = np.dot(V, J)

    raise ValueError("Jacobi's method did not converge.")

A = np.array([
    [1, np.sqrt(2), 2],
    [np.sqrt(2), 3, np.sqrt(2)],
    [2, np.sqrt(2), 1]
], dtype=float)

eigenvalues, eigenvectors = jacobi_eigenvalues(A)
print("\nEigenvalues and eigenvectors using Jacobi's method:")
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)