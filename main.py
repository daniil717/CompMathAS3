import numpy as np

def lu_factorization_inverse(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    inv_A = np.zeros_like(A, dtype=float)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = np.linalg.solve(L, e)
        inv_A[:, i] = np.linalg.solve(U, y)

    return inv_A

A = np.array([
    [50, 107, 36],
    [35, 54, 20],
    [31, 66, 21]
], dtype=float)

inverse_A = lu_factorization_inverse(A)
print("Inverse of matrix A using LU factorization:")
print(inverse_A)

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
