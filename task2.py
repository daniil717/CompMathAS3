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