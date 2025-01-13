import numpy as np

def refine_inverse(A, B, max_iterations=10, tolerance=1e-6):
    I = np.eye(A.shape[0])
    for iteration in range(max_iterations):
        E = np.dot(A, B) - I
        error_norm = np.linalg.norm(E, ord='fro')
        if error_norm < tolerance:
            print(f"Iterative method converged after {iteration} iterations.")
            return B
        B = np.dot(B, I - E)
    print(f"Iterative method reached maximum iterations with error norm {error_norm:.2e}")
    return B

def lu_factorization(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i, i] = 1
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U

def invert_matrix(A):
    L, U = lu_factorization(A)
    n = A.shape[0]

    L_inv = np.zeros_like(L)
    for i in range(n):
        L_inv[i, i] = 1 / L[i, i]
        for j in range(i):
            L_inv[i, j] = -sum(L[i, k] * L_inv[k, j] for k in range(j, i)) / L[i, i]

    U_inv = np.zeros_like(U)
    for i in range(n - 1, -1, -1):
        U_inv[i, i] = 1 / U[i, i]
        for j in range(i + 1, n):
            U_inv[i, j] = -sum(U[i, k] * U_inv[k, j] for k in range(i + 1, j + 1)) / U[i, i]
    return np.dot(U_inv, L_inv)

def power_method(A, max_iterations=1000, tolerance=1e-6):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0
    for iteration in range(max_iterations):
        vk = np.dot(A, v)
        vk = vk / np.linalg.norm(vk)
        eigenvalue_next = np.dot(vk.T, np.dot(A, vk))
        if abs(eigenvalue_next - eigenvalue) < tolerance:
            print(f"Power method converged after {iteration + 1} iterations.")
            return eigenvalue_next, vk
        v = vk
        eigenvalue = eigenvalue_next
    print("Power method reached maximum iterations without convergence.")
    return eigenvalue, v

def jacobi_method(A, tolerance=1e-6, max_iterations=100):
    n = A.shape[0]
    V = np.eye(n)
    for iteration in range(max_iterations):
        off_diagonal = np.triu(np.abs(A), k=1)
        p, q = np.unravel_index(np.argmax(off_diagonal), A.shape)
        if np.abs(A[p, q]) < tolerance:
            print(f"Jacobi method converged after {iteration + 1} iterations.")
            break
        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))
        c = np.cos(theta)
        s = np.sin(theta)
        G = np.eye(n)
        G[p, p] = c
        G[q, q] = c
        G[p, q] = s
        G[q, p] = -s
        A = G.T @ A @ G
        V = V @ G
    eigenvalues = np.diag(A)
    return eigenvalues, V

def main_function():
    A = np.array([[4, 1, 2], [1, 3, 5], [2, 5, 6]], dtype=float)

    print("Iterative Method:")
    B = np.linalg.inv(A)
    refined_B = refine_inverse(A, B)
    print(refined_B)

    print("\nLU Factorization:")
    L, U = lu_factorization(A)
    print("L:\n", L)
    print("U:\n", U)
    A_inv = invert_matrix(A)
    print("Inverse via LU:\n", A_inv)

    print("\nPower Method:")
    eigenvalue, eigenvector = power_method(A)
    print("Dominant Eigenvalue:", eigenvalue)
    print("Dominant Eigenvector:\n", eigenvector)

    print("\nJacobi Method:")
    eigenvalues, eigenvectors = jacobi_method(A)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

main_function()