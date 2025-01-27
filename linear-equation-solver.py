import numpy as np

def gauss_jordan(a, b):
    """
    Solve a system of linear equations using the Gauss-Jordan method.
    
    Parameters:
        a: Coefficient matrix
        b: Vector of constants
    
    Returns:
        Vector of solutions
    """
    n = len(b)
    # Create the augmented matrix
    M = np.hstack((a, b.reshape(-1, 1)))

    for i in range(n):
        # Find the pivot element
        max_row = np.argmax(abs(M[i:, i])) + i
        if M[max_row, i] == 0:
            raise ValueError("The system has no unique solution.")
        # Swap the pivot row with the current row
        M[[i, max_row]] = M[[max_row, i]]

        # Normalize the pivot row
        M[i] = M[i] / M[i, i]

        # Eliminate other elements in the column
        for j in range(n):
            if j != i:
                M[j] = M[j] - M[i] * M[j, i]

    # Extract the solution
    return M[:, -1]

def lu_decomposition(a):
    """
    Perform LU decomposition of a matrix into lower (L) and upper (U) triangular matrices.

    Parameters:
        a: Square matrix
    
    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
    """
    n = a.shape[0]
    L = np.zeros_like(a)
    U = np.zeros_like(a)

    for i in range(n):
        # Compute elements of U
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = a[i][k] - sum_

        # Compute elements of L
        L[i][i] = 1
        for k in range(i + 1, n):
            sum_ = sum(L[k][j] * U[j][i] for j in range(i))
            if U[i][i] == 0:
                raise ValueError("LU decomposition is not possible.")
            L[k][i] = (a[k][i] - sum_) / U[i][i]

    return L, U

def forward_substitution(L, b):
    """
    Solve the system Lx = b using forward substitution.

    Parameters:
        L: Lower triangular matrix
        b: Vector of constants

    Returns:
        Vector x (solution)
    """
    n = len(b)
    x = np.zeros_like(b, dtype=float)

    for i in range(n):
        x[i] = b[i] - sum(L[i][j] * x[j] for j in range(i))
    return x

def backward_substitution(U, y):
    """
    Solve the system Ux = y using backward substitution.

    Parameters:
        U: Upper triangular matrix
        y: Vector obtained from forward substitution

    Returns:
        Vector x (solution)
    """
    n = len(y)
    x = np.zeros_like(y, dtype=float)

    for i in reversed(range(n)):
        if U[i][i] == 0:
            raise ValueError("The system has no unique solution.")
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def solve_lu(a, b):
    """
    Solve a system of linear equations using LU decomposition.

    Parameters:
        a: Coefficient matrix
        b: Vector of constants

    Returns:
        Vector of solutions
    """
    L, U = lu_decomposition(a)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

# Example usage
if __name__ == "__main__":
    # Define the system of equations
    # Example:
    # 2x + 1y - 1z =  8
    # -3x - 1y + 2z = -11
    # -2x + 1y + 2z = -3
    a = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    # Solve using Gauss-Jordan method
    try:
        solution_gj = gauss_jordan(a, b)
        print("Solution using Gauss-Jordan:")
        print(solution_gj)
    except ValueError as e:
        print(e)

    # Solve using LU decomposition
    try:
        solution_lu = solve_lu(a, b)
        print("Solution using LU Decomposition:")
        print(solution_lu)
    except ValueError as e:
        print(e)
