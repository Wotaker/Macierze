from numpy import ndarray as Matrix

import numpy as np


def multiply_classic(A: Matrix, B: Matrix):
    """
    Multiplies `A` times `B` with a classic algorithm, where 
    `A` is an `m x n` matrix and `B` is an `n x l` matrix.
    """

    m, n, l = A.shape[0], A.shape[1], B.shape[1]
    multiply = np.empty((m, l))
    sum = 0
    for i in range(m):                      # rows in multiply
        for j in range(l):                  # columns in multiply
            for k in range(n):              # columns in A and rows in B
                sum += A[i, k] * B[k, j]
            multiply[i, j] = sum
            sum = 0

    return multiply


def multiply_strassen(A: Matrix, B: Matrix):
    """
    Multiplies `A` times `B` with a Strassen algorithm, where
    both `A` and `B` are square `n x n` matrices.
    """

    def strassen(A: Matrix, B: Matrix, n: int):

        if n == 1:
            return A * B
        
        m = n // 2

        A11 = A[:m, :m]
        A12 = A[:m, m:]
        A21 = A[m:, :m]
        A22 = A[m:, m:]

        B11 = B[:m, :m]
        B12 = B[:m, m:]
        B21 = B[m:, :m]
        B22 = B[m:, m:]

        P1 = strassen(A11 + A22, B11 + B22, m)
        P2 = strassen(A21 + A22, B11, m)
        P3 = strassen(A11, B12 - B22, m)
        P4 = strassen(A22, B21 - B11, m)
        P5 = strassen(A11 + A12, B22, m)
        P6 = strassen(A21 - A11, B11 + B12, m)
        P7 = strassen(A12 - A22, B21 + B22, m)

        C = np.concatenate((
            np.concatenate((P1 + P4 - P5 + P7, P3 + P5), axis=1),
            np.concatenate((P2 + P4, P1 - P2 + P3 + P6), axis=1)
        ), axis=0)

        return C

    return strassen(A, B, A.shape[0])


def multiply_strassen_with_classic(A: Matrix, B: Matrix, size_classic: int = 1):
    """
    Multiplies `A` times `B` with a Strassen algorithm, where
    both `A` and `B` are square `n x n` matrices.
    If submatrix size is less or equal than `size_classic` uses `multiply_classic`.
    """

    def strassen(A: Matrix, B: Matrix, n: int):

        if n == 1:
            return A * B
        elif n <= size_classic:
            return multiply_classic(A, B)
        
        m = n // 2

        A11 = A[:m, :m]
        A12 = A[:m, m:]
        A21 = A[m:, :m]
        A22 = A[m:, m:]

        B11 = B[:m, :m]
        B12 = B[:m, m:]
        B21 = B[m:, :m]
        B22 = B[m:, m:]

        P1 = strassen(A11 + A22, B11 + B22, m)
        P2 = strassen(A21 + A22, B11, m)
        P3 = strassen(A11, B12 - B22, m)
        P4 = strassen(A22, B21 - B11, m)
        P5 = strassen(A11 + A12, B22, m)
        P6 = strassen(A21 - A11, B11 + B12, m)
        P7 = strassen(A12 - A22, B21 + B22, m)

        C = np.concatenate((
            np.concatenate((P1 + P4 - P5 + P7, P3 + P5), axis=1),
            np.concatenate((P2 + P4, P1 - P2 + P3 + P6), axis=1)
        ), axis=0)

        return C

    return strassen(A, B, A.shape[0])
