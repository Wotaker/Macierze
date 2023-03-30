import numpy as np
from numpy import ndarray as Matrix

from classic_strasen import multiply



# short alias
def matmul_chain(*Ms: Matrix) -> Matrix:
    result = Ms[0]
    for M in Ms[1:]:
        result = multiply(result, M)
    return result



def inverse(A: Matrix) -> Matrix:
    if A.shape[0] == 1:
        if np.isclose(A[0, 0], 0.0):
            raise ValueError('singular matrix')
        return np.array([[1 / A[0, 0]]])

    m = A.shape[0] // 2

    A11 = A[:m, :m]
    A12 = A[:m, m:]
    A21 = A[m:, :m]
    A22 = A[m:, m:]

    A11_inv = inverse(A11)

    S22 = A22 - matmul_chain(A21, A11_inv, A12)
    S22_inv = inverse(S22)

    # B = A^(-1)
    B11 = A11_inv + matmul_chain(A11_inv, A12, S22_inv, A21, A11_inv)
    B12 = matmul_chain(-A11_inv, A12, S22_inv)
    B21 = matmul_chain(-S22_inv, A21, A11_inv)
    B22 = S22_inv

    B = np.concatenate((
        np.concatenate([B11, B12], axis=1),
        np.concatenate([B21, B22], axis=1)
    ), axis=0) 

    return B
