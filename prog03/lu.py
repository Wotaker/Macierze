import numpy as np
from numpy import ndarray as Matrix
from typing import Tuple

from inverse import inverse
from classic_strasen import multiply


def lu_decomposition(A: Matrix) -> Tuple[Matrix]:
    if A.shape[0] == 1:
        return np.array([[1]]), A

    m = A.shape[0] // 2

    A11 = A[:m, :m]
    A12 = A[:m, m:]
    A21 = A[m:, :m]
    A22 = A[m:, m:]

    L11, U11 = lu_decomposition(A11)
    U11_inv = inverse(U11)
    L21 = multiply(A21, U11_inv)
    L11_inv = inverse(L11)
    U12 = multiply(L11_inv, A12)
    S = A22 - multiply(multiply(A21, U11_inv), multiply(L11_inv, A12))
    L22, U22 = lu_decomposition(S)
    L12 = U21 = np.zeros((m, m))

    L = np.concatenate((
        np.concatenate([L11, L12], axis=1),
        np.concatenate([L21, L22], axis=1)
    ), axis=0)
    U = np.concatenate((
        np.concatenate([U11, U12], axis=1),
        np.concatenate([U21, U22], axis=1)
    ), axis=0)

    return L, U


def det_from_lu(A: Matrix) -> float:
    _, U = lu_decomposition(A)
    diag = np.diag(U)
    return np.prod(diag)
