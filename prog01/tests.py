from classic_strasen import *

from typing import Callable
from numpy import ndarray as Matrix

import numpy as np


def test_multiplication(multiplication_algorithm: Callable, A: Matrix | None = None, B: Matrix | None = None):

    print(f"=== Testing {multiplication_algorithm.__name__} algorithm ===")

    A = np.reshape(np.arange(1, 4 * 4 + 1), (4, 4)) if not A else A
    B = np.reshape(np.arange(17, 4 * 4 + 17), (4, 4)) if not B else B

    assert A.shape == B.shape and A.shape[0] == A.shape[1], "Matrices are not square!"
    assert np.isclose(np.log2(A.shape[0]) % 1, 0), "Matrices size is not a power of 2!"
    
    AB = multiplication_algorithm(A, B)
    AB_numpy = A @ B

    print(f"\nMatrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")
    print(f"\nMatrix AxB:\n{AB}")
    print(f"\nMatrix AxB (numpy):\n{AB_numpy}")

    assert np.array_equal(AB, AB_numpy), "Results are inconsistent! Check your algorithm!"

    print("\nTest Passed!\n")


if __name__ == "__main__":
    test_multiplication(multiply_classic)
    test_multiplication(multiply_strassen)

    print("ALL TESTS PASSED!")
