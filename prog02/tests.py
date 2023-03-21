from inverse import inverse
from utils import *

from typing import Callable
from numpy import ndarray as Matrix

import numpy as np
import os


TEST_MATRICES = "test_compression.dat"


def test_inverse(A: Matrix):

    print(f"=== Testing matrix inverse algorithm ===")

    assert A.shape[0] == A.shape[1], "Matrix is not square!"
    assert np.isclose(np.log2(A.shape[0]) % 1, 0), "Matrix size is not a power of 2!"
    
    A_inv = inverse(A)
    AxA_inv = A @ A_inv

    print(f"\nMatrix A:\n{A}")
    print(f"\nMatrix A^(-1):\n{A_inv}")
    print(f"\nMatrix AxA^(-1) (should be identity matrix):\n{AxA_inv}")

    assert np.allclose(AxA_inv, np.eye(A.shape[0], A.shape[1])), "Results are inconsistent! Check your algorithm!"

    print("\nTest Passed!")


def test_compression():

    matrices = [generate_invertible_matrix(2 ** k, 'I') for k in range(1, 9)]
    save(TEST_MATRICES, matrices)
    matrices_retrieved = load(TEST_MATRICES)

    for m, mr in zip(matrices, matrices_retrieved):
        assert np.all(m - mr == 0), "Retrieved matrices differ from the saved ones!"

    os.remove(TEST_MATRICES)

    print("Compression Test Passed!")


if __name__ == "__main__":

    A1 = np.array([[ 7,  1, -1,  9],
                   [-8,  1,  5, -8],
                   [-2, -9, -3, -8],
                   [ 0, -3,  4, -4]])

    A2 = np.array([[ -6,   6,  -3,   6],
                   [  7,   1,  -8,   1],
                   [ -9,   0,   4,  -4],
                   [ -8, -10,   9,   3]])

    A3 = np.array([[ -27,  -80,   80,   29,  -90,  -26,   22,    2,   59,   -2,  -54,   27,   28,  -41,  -31,  -31],
                   [  59,  -33,   58,   51,  -84,  -62,   80,   54,  -46,  -93,  -37,    1,  -85,  -54,  -66,   95],
                   [ -25,   52,  -27,   -1,  -82,   55,   -4,   48,   32,   44,  -89,  -24,   74,  -83,  -12,   64],
                   [  66,  -42,  -20,   79,   75,   25,   89,  -92, -100,   65,  -18,   36,   69,  -51,   29,  -96],
                   [ -56,  -95,   33,   -2,  -78,  -41,   67,   54,   39,    2,   -5,  -23,   57,  -31,  -70,  -13],
                   [  67,   80,   62,   -6,   35,   47,   41,  -82,  -33,  -84,  -81,  -83,  -63,  -15,   44,  -21],
                   [  53,   -4,  -71,    0,   73,   48,   13,   47,  -78,   45,   63,    5,  -84,   35,    0,   17],
                   [ -32,  -37,  -83,   68,  -82,  -80,   94,   20,  -19,   56,  -88,   90,   -4,  -71,  -35,   29],
                   [ -45,   99,  -59,  -30,   91,  -10,  -39,  -77,  -89,   75,  -10,  -64,  -86,   85,  -66,  -41],
                   [ -67,  -55,   64,   88,  -27,   -3,   26,  -98,  -93,   49,  -93,   96,  -19,  -78,  -70,   99],
                   [ -25,  -72,   10,  -99, -100,   92,  -55,  -87,   -4,   35,   33,   22,   76,  -64,   87,   -7],
                   [  45,   73,  -70,   43,  -68,  -89,  -84,  -53,   93,   48,  -67,  -86,   59,  -94,  -52,   44],
                   [  15,   49,   42,   96,   69,   72,   16,  -21,  -28,   34,  -61,   71,   82,   17,  -63,  -64],
                   [ -56,   24,   40,  -58,   90,   21,   -9,   62,  -10,   41,   35,  -68,   94,   16,   23,   25],
                   [  94, -100,   83,  -38,  -80,  -91,  -82,   14,   73,  -71,   66,  -18,  -16,   -4,  -79,  -52],
                   [   2,  -96,  -29,   88,  -42,   50,   59,   57,   58,   94,  -15,   22,  -99, -100,  -99,  -11]])

    test_inverse(A1)
    test_inverse(A2)
    test_inverse(A3)

    test_compression()

    print("\nALL TESTS PASSED!")
