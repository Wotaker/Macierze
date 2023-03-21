from typing import Callable

from inverse import inverse
from numpy import ndarray as Matrix
from timeit import default_timer as timer

import numpy as np
import pickle
import lz4.frame


def is_invertable(A: Matrix):

    assert A.shape[0] == A.shape[1], "Matrix is not square!"
    assert np.isclose(np.log2(A.shape[0]) % 1, 0), "Matrix size is not a power of 2!"

    A_inv = inverse(A)
    AxA_inv = A @ A_inv

    return np.allclose(AxA_inv, np.eye(A.shape[0], A.shape[1]))


def save(file_path, arr_list):

    with lz4.frame.open(file_path, 'wb') as f:
        f.write(pickle.dumps(arr_list))


def load(file_path):

    with lz4.frame.open(file_path, 'rb') as f:
        arr_list = pickle.loads(f.read())
    
    return arr_list


def generate_invertible_matrix(n: int, condition: str = "det", verbose: bool = False):
    """
    Generates invertible matrix, based on invertibility condition.

    Parameters
    ----------
    n : int
        Generated matrix size
    condition : str, optional
        Invertibility condition, either 'det' - positive determinant or 'I' - multiplication to Identity matrix, by default "det"

    Returns
    -------
    Matrix : numpy.ndarray
    """

    while True:
        A = np.random.rand(n, n)

        if condition == "det":
            det = np.linalg.det(A)
            if not np.isclose(det, 0., atol=0.1):
                print(f"Generated {n}x{n} matrix!") if verbose else None
                return A
            else:
                print(f"det(A)={det} is to close to 0, keep searching...") if verbose else None
        
        if condition == "I":
            if is_invertable(A):
                print(f"Generated {n}x{n} matrix!") if verbose else None
                return A
            else:
                print(f"Keep searching...") if verbose else None


def measure_exec_time(func: Callable, *args):

    start = timer()
    func(*args)
    end = timer()
    
    return end - start


def generate_data(reps, max_k, seed, path, verbose: bool = False):

    np.random.seed(seed)

    matrices = []
    for k in range(1, max_k + 1):
        matrices.append([generate_invertible_matrix(2 ** k, 'det') for _ in range(reps)])
        print(f"Matrices {2**k}x{2**k} are ready") if verbose else None
    save(path, matrices)
    
    return matrices
