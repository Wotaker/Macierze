from typing import Callable

from inverse import inverse
from numpy import ndarray as Matrix
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import lz4.frame


# Funkcja anonimowa obliczająca liczbę operacji dla k
inv_flo = lambda k: 17 * pow(2, k - 1) + sum([pow(2, k - i) * (46 * pow(7, i - 1) + 4 * pow(4, i - 1) - 36 * pow(2, i - 1) + 1) for i in range(2, k + 1)])


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


def plot_results(sizes, times_mean, flos):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(8, 12)

    # Log-plot
    ax1_flo = ax1.twinx()
    ax1.scatter(sizes, times_mean, color='tab:blue', label="t")
    ax1_flo.scatter(sizes, flos, color='tab:orange', label="FLO", marker='x')
    ax1.set_ylabel("Czas obliczeń t [s]")
    ax1_flo.set_ylabel("Liczba operacji zmiennoprzecinkowych FLO")
    ax1.set_xlabel("Rozmiar macierzy")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1_flo.set_yscale('log')

    for i, size in enumerate(list(sizes)):
        ax1.annotate(f"{size}x{size}", (sizes[i], times_mean[i]), fontsize=8, ha='center')
        ax2.annotate(f"{size}x{size}", (sizes[i], times_mean[i]), fontsize=8, ha='center')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_flo.get_legend_handles_labels()
    ax1_flo.legend(lines + lines2, labels + labels2, loc='upper left')

    ax1.set_title("Złożoność rekurencyjnego odwracania macierzy")

    # Normal-plot
    ax2_flo = ax2.twinx()
    ax2.scatter(sizes, times_mean, color='tab:blue', label="t")
    ax2_flo.scatter(sizes, flos, color='tab:orange', label="FLO", marker='x')
    ax2.set_ylabel("Czas obliczeń t [s]")
    ax2_flo.set_ylabel("Liczba operacji zmiennoprzecinkowych FLO")
    ax2.set_xlabel("Rozmiar macierzy")
    ax2.set_xscale('log')

    for i, size in enumerate(list(sizes)):
        ax2.annotate(f"{size}x{size}", (sizes[i], times_mean[i]), fontsize=8, ha='center')
        ax2.annotate(f"{size}x{size}", (sizes[i], times_mean[i]), fontsize=8, ha='center')

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_flo.get_legend_handles_labels()
    ax2_flo.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()
