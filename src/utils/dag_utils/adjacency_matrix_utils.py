from networkx.linalg.graphmatrix import adjacency_matrix
from numpy import asarray, ndarray, zeros_like, zeros
from typing import Tuple


def get_emit_and_trans_adjacency_mats(G) -> Tuple[ndarray, ndarray]:
    # Function to separate adjacency matrix into one for transitions and one for emissions. This function, like the paper, assumes that the within time-slice topology does not change.

    #  Count number of nodes in first time-slice
    n = G.number_of_nodes() / G.T
    assert n % 1 == 0, (n, G.T)
    n = int(n)
    A = asarray(adjacency_matrix(G).todense())
    if G.T == 1:
        return block_diag_view(A[0:n, 0:n], G.T), None
    else:
        # return (emission adjacency matrix, transition adjacency matrix)
        return block_diag_view(A[0:n, 0:n], G.T), get_off_diagonal_trans_mat(A, n, G.T)


def block_diag_view(block_mat, block_repeats) -> ndarray:
    # This is very fast. See: https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    rows, cols = block_mat.shape
    A = zeros((block_repeats * rows, block_repeats * cols), dtype=block_mat.dtype)
    for k in range(block_repeats):
        A[k * rows : (k + 1) * rows, k * cols : (k + 1) * cols] = block_mat
    return A


def get_off_diagonal_trans_mat(A, n, T) -> ndarray:
    B = zeros_like(A)
    #  Because we allow for non-stationarity in the transitions we have to compose the matrix like this so that changes in the matrix can be propagated.
    for t in range(T - 1):
        B[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n] = A[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n]
    return B
