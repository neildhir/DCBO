from networkx.classes.multidigraph import MultiDiGraph
from networkx.linalg.graphmatrix import adjacency_matrix
from numpy import asarray, ndarray, zeros_like, zeros
from typing import Tuple


def get_emit_and_trans_adjacency_mats(G: MultiDiGraph) -> Tuple[ndarray, ndarray]:
    """
    Function to separate adjacency matrix into one for transitions and one for emissions. This function, like the paper, assumes that the within time-slice topology does not change.

    Parameters
    ----------
    G : MultiDiGraph
        DAG

    Returns
    -------
    Tuple[ndarray, ndarray]
        A tuple consisting of an emission and transition adjacency matrices.
    """

    #  Count number of nodes in each time-slice
    n = G.number_of_nodes() / G.T
    assert n % 1 == 0, (n, G.T)
    n = int(n)
    A = asarray(adjacency_matrix(G).todense())
    if G.T == 1:
        return block_diag_view(A[0:n, 0:n], G.T), None
    else:
        # return (emission adjacency matrix, transition adjacency matrix)
        return block_diag_view(A[0:n, 0:n], G.T), get_off_diagonal_trans_mat(A, n, G.T)


def block_diag_view(block_mat: ndarray, block_repeats: int) -> ndarray:
    """
    Function to get adjacency matrix but only for the emission terms.

    Parameters
    ----------
    block_mat : ndarray
        The adjacency matrix which describes the connectivity within each time-slice -- we assume the topology does not change across time.
    block_repeats : int
        How many times we repeat block_mat

    Returns
    -------
    ndarray
        Adjacency matrix which only concerns the emission connectivity.

    Notes
    -----
    This is a very fast implementation. See: https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    """
    rows, cols = block_mat.shape
    A = zeros((block_repeats * rows, block_repeats * cols), dtype=block_mat.dtype)
    for k in range(block_repeats):
        A[k * rows : (k + 1) * rows, k * cols : (k + 1) * cols] = block_mat
    return A


def get_off_diagonal_trans_mat(A: ndarray, n: int, T: int) -> ndarray:
    """
    Function to get adjacency matrix but only for the transition terms.

    Parameters
    ----------
    A : ndarray
        Original adjacency matrix
    n : int
        Number of nodes per time-slice
    T : int
        Total number of time-steps in the CBN

    Returns
    -------
    ndarray
        Adjacency matrix which only contains entries for transition edges.
    """
    B = zeros_like(A)
    #  Because we allow for non-stationarity in the transitions we have to compose the matrix like this so that changes in the matrix can be propagated.
    for t in range(T - 1):
        B[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n] = A[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n]
    return B
