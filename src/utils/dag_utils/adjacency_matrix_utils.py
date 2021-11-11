from networkx.linalg.graphmatrix import adjacency_matrix
from numpy import asarray, zeros_like, zeros


def get_emit_and_trans_adjacency_mats(G):
    # Function to separate adjacency matrix into one for transitions and one for emissions. This function, like the paper, assumes that the within time-slice topology does not change.

    T = int(list(G.nodes())[-1].split("_")[-1]) + 1

    #  Count number of nodes in first time-slice
    n = G.number_of_nodes() / T
    assert n % 1 == 0, (n, T)
    n = int(n)
    A = asarray(adjacency_matrix(G).todense())
    if T == 1:
        return (block_diag_view(A[0:n, 0:n], T), None)
    else:
        # return (emission adjacency matrix, transition adjacency matrix)
        return (block_diag_view(A[0:n, 0:n], T), get_off_diagonal_trans_mat(A, n, T))


def block_diag_view(block_mat, block_repeats):
    # This is very fast. See: https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    rows, cols = block_mat.shape
    result = zeros((block_repeats * rows, block_repeats * cols), dtype=block_mat.dtype)
    for k in range(block_repeats):
        result[k * rows : (k + 1) * rows, k * cols : (k + 1) * cols] = block_mat
    return result


def get_off_diagonal_trans_mat(A, n, T):
    B = zeros_like(A)
    #  Because we allow for non-stationarity in the transitions we have to compose the matrix like this so that changes in the matrix can be propagated.
    for t in range(T - 1):
        B[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n] = A[t * n : (t + 1) * n, (t + 1) * n : (t + 2) * n]
    return B
