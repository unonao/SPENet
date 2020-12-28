
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix
from slq_fast import slq


def ste_spenet(M, k=3, nv=100, seed=None):
    """
    Args:
        M (dense or sparse matrix)  : real symetric matrix
        k (int)                     : integer index of power
        nv (int)                    : random vector number
        seed (int, optional)                    : random seed
    Returns:
        float: approximate sum of k-th powers of eigenvalues of Network
    """
    if seed is not None:
        np.random.seed(seed)

    M = csc_matrix(M)

    n = M.shape[0]
    # Initialize random vectors in columns (n x nv).
    start_vectors = np.random.randn(n, nv).astype(np.float64)
    np.divide(start_vectors, np.linalg.norm(start_vectors, axis=0),
              out=start_vectors)  # Normalize each column.

    vs = start_vectors.T
    for j in range(k):
        vs = vs @ M
    vs = np.einsum('ji,ij->j', vs, start_vectors)
    return n/nv * vs.sum()


def slq_spenet(M, k=3, step=10, nv=100, seed=None):
    """
    Args:
        M (dense or sparse matrix)  : real symetric matrix
        k (int or float, optional)  : index of power
        step (int, optional)        : step size of Lanczos algorithm
        nv (int, optional)          : random vector number
        seed (int, optional)        : random seed
    Returns:
        float: approximate sum of k-th powers of eigenvalues of Network
    """
    if seed is not None:
        np.random.seed(seed)

    def make_power_function(k):
        return lambda x: np.power(x, k)
    f = [make_power_function(k)]

    return slq(M, step, nv, f).flatten()[0]


def exact_spenet(M, k=3, method="eig"):
    """
    Args:
        M (dense or sparse matrix)      : real symetric matrix
        k (int or float, optional)      : index of power
        method (:obj:`str`, optional)   : random vector number
    Returns:
        float: exact sum of k-th powers of eigenvalues of Network
    Examples:
        >>> exact_spenet(np.array([[1,0],[0,1]]), k=3, method="prod")
        2
        >>> exact_spenet(np.array([[1,0],[0,2]]), k=2, method="prod")
        5
    """

    if isspmatrix(M):
        M = M.todense()

    if method == "eig":
        e = scipy.linalg.eigvalsh(M)
        return np.power(e, k).sum()
    elif method == "prod":
        A = M**k
        return A.trace().sum()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    def print_all(G, ks=[2, 3, 4, 5], step=10, nv=100, seed=1):
        print("normalized laplacian:")
        M = nx.normalized_laplacian_matrix(G)
        for k in ks:
            print(f"k:{k}")
            print(f"\t ste:{ste_spenet(M, k, nv=nv, seed=seed)}")
            print(f"\t slq:{slq_spenet(M, k, step=step, nv=nv, seed=seed)}")
            print(f"\t exact:{exact_spenet(M, k)}")

        print("laplacian:")
        M = nx.laplacian_matrix(G)
        for k in ks:
            print(f"k:{k}")
            print(f"\t ste:{ste_spenet(M, k, nv=nv, seed=seed)}")
            print(f"\t slq:{slq_spenet(M, k, step=step, nv=nv, seed=seed)}")
            print(f"\t exact:{exact_spenet(M, k)}")

        print("adjacency:")
        M = nx.adjacency_matrix(G)
        for k in ks:
            print(f"k:{k}")
            print(f"\t ste:{ste_spenet(M, k, nv=nv, seed=seed)}")
            print(f"\t slq:{slq_spenet(M, k, step=step, nv=nv, seed=seed)}")
            print(f"\t exact:{exact_spenet(M, k)}")

    ks = [2, 3, 4, 5]
    step = 10
    nv = 100
    seed = 1

    # example: generated graph
    n = 100
    m = 1000
    print("generating graph...")
    G = nx.gnm_random_graph(n, m)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"n:{n}, m:{m}")
    print_all(G, ks=ks, step=step, nv=nv, seed=seed)

    # example: real-world network
    print("real-world network")
    path = "data/networkrepository/bio/bio-celegans/bio-celegans.mtx"
    from utils import load_graph
    G = load_graph(path, is_weighted=False)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"n:{n}, m:{m}")
    print_all(G, ks=ks, step=step, nv=nv, seed=seed)
