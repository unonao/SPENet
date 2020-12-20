
import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix, csc_matrix
import slq
from slq_fast import slq


def slq_spenet_naive(G, k, step=10, nv=100, Gtype="normalized_laplacian"):
    """
    input:
        G       : Networkx graph
        k       :
        step    :
        nv      : random vector number
    output:
        sum of k-th powers of eigenvalues of Network
    """
    if Gtype == "normalized_laplacian":
        L = nx.normalized_laplacian_matrix(G)
    elif Gtype == "laplacian":
        L = nx.laplacian_matrix(G)
    elif Gtype == "adjacency":
        L = nx.adjacency_matrix(G)

    def f(x): return np.power(x, k)
    return slq.slq(L.astype(np.float32), step, nv, f)


def slq_spenet(G, k, step=10, nv=100, Gtype="normalized_laplacian"):
    """
    input:
        G       : Networkx graph
        k       :
        step    :
        nv      : random vector number
    output:
        sum of k-th powers of eigenvalues of Network
    """
    if Gtype == "normalized_laplacian":
        L = nx.normalized_laplacian_matrix(G)
    elif Gtype == "laplacian":
        L = nx.laplacian_matrix(G)
    elif Gtype == "adjacency":
        L = nx.adjacency_matrix(G)

    f = [lambda x: np.power(x, k)]
    return slq(L.astype(np.float32), step, nv, f)[0, 0]


def exact_spenet(G, k, Gtype="normalized_laplacian"):
    if Gtype == "normalized_laplacian":
        L = nx.normalized_laplacian_matrix(G)
    elif Gtype == "laplacian":
        L = nx.laplacian_matrix(G)
    elif Gtype == "adjacency":
        L = nx.adjacency_matrix(G)
    e = LA.eigvals(L.astype(np.float32).A)
    return np.power(e, k).sum()


if __name__ == "__main__":
    G = nx.fast_gnp_random_graph(100, 0.2)
    Gtype = "adjacency"
    for k in range(1, 10):
        print(f"k {k},  slq:{slq_spenet(G, k=k, nv=100, Gtype=Gtype)}  , exact:{exact_spenet(G,k, Gtype=Gtype)}")
