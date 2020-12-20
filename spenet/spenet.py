
import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix, csc_matrix
from slq_fast import slq


def slq_spenet(G, ks, step=10, nv=100, Gtype="normalized_laplacian"):
    """
    input:
        G       : Networkx graph
        ks       : list of k
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

    if type(ks) == int:
        ks = [ks]

    def make_power_function(k):
        return lambda x: np.power(x, k)
    f = [make_power_function(k) for k in ks]

    if len(ks) == 1:
        return slq(L.astype(np.float32), step, nv, f).flatten()[0]
    else:
        return slq(L.astype(np.float32), step, nv, f).flatten()


def exact_spenet(G, ks, Gtype="normalized_laplacian"):
    if Gtype == "normalized_laplacian":
        L = nx.normalized_laplacian_matrix(G)
    elif Gtype == "laplacian":
        L = nx.laplacian_matrix(G)
    elif Gtype == "adjacency":
        L = nx.adjacency_matrix(G)

    if type(ks) == int:
        ks = [ks]
    answers = []
    for k in ks:
        e = LA.eigvals(L.astype(np.float32).A)
        answers.append(np.power(e, k).sum())

    if len(ks) == 1:
        return answers[0]
    else:
        return answers


if __name__ == "__main__":
    print("generating graph...")
    N = 10000
    M = 100000
    G = nx.gnm_random_graph(N, M)
    ks = [2, 3, 4, 5]
    print("ks:", ks)
    print("start")
    approx = slq_spenet(G, ks, step=10, nv=100, Gtype="normalized_laplacian")
    print("Big graph: ", approx)

    ks = 2
    print("ks:", ks)
    print("start")
    approx = slq_spenet(G, ks, step=10, nv=100, Gtype="normalized_laplacian")
    print("Big graph: ", approx)
