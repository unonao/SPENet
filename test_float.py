
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix, isspmatrix
from slq_fast import slq
from spenet import slq_spenet, exact_spenet


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    def print_all(G, ks=[2, 3, 4, 5], step=10, nv=100, seed=1):
        print("normalized laplacian:")
        M = nx.normalized_laplacian_matrix(G)
        for k in ks:
            print(f"k:{k}")
            print(f"\t slq:{slq_spenet(M, k, step=step, nv=nv, seed=seed)}")
            print(f"\t exact:{exact_spenet(M, k, method='eig')}")

        print("laplacian:")
        M = nx.laplacian_matrix(G)
        for k in ks:
            print(f"k:{k}")
            print(f"\t slq:{slq_spenet(M, k, step=step, nv=nv, seed=seed)}")
            print(f"\t exact:{exact_spenet(M, k, method='eig')}")

    ks = np.arange(2.0, 4.0, 0.2)
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
