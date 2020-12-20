import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from utils import normalize_vec, random_vec, relative_error
from spenet import slq_spenet, exact_spenet

if __name__ == "__main__":
    print("generating graph...")
    N = 10000
    M = 100000
    G = nx.gnm_random_graph(N, M)
    print("start")
    approx = slq_spenet(G, k=3, step=10, nv=100, Gtype="normalized_laplacian")
    print("Big graph: ", approx)

    G = nx.gnm_random_graph(100, 1000)
    Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
    for Gtype in Gtypes:
        print(Gtype)
        for k in range(2, 10):
            approx = slq_spenet(G, k=k, step=10, nv=100, Gtype=Gtype)
            exact = exact_spenet(G, k, Gtype=Gtype)
            print(f"k {k},\t slq:{approx},\t exact:{exact},\t relative error:{relative_error(approx, exact)}")
