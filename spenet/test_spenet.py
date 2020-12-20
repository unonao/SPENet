import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from utils import normalize_vec, random_vec, relative_error
from spenet import slq_spenet, exact_spenet

if __name__ == "__main__":
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("ks:", ks)

    print("generating graph...")
    N = 10000
    M = 100000
    G = nx.gnm_random_graph(N, M)
    print("start")
    approx = slq_spenet(G, ks, step=10, nv=100, Gtype="normalized_laplacian")
    print("Big graph: ", approx)

    G = nx.gnm_random_graph(100, 1000)
    Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
    for Gtype in Gtypes:
        print(Gtype)
        approx = slq_spenet(G, ks, step=10, nv=100, Gtype=Gtype)
        exact = exact_spenet(G, ks, Gtype=Gtype)
        error = relative_error(approx, exact)
        for i, k in enumerate(ks):
            print(f"k:{k}\tslq:{approx[i]},\t\t exact:{exact[i]},\t\t relative error:{error[i]}")
