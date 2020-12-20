import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from utils import normalize_vec, random_vec, relative_error
from spenet import slq_spenet, exact_spenet

if __name__ == "__main__":
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    gtype = "normalized_laplacian"

    path = "data/networkrepository/animal/aves-weaver-social/aves-weaver-social.edges"

    print("ks:", ks)
    print("gtype:", gtype)
    print("path:", path)

    print("loading graph...")
    # loading a graph as a undirected graph
    G = nx.read_edgelist(path, comments="%", data=(("weight", float),), create_using=nx.Graph)
    N = G.number_of_nodes()
    M = G.number_of_edges()
    print(f"N:{N}, M:{M}")
    print("start")
    Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
    for Gtype in Gtypes:
        print(Gtype)
        approx = slq_spenet(G, ks, step=10, nv=100, Gtype=Gtype)
        exact = exact_spenet(G, ks, Gtype=Gtype)
        error = relative_error(approx, exact)
        for i, k in enumerate(ks):
            print(f"k:{k}\tslq:{approx[i]},\t\t exact:{exact[i]},\t\t relative error:{error[i]}")
