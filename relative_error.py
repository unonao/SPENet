import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy
from spenet import slq_spenet, ste_spenet, exact_spenet
from utils import load_graph, rodger_graphs, weighted_graphs, unweighted_graphs
import os
import pandas as pd


def relative_error(pred, true):
    """
        If true SPE is too small, we set -1.
    """
    errors = np.abs((pred-true)/true)
    return errors


if __name__ == "__main__":

    # print strange relative errors

    def print_spenet_rel_error(path, is_weighted=True,  ks=[2, 3, 4, 5], step=10, nv=100, seed=1, method="prod"):
        G = load_graph(path, is_weighted=False)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"path:{path}, n:{n}, m:{m}")

        print("normalized laplacian:")
        M = nx.normalized_laplacian_matrix(G)
        for k in ks:
            ste = ste_spenet(M, k, nv=nv, seed=seed)
            slq = slq_spenet(M, k, step=step, nv=nv, seed=seed)
            exact = exact_spenet(M, k, method=method)
            print(f"k:{k}\t ste:{ste}({relative_error(ste, exact)})\t slq:{slq}({relative_error(ste, exact)})\t exact:{exact}")

        print("laplacian:")
        M = nx.laplacian_matrix(G)
        for k in ks:
            ste = ste_spenet(M, k, nv=nv, seed=seed)
            slq = slq_spenet(M, k, step=step, nv=nv, seed=seed)
            exact = exact_spenet(M, k, method=method)
            print(f"k:{k}\t ste:{ste}({relative_error(ste, exact)})\t slq:{slq}({relative_error(ste, exact)})\t exact:{exact}")

        print("adjacency:")
        M = nx.adjacency_matrix(G)
        for k in ks:
            ste = ste_spenet(M, k, nv=nv, seed=seed)
            slq = slq_spenet(M, k, step=step, nv=nv, seed=seed)
            exact = exact_spenet(M, k, method=method)
            print(f"k:{k}\t ste:{ste}({relative_error(ste, exact)})\t slq:{slq}({relative_error(ste, exact)})\t exact:{exact}")

    strange_graph = ["data/networkrepository/miscellaneous/GD96_b/GD96_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_b/GD98_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_c/GD98_c.mtx"]
    is_weighted = False
    for path in strange_graph:
        print_spenet_rel_error(path,  is_weighted)

    strange_graph = ["data/networkrepository/miscellaneous/IG5-7/IG5-7.mtx"]
    is_weighted = True
    for path in strange_graph:
        print_spenet_rel_error(path,  is_weighted)
