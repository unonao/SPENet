"""
グラフごとの relative error を出力(数が少ないし表にするか？)
"""
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy
from spenet import slq_spenet, exact_spenet
from utils import load_graph, rodger_graphs, weighted_graphs, unweighted_graphs
import os
import pandas as pd


def relative_error(pred, true):
    """
        If true SPE is too small, we set -1.
    """
    errors = np.abs((pred-true)/true)
    return errors


def spe_relative_error(G, ks, s, nv, Gtype="normalized_laplacian", avg_times=10, graph_path=""):
    """
    return average relative error
    """
    np.random.seed(0)
    errors = np.zeros((avg_times, len(ks)))
    for i in range(avg_times):
        approx = slq_spenet(G, ks, step=s, nv=nv, Gtype=Gtype)
        exact = exact_spenet(G, ks, Gtype=Gtype, graph_path=graph_path)
        error = relative_error(approx, exact)
        errors[i, :] = error
    return errors.mean(axis=0)


if __name__ == "__main__":
    # for check
    def print_relative_error(path, ks, s, nv, is_weighted):
        G = load_graph(path, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
        for Gtype in Gtypes:
            print("gtype:", Gtype)
            error = spe_relative_error(G, ks, s, nv, Gtype=Gtype, graph_path=path)
            print(f"relative error:{error}")
        print()

    def print_error_each_graph(graphs, ks, s, nv, is_weighted):
        for path in graphs:
            print("path:", path)
            print("loading graph...")
            G = load_graph(path, is_weighted)
            print("is bipartite:", nx.is_bipartite(G))
            N = G.number_of_nodes()
            M = G.number_of_edges()
            print(f"N:{N}, M:{M}")
            print("start")
            Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
            for Gtype in Gtypes:
                print("gtype:", Gtype)
                np.random.seed(1)
                approx = slq_spenet(G, ks, step=s, nv=nv, Gtype=Gtype)
                exact = exact_spenet(G, ks, Gtype=Gtype, graph_path=path, method="prod")
                error = relative_error(approx, exact)
                for i, k in enumerate(ks):
                    print(f"k:{k}\tslq:{approx[i]},\texact:{exact[i]},\trelative error:{error[i]}")
            print()

    ks = [2, 3, 4, 5, 6, 7]
    s = 20
    nv = 200
    print(f"ks:{ks}, s{s}, nv{nv}")

    strange_graph = ["data/networkrepository/miscellaneous/GD96_b/GD96_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_b/GD98_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_c/GD98_c.mtx"]
    print_error_each_graph(strange_graph, ks, s, nv, False)
    strange_graph = ["data/networkrepository/miscellaneous/IG5-7/IG5-7.mtx"]
    print_error_each_graph(strange_graph, ks, s, nv, True)
    """
    is_weighted = True
    print_error_each_graph(weighted_graphs, ks, s, nv, is_weighted)
    is_weighted = False
    print_error_each_graph(rodger_graphs, ks, s, nv, is_weighted)
    is_weighted = False
    print_error_each_graph(unweighted_graphs, ks, s, nv, is_weighted)
    """
