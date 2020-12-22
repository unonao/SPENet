"""
グラフごとの relative error を出力(数が少ないし表にするか？)
"""
import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from spenet import slq_spenet, exact_spenet, exact_spenet_by_path
from utils import load_graph, rodger_graphs, weighted_graphs, unweighted_graphs
import os
import pandas as pd


def relative_error(pred, true):
    return np.abs(pred-true)/true


def spe_relative_error(G, ks, s, nv, Gtype="normalized_laplacian", avr_times=10, graph_path=""):
    np.random.seed(0)
    errors = np.zeros(avr_times)
    for i in range(avr_times):
        approx = slq_spenet(G, ks, step=s, nv=nv, Gtype=Gtype)
        exact = exact_spenet(G, ks, Gtype=Gtype, graph_path=graph_path)
        error = relative_error(approx, exact)
        errors[i] = error
    return errors.mean()


# for check
def print_relative_error(path, ks, s, nv, is_weighted):
    G = load_graph(path, is_weighted)
    N = G.number_of_nodes()
    M = G.number_of_edges()
    Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
    for Gtype in Gtypes:
        print("gtype:", Gtype)
        error = spe_relative_error(G, ks, s, nv, Gtype=Gtype, graph_path=path)
        for i, k in enumerate(ks):
            print(f"relative error:{error}")
    print()


def print_error_each_graph(graphs, ks, s, nv, is_weighted):
    for path in graphs:
        print("path:", path)
        print("loading graph...")
        G = load_graph(path, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(f"N:{N}, M:{M}")
        print("start")
        print_relative_error(path, ks, s, nv, is_weighted)
        Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
        for Gtype in Gtypes:
            print("gtype:", Gtype)
            approx = slq_spenet(G, ks, step=s, nv=nv, Gtype=Gtype)
            exact = exact_spenet(G, ks, Gtype=Gtype, graph_path=path)
            error = relative_error(approx, exact)
            for i, k in enumerate(ks):
                print(f"k:{k}\tslq:{approx[i]},\texact:{exact[i]},\trelative error:{error[i]}")
        print()


if __name__ == "__main__":

    ks = [4]
    s = 10
    nv = 100
    print(f"ks:{ks}, s{s}, nv{nv}")
    is_weighted = True
    print_error_each_graph(weighted_graphs, ks, s, nv, is_weighted)
    is_weighted = False
    print_error_each_graph(rodger_graphs, ks, s, nv, is_weighted)
    is_weighted = False
    print_error_each_graph(unweighted_graphs, ks, s, nv, is_weighted)
