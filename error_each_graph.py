"""
グラフごとの relative error を出力(数が少ないし表にするか？)
"""
import networkx as nx
import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix
from spenet import slq_spenet, exact_spenet, exact_spenet_by_path
from utils import relative_error
import os
import pandas as pd

if __name__ == "__main__":
    graphs = ["as-caida20060911", "as19991115", "Erdos02-cc", "homo-cc", "marvel-chars-cc", "musm-cc", "pgp-cc", "yeast-cc"]

    ks = [3]
    s = 10
    nv = 100
    print(f"ks:{ks}, s{s}, nv{nv}")
    for graph_name in graphs:
        path = os.path.join("data/graph-eigs-v1/", graph_name+".smat")
        print("path:", path)

        print("loading graph...")

        # loading a graph as a undirected graph
        # G = nx.read_weighted_edgelist(path, comments="%", create_using=nx.Graph)
        G = nx.Graph()
        df = pd.read_csv(path, delimiter="\s+")
        edges_list = df.values.tolist()
        G.add_weighted_edges_from(edges_list)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(f"N:{N}, M:{M}")
        print("start")
        Gtypes = ["normalized_laplacian", "laplacian", "adjacency"]
        for Gtype in Gtypes:
            print("gtype:", Gtype)
            approx = slq_spenet(G, ks, step=s, nv=nv, Gtype=Gtype)
            exact = exact_spenet_by_path(path, ks, Gtype=Gtype)
            error = relative_error(approx, exact)
            for i, k in enumerate(ks):
                print(f"k:{k}\tslq:{approx[i]},\texact:{exact[i]},\trelative error:{error[i]}")
