import numpy as np
import numpy.linalg as LA
import os
import networkx as nx
import pandas as pd


def load_graph(filepath, is_weighted=False):
    """
        If the extention of file is `smat` or `mtx`, there is a header that should be ignored.
    """
    head_tail = os.path.split(filepath)
    (name, ext) = os.path.splitext(head_tail[-1])
    G = nx.Graph()
    if ext == ".smat" or ext == ".mtx":  # ignore header
        df = pd.read_csv(filepath, delimiter="\s+", comment="%")
        if is_weighted:
            edges_list = df.values.tolist()
            G.add_weighted_edges_from(edges_list)
        else:
            edges_list = df.iloc[:, :2].values.tolist()  # ignore last cols
            G.add_edges_from(edges_list)
    else:
        if is_weighted:
            G = nx.read_edgelist(filepath, comments="%", data=(("weight", float),), create_using=nx.Graph)
        else:
            G = nx.read_edgelist(filepath, comments="%", create_using=nx.Graph)

    return G


def random_vec(n, type="stdnorm"):
    """
    input:
        n   : vector size
    output:
        random vector that follows rademacher distribution & std normal distribution
    example:
        "randmacher" : [1,1,-1,1,-1,-1,-1,1,-1,1]
    """
    if type == "stdnorm":
        return np.random.standard_normal(n)
    else:  # randmacher
        #  from a uniform distribution over [0, 1)
        vec = np.random.rand(n)
        vec[vec < 0.5] = -1
        vec[vec >= 0.5] = 1
        return vec


def std_random_vec(n):
    """
    input:
        n   : vector size
    output:
        random vector that follows standard normal distribution
    example:
        [1,1,-1,1,-1,-1,-1,1,-1,1]
    """
    #  from a uniform distribution over [0, 1)
    vec = np.random.rand(n)
    vec[vec < 0.5] = -1
    vec[vec >= 0.5] = 1
    return vec


def normalize_vec(v):
    l2 = LA.norm(v, ord=2, axis=-1)
    if l2 == 0:
        l2 = 1
    return v/l2


def relative_error(pred, true):
    return np.abs(pred-true)/true
