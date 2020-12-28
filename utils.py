import numpy as np
import numpy.linalg as LA
import os
import networkx as nx
import pandas as pd

# graph path
rodger_graphs = ["data/graph-eigs-v1/as-caida20060911.smat",
                 "data/graph-eigs-v1/as19991115.smat",
                 "data/graph-eigs-v1/Erdos02-cc.smat",
                 "data/graph-eigs-v1/homo-cc.smat",
                 "data/graph-eigs-v1/marvel-chars-cc.smat",
                 "data/graph-eigs-v1/musm-cc.smat",
                 "data/graph-eigs-v1/pgp-cc.smat",
                 "data/graph-eigs-v1/yeast-cc.smat"
                 ]
unweighted_graphs = ["data/networkrepository/bio/bio-celegans/bio-celegans.mtx",
                     "data/networkrepository/bio/bio-celegans-dir/bio-celegans-dir.edges",
                     "data/networkrepository/bio/bio-diseasome/bio-diseasome.mtx",
                     "data/networkrepository/cheminformatics/ENZYMES_g295/ENZYMES_g295.edges",
                     "data/networkrepository/cheminformatics/ENZYMES_g296/ENZYMES_g296.edges",
                     "data/networkrepository/cheminformatics/ENZYMES_g297/ENZYMES_g297.edges",
                     "data/networkrepository/interaction/ia-crime-moreno/ia-crime-moreno.edges",
                     "data/networkrepository/interaction/ia-email-univ/ia-email-univ.mtx",
                     "data/networkrepository/interaction/ia-enron-only/ia-enron-only.mtx",
                     "data/networkrepository/interaction/ia-fb-messages/ia-fb-messages.mtx",
                     "data/networkrepository/interaction/ia-infect-dublin/ia-infect-dublin.mtx",
                     "data/networkrepository/interaction/ia-infect-hyper/ia-infect-hyper.mtx",
                     "data/networkrepository/miscellaneous/adjnoun/adjnoun.mtx",
                     "data/networkrepository/miscellaneous/bcspwr03/bcspwr03.mtx",
                     "data/networkrepository/miscellaneous/bibd_9_5/bibd_9_5.mtx",
                     "data/networkrepository/miscellaneous/can_144/can_144.mtx",
                     "data/networkrepository/miscellaneous/polbooks/polbooks.mtx",
                     "data/networkrepository/miscellaneous/flower_4_1/flower_4_1.mtx",
                     "data/networkrepository/miscellaneous/fs-adjnoun_adj_copperfield/fs-adjnoun_adj_copperfield.edges",
                     "data/networkrepository/miscellaneous/GD06_theory/GD06_theory.mtx",
                     "data/networkrepository/miscellaneous/GD96_b/GD96_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_b/GD98_b.mtx",
                     "data/networkrepository/miscellaneous/GD98_c/GD98_c.mtx",
                     "data/networkrepository/miscellaneous/GD99_c/GD99_c.mtx",
                     "data/networkrepository/miscellaneous/gent113/gent113.mtx"
                     ]
weighted_graphs = ["data/networkrepository/bio/bio-CE-GT/bio-CE-GT.edges",
                   "data/networkrepository/bio/bio-CE-LC/bio-CE-LC.edges",
                   "data/networkrepository/bio/bio-DM-LC/bio-DM-LC.edges",
                   "data/networkrepository/bio/bio-SC-TS/bio-SC-TS.edges",
                   "data/networkrepository/miscellaneous/eco-florida/eco-florida.edges",
                   "data/networkrepository/miscellaneous/eco-foodweb-baydry/eco-foodweb-baydry.edges",
                   "data/networkrepository/miscellaneous/gre_115/gre_115.mtx",
                   "data/networkrepository/miscellaneous/gre_185/gre_185.mtx",
                   "data/networkrepository/miscellaneous/IG5-7/IG5-7.mtx",
                   "data/networkrepository/miscellaneous/jazz/jazz.mtx",
                   "data/networkrepository/miscellaneous/misc-football/misc-football.mtx",
                   "data/networkrepository/miscellaneous/rw136/rw136.mtx",
                   "data/networkrepository/miscellaneous/TF10/TF10.mtx",
                   "data/networkrepository/miscellaneous/Trefethen_150/Trefethen_150.mtx",
                   "data/networkrepository/miscellaneous/Trefethen_200b/Trefethen_200b.mtx"
                   ]


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


"""
def exact_spenet_rodger(graph_path, ks, Gtype):
    # for rodger
    if Gtype == "normalized_laplacian":
        eig_path = graph_path + ".normalized.eigs"
    elif Gtype == "laplacian":
        eig_path = graph_path + ".laplacian.eigs"
    elif Gtype == "adjacency":
        eig_path = graph_path + ".adjacency.eigs"
    if os.path.exists(eig_path):  # for rodger
        if type(ks) == int:
            ks = [ks]
        answers = []
        e = np.loadtxt(eig_path).flatten()
        for k in ks:
            answers.append(np.power(e, k).sum())
        return answers

"""


if __name__ == "__main__":

    is_weighted = False

    for filepath in rodger_graphs:
        G = load_graph(filepath, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(filepath, N, M)

    for filepath in unweighted_graphs:
        G = load_graph(filepath, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(filepath, N, M)

    is_weighted = True
    for filepath in weighted_graphs:
        G = load_graph(filepath, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(filepath, N, M, max(dict(G.edges).items(), key=lambda x: x[1]['weight'])
              )

    print(len(weighted_graphs)+len(unweighted_graphs))
