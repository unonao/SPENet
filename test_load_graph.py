import networkx as nx
from utils import load_graph, rodger_graphs, weighted_graphs, unweighted_graphs
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
