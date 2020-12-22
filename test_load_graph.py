import networkx as nx
from utils import load_graph
if __name__ == "__main__":
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
                       "data/networkrepository/miscellaneous/bcsstm04/bcsstm04.mtx",
                       "data/networkrepository/miscellaneous/ck104/ck104.mtx",
                       "data/networkrepository/miscellaneous/eco-florida/eco-florida.edges",
                       "data/networkrepository/miscellaneous/eco-foodweb-baydry/eco-foodweb-baydry.edges",
                       "data/networkrepository/miscellaneous/gre_115/gre_115.mtx",
                       "data/networkrepository/miscellaneous/IG5-7/IG5-7.mtx",
                       "data/networkrepository/miscellaneous/impcol_c/impcol_c.mtx",
                       "data/networkrepository/miscellaneous/lp_adlittle/lp_adlittle.mtx",
                       "data/networkrepository/miscellaneous/lp_blend/lp_blend.mtx",
                       "data/networkrepository/miscellaneous/lpi_forest6/lpi_forest6.mtx",
                       "data/networkrepository/miscellaneous/misc-football/misc-football.mtx",
                       "data/networkrepository/miscellaneous/n2c6-b1/n2c6-b1.mtx",
                       "data/networkrepository/miscellaneous/n3c5-b2/n3c5-b2.mtx",
                       "data/networkrepository/miscellaneous/n3c6-b1/n3c6-b1.mtx",
                       "data/networkrepository/miscellaneous/n4c5-b1/n4c5-b1.mtx",
                       "data/networkrepository/miscellaneous/nos4/nos4.mtx",
                       "data/networkrepository/miscellaneous/olm100/olm100.mtx",
                       "data/networkrepository/miscellaneous/pivtol/pivtol.mtx",
                       "data/networkrepository/miscellaneous/rajat11/rajat11.mtx",
                       "data/networkrepository/miscellaneous/robot/robot.mtx",
                       "data/networkrepository/miscellaneous/rotor1/rotor1.mtx",
                       "data/networkrepository/miscellaneous/rw136/rw136.mtx",
                       "data/networkrepository/miscellaneous/TF10/TF10.mtx",
                       "data/networkrepository/miscellaneous/Trefethen_150/Trefethen_150.mtx"
                       ]
    is_weighted = True
    for filepath in weighted_graphs:
        G = load_graph(filepath, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(filepath, N, M, max(dict(G.edges).items(), key=lambda x: x[1]['weight'])
              )

    is_weighted = False
    for filepath in unweighted_graphs:
        G = load_graph(filepath, is_weighted)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        print(filepath, N, M)

    print(len(weighted_graphs)+len(unweighted_graphs))
