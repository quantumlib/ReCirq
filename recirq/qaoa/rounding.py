import networkx as nx
import numpy as np


def round_graph(graph: nx.Graph):
    graph = graph.copy()
    new_edgeweights = {}
    for i1, i2, w in graph.edges.data('weight'):
        new_edgeweights[i1, i2] = np.sign(w)
    nx.set_edge_attributes(graph, values=new_edgeweights, name='weight')

    new_nodeweights = {}
    for i, w in graph.nodes.data('weight'):
        if w is None:
            continue
        new_nodeweights[i] = np.sign(w)
    nx.set_node_attributes(graph, values=new_nodeweights, name='weight')
    return graph
