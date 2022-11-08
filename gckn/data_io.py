import os
import re
import statistics
import numpy as np
import networkx as nx


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0

        self.max_neighbor = 0
        self.mean_neighbor = 0


def get_motif(mask, path_indices, graph, max_component=True, eps=0.1):
    if not isinstance(mask, list):
        mask = [mask]
    if not isinstance(path_indices, list):
        path_indices = [path_indices]
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    n = len(g.nodes())
    for node in graph.nodes():
        g.nodes[node]['tag'] = graph.nodes[node]['tag']

    # edge_list = []
    adj = np.zeros((n, n))
    for m, path in zip(mask, path_indices):
        if len(path[0]) <= 1:
            continue
        for i in range(len(m)):
            if m[i] > eps:
                p = path[i]
                for j in range(len(p) - 1):
                    adj[p[j], p[j+1]] += m[i]
    adj /= np.max(adj)
    edge_list = [(i, j, adj[i, j]) for i in range(n) for j in range(n) if adj[i, j] > eps]
    # print(adj)
    g.add_weighted_edges_from(edge_list)

    if max_component:
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    else:
        # remove zero degree nodes
        g.remove_nodes_from(list(nx.isolates(g)))
    return g


def log_graph(
    graph,
    outdir,
    filename,
    identify_self=False,
    nodecolor="tag",
    fig_size=(4, 3),
    dpi=300,
    label_node_feat=True,
    edge_vmax=None,
    args=None,
    eps=1e-6,
):
    """
    Args:
        nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
            be one-hot'
    """
    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    cmap = plt.get_cmap("tab20")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in graph.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 19
    # for i in graph.nodes():
    #     if nodecolor == "feat" and "feat" in graph.nodes[i]:
    #         num_classes = graph.nodes[i]["feat"].size()[0]
    #         if num_classes >= 10:
    #             cmap = plt.get_cmap("tab20")
    #             vmax = 19
    #         elif num_classes >= 8:
    #             cmap = plt.get_cmap("tab10")
    #             vmax = 9
    #         break

    feat_labels = {}
    for i in graph.nodes():
        if identify_self and "self" in graph.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "tag" and "tag" in graph.nodes[i]:
            node_colors.append(graph.nodes[i]["tag"])
            feat_labels[i] = graph.nodes[i]["tag"]
        elif nodecolor == "feat" and "feat" in graph.nodes[i]:
            # print(Gc.nodes[i]['feat'])
            feat = graph.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if graph.number_of_nodes() == 0:
        raise Exception("empty graph")
    if graph.number_of_edges() == 0:
        raise Exception("empty edge")
    # remove_nodes = []
    if len(graph.nodes) > 20:
        pos_layout = nx.kamada_kawai_layout(graph, weight=None)
        # pos_layout = nx.spring_layout(graph, weight=None)
    else:
        pos_layout = nx.kamada_kawai_layout(graph, weight=None)

    weights = [d for (u, v, d) in graph.edges(data="weight", default=1)]
    if edge_vmax is None:
        edge_vmax = statistics.median_high(
            [d for (u, v, d) in graph.edges(data="weight", default=1)]
        )
    min_color = min([d for (u, v, d) in graph.edges(data="weight", default=1)])
    # color range: gray to black
    edge_vmin = 2 * min_color - edge_vmax
    print(edge_vmin)
    print(edge_vmax)
    print(edge_colors)
    nx.draw(
        graph,
        pos=pos_layout,
        with_labels=False,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin-eps,
        edge_vmax=edge_vmax,
        width=1.3,
        node_size=100,
        alpha=0.9,
        arrows=False
    )
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nx.write_gpickle(graph, os.path.splitext(save_path)[0] + '.gpickle')
    plt.savefig(save_path, format="pdf")


if __name__ == "__main__":
    graphs = load_data('Mutagenicity', '../dataset')
    # print(list(graphs[0].adjacency()))
    print([list(adj.keys()) for _, adj in graphs[0].adjacency()])
    print(graphs[0].graph['label'])
    print(len(graphs))
