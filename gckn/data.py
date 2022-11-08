# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as utils
from .graphs import get_paths, get_walks


def get_adj_list(g):
    neighbors = [[] for _ in range(g.num_nodes)]
    for k in range(g.edge_index.shape[-1]):
        i, j = g.edge_index[:, k]
        neighbors[i.item()].append(j.item())
    return neighbors

def convert_dataset(dataset, n_tags=None):
    """Convert a PyG dataset to GCKN dataset
    """
    if dataset is None:
        return dataset
    graph_list = []
    for i, g in enumerate(dataset):
        new_g = S2VGraph(g, g.y)
        new_g.neighbors = get_adj_list(g)
        if n_tags is not None:
            new_g.node_features = F.one_hot(g.x.view(-1).long(), n_tags).numpy()
        else:
            new_g.node_features = g.x.numpy()
        degree_list = utils.degree(g.edge_index[0], g.num_nodes).numpy()
        new_g.max_neighbor = max(degree_list)
        new_g.mean_neighbor = (sum(degree_list) + len(degree_list) - 1) // len(degree_list)
        graph_list.append(new_g)
    return graph_list


class GraphLoader(object):
    """
    This class takes a list of graphs and transforms it into a
    data_loader.
    """
    def __init__(self, path_size, batch_size, dataset, walk=False):
        self.path_size = path_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.walk = walk

    def transform(self, graphs):
        data_loader = PathLoader(graphs, max(self.path_size), self.batch_size,
                                 True, dataset=self.dataset, walk=self.walk)
        if self.dataset != 'COLLAB' or max(self.path_size) <= 2:
            data_loader.get_all_paths()
        return data_loader

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the
            tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.pe = None

        self.max_neighbor = 0
        self.mean_neighbor = 0

def get_path_indices(paths, n_paths_for_graph, n_nodes):
    """
    paths: all_paths x k
    n_paths:: n_graphs (sum=all_paths)
    """
    incr_indices = torch.cat([torch.zeros(1, dtype=torch.long), n_nodes[:-1]])
    incr_indices = incr_indices.cumsum(dim=0)
    incr_indices = incr_indices.repeat_interleave(n_paths_for_graph, dim=0).view(-1, 1)
    paths = paths + incr_indices
    return paths

class PathLoader(object):
    def __init__(self, graphs, k, batch_size, aggregation=True, dataset='MUTAG', padding=False, walk=False, mask=False):
        # self.data = data
        self.dataset = dataset
        self.graphs = graphs
        self.batch_size = batch_size
        self.aggregation = aggregation
        self.input_size = graphs[0].node_features.shape[-1]
        self.n = len(graphs)
        self.k = k
        self.data = None
        self.labels = None
        self.padding = padding
        self.walk = walk
        self.mask = mask

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_all_paths(self, dirname=None):
        all_paths = []
        n_paths = []
        n_nodes = torch.zeros(self.n, dtype=torch.long)
        if self.aggregation:
            n_paths_for_graph = torch.zeros((self.n, self.k), dtype=torch.long)
        else:
            n_paths_for_graph = torch.zeros(self.n, dtype=torch.long)
        features = []
        labels = torch.zeros(self.n, dtype=torch.long)
        mask_vec = []
        if dirname is not None and self.dataset == 'COLLA':
            try:
                self.data = torch.load(dirname + '/all_paths_{}.pkl'.format(self.k))
                self.labels = self.data['labels']
                return
            except:
                pass
        for i, g in enumerate(self.graphs):
            # print(i)
            if self.walk:
                p, c = get_walks(g, self.k)
            else:
                p, c = get_paths(g, self.k)
            if self.aggregation:
                all_paths.append([torch.from_numpy(p[j]) for j in range(self.k)])
                n_paths.append([torch.from_numpy(c[:, j]) for j in range(self.k)])
                n_paths_for_graph[i] = torch.LongTensor([len(p[j]) for j in range(self.k)])
                if self.mask:
                    mask_vec.append([torch.ones(len(p[j])) for j in range(self.k)])
            else:
                all_paths.append(torch.from_numpy(p[-1]))
                n_paths.append(torch.from_numpy(c[:, -1]))
                n_paths_for_graph[i] = len(p[-1])
                mask_vec.append(torch.ones(len(p[-1])))
            n_nodes[i] = len(g.neighbors)
            features.append(torch.from_numpy(g.node_features.astype('float32')))

            labels[i] = g.label

        self.data = {
            'features': features,
            'paths': all_paths,
            'n_paths': n_paths,
            'n_paths_for_graph': n_paths_for_graph,
            'n_nodes': n_nodes,
            'labels': labels
        }
        self.mask_vec = mask_vec
        self.labels = labels
        if dirname is not None and self.dataset == 'COLLA':
            torch.save(self.data, dirname + '/all_paths_{}.pkl'.format(self.k))

    def make_batch(self, shuffle=True):
        if self.data is None:
            # raise ValueError('Plase first run self.get_all_paths() to compute paths!')
            if self.labels is None:
                self.labels = torch.LongTensor([g.label for g in self.graphs])
            if shuffle:
                indices = np.random.permutation(self.n)
            else:
                indices = list(range(self.n))
            for index in range(0, self.n, self.batch_size):
                idx = indices[index:min(index + self.batch_size, self.n)]
                size = len(idx)
                # current_features = torch.cat([torch.from_numpy(
                #     self.graphs[i].node_features.astype('float32')) for i in idx])
                current_features = []
                current_n_nodes = torch.zeros(size, dtype=torch.long)
                if self.aggregation:
                    current_paths = [[] for i in range(self.k)]
                    current_n_paths = [[] for i in range(self.k)]
                    current_n_paths_for_graph = torch.zeros((size, self.k), dtype=torch.long)
                else:
                    current_paths = []
                    current_n_paths = []
                    current_n_paths_for_graph = torch.zeros(size, dtype=torch.long)

                for i, g_index in enumerate(idx):
                    g = self.graphs[g_index]
                    current_features.append(torch.from_numpy(
                        g.node_features.astype('float32')))
                    if self.walk:
                        p, c = get_walks(g, self.k)
                    else:
                        p, c = get_paths(g, self.k)
                    current_n_nodes[i] = len(g.neighbors)
                    if self.aggregation:
                        for j in range(self.k):
                            current_paths[j].append(torch.from_numpy(p[j]))
                            current_n_paths[j].append(torch.from_numpy(c[:, j]))
                            current_n_paths_for_graph[i, j] = len(p[j])
                    else:
                        current_paths.append(torch.from_numpy(p[-1]))
                        current_n_paths.append(torch.from_numpy(c[:, -1]))
                        current_n_paths_for_graph[i] = len(p[-1])
                current_features = torch.cat(current_features)
                if self.aggregation:
                    for j in range(self.k):
                        current_paths[j] = get_path_indices(
                            torch.cat(current_paths[j]), current_n_paths_for_graph[:, j], current_n_nodes)
                        current_n_paths[j] = torch.cat(current_n_paths[j])
                else:
                    current_paths = get_path_indices(
                        torch.cat(current_paths), current_n_paths_for_graph, current_n_nodes)
                    current_n_paths = torch.cat(current_n_paths)
                yield {'features': current_features,
                       'paths': current_paths,
                       'n_paths': current_n_paths,
                       'n_nodes': current_n_nodes,
                       'labels': self.labels[idx]}
            return

        if shuffle:
            indices = np.random.permutation(self.n)
        else:
            indices = list(range(self.n))
        features = self.data['features']

        for index in range(0, self.n, self.batch_size):
            idx = indices[index:min(index + self.batch_size, self.n)]
            current_features = torch.cat([features[i] for i in idx])
            if self.padding:
                current_features = torch.cat([torch.zeros(self.input_size).view(1, -1), current_features])

            if self.aggregation:
                current_paths = [torch.cat([self.data['paths'][i][j] for i in idx]) for j in range(self.k)]
            else:
                current_paths = torch.cat([self.data['paths'][i] for i in idx]) + self.padding
            current_n_paths_for_graph = self.data['n_paths_for_graph'][idx]
            current_n_nodes = self.data['n_nodes'][idx]

            if self.aggregation:
                current_paths = [get_path_indices(
                    current_paths[j], current_n_paths_for_graph[:, j],
                    current_n_nodes) for j in range(self.k)]
                current_n_paths = [torch.cat(
                    [self.data['n_paths'][i][j] for i in idx]) for j in range(self.k)]
            else:
                current_paths = get_path_indices(
                    current_paths, current_n_paths_for_graph, current_n_nodes)
                current_n_paths = torch.cat([self.data['n_paths'][i] for i in idx])
            yield {'features': current_features,
                   'paths': current_paths,
                   'n_paths': current_n_paths,
                   'n_nodes': current_n_nodes,
                   'labels': self.labels[idx]}

