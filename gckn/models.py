# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import PathLayer, NodePooling, Linear


class PathSequential(nn.Module):
    def __init__(self, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', #global_pooling='sum',
                 aggregation=False, **kwargs):
        super(PathSequential, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.path_sizes = path_sizes
        self.n_layers = len(hidden_sizes)
        self.aggregation = aggregation

        layers = []
        output_size = hidden_sizes[-1]
        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]

            layer = PathLayer(input_size, hidden_sizes[i], path_sizes[i],
                              kernel_func, kernel_args, pooling, aggregation, **kwargs)
            layers.append(layer)
            input_size = hidden_sizes[i]
            if aggregation:
                output_size *= path_sizes[i]
        self.output_size = output_size

        self.layers = nn.ModuleList(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return self.n_layers

    def __iter__(self):
        return iter(self.layers._modules.values())

    def forward(self, features, paths_indices, other_info):
        output = features
        for layer in self.layers:
            output = layer(output, paths_indices, other_info)
        return output

    def representation(self, features, paths_indices, other_info, n=-1):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            features = self.layers[i](features, paths_indices, other_info)
        return features

    def normalize_(self):
        for module in self.layers:
            module.normalize_()

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.train(False)
        for i, layer in enumerate(self.layers):
            print("Training layer {}".format(i + 1))
            n_sampled_paths = 0
            try:
                n_paths_per_batch = (
                    n_sampling_paths + len(data_loader) - 1) // len(data_loader)
            except:
                n_paths_per_batch = 1000

            paths = torch.Tensor(
                n_sampling_paths, layer.path_size, layer.input_size)
            if use_cuda:
                paths = paths.cuda()

            for data in data_loader.make_batch():
                if n_sampled_paths >= n_sampling_paths:
                    continue
                features = data['features']
                paths_indices = data['paths']
                n_paths = data['n_paths']
                n_nodes = data['n_nodes']

                if use_cuda:
                    features = features.cuda()
                    if isinstance(n_paths, list):
                        paths_indices = [p.cuda() for p in paths_indices]
                        n_paths = [p.cuda() for p in n_paths]
                    else:
                        paths_indices = paths_indices.cuda()
                        n_paths = n_paths.cuda()
                    n_nodes = n_nodes.cuda()
                with torch.no_grad():
                    features = self.representation(
                        features, paths_indices,
                        {'n_paths': n_paths, 'n_nodes': n_nodes}, i)
                    paths_batch = layer.sample_paths(
                        features, paths_indices, n_paths_per_batch)
                    size = paths_batch.shape[0]
                    size = min(size, n_sampling_paths - n_sampled_paths)
                    paths[n_sampled_paths: n_sampled_paths + size] = paths_batch[:size]
                    n_sampled_paths += size

            print("total number of sampled paths: {}".format(n_sampled_paths))
            paths = paths[:n_sampled_paths]
            layer.unsup_train(paths, init=init)


class GCKNetFeature(nn.Module):
    def __init__(self, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', global_pooling='sum',
                 aggregation=False, **kwargs):
        super().__init__()
        self.path_layers = PathSequential(
            input_size, hidden_sizes, path_sizes,
            kernel_funcs, kernel_args_list,
            pooling, aggregation, **kwargs)
        self.node_pooling = NodePooling(global_pooling)
        self.output_size = self.path_layers.output_size

    def forward(self, input, paths_indices, other_info):
        output = self.path_layers(input, paths_indices, other_info)
        return self.node_pooling(output, other_info)

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.path_layers.unsup_train(data_loader, n_sampling_paths,
                                     init, use_cuda)

    def predict(self, data_loader, use_cuda=False):
        if use_cuda:
            self.cuda()
        self.eval()
        output = torch.Tensor(data_loader.n, self.output_size)

        batch_start = 0
        for data in data_loader.make_batch(shuffle=False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            size = len(n_nodes)
            if use_cuda:
                features = features.cuda()
                if isinstance(n_paths, list):
                    paths_indices = [p.cuda() for p in paths_indices]
                    n_paths = [p.cuda() for p in n_paths]
                else:
                    paths_indices = paths_indices.cuda()
                    n_paths = n_paths.cuda()
                n_nodes = n_nodes.cuda()
            with torch.no_grad():
                batch_out = self(features, paths_indices,
                    {'n_paths': n_paths, 'n_nodes': n_nodes}).cpu()
            output[batch_start: batch_start + size] = batch_out
            batch_start += size
        return output, data_loader.labels


class GCKNet(nn.Module):
    def __init__(self, nclass, input_size, hidden_sizes, path_sizes,
                 kernel_funcs=None, kernel_args_list=None,
                 pooling='mean', global_pooling='sum',
                 weight_decay=0.0, **kwargs):
        super().__init__()

        self.features = GCKNetFeature(
            input_size, hidden_sizes, path_sizes,
            kernel_funcs, kernel_args_list,
            pooling, global_pooling, **kwargs)
        self.output_size = self.features.output_size
        self.nclass = nclass

        self.classifier = Linear(self.output_size, nclass, weight_decay)

    def representation(self, input, paths_indices, other_info):
        return self.features(input, paths_indices, other_info)

    def forward(self, input, paths_indices, other_info):
        features = self.representation(input, paths_indices, other_info)
        return self.classifier(features)

    def unsup_train(self, data_loader, n_sampling_paths=100000,
                    init=None, use_cuda=False):
        self.features.unsup_train(data_loader, n_sampling_paths,
                                  init, use_cuda)

    def unsup_train_classifier(self, data_loader, criterion, use_cuda=False):
        encoded_data, labels = self.features.predict(data_loader, use_cuda)
        print(encoded_data.shape)
        self.classifier.fit(encoded_data, labels, criterion)
