# -*- coding: utf-8 -*-
import os
import torch
from torch import nn, optim
import numpy as np
from gckn.data import load_data, PathLoader, separate_data
from gckn.models import GCKNet
from gckn.loss import LOSS

import pandas as pd
import argparse

from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description='Supervised GCKN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--path-size', type=int, nargs='+', default=[3],
                        help='path sizes for layers')
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[32],
                        help='number of filters for layers')
    parser.add_argument('--pooling', type=str, default='sum',
                        help='local path pooling for each node')
    parser.add_argument('--global-pooling', type=str, default='sum',
                        help='global node pooling for each graph')
    parser.add_argument('--aggregation', action='store_true',
                        help='aggregate all path features until path size')
    parser.add_argument('--kernel-funcs', type=str, nargs='+', default=None,
                        help='kernel functions')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.5],
                        help='sigma of exponential (Gaussian) kernels for layers')
    parser.add_argument('--sampling-paths', type=int, default=300000,
                        help='number of paths to sample for unsupervised training')
    parser.add_argument('--weight-decay', type=float, default=1e-04,
                        help='weight decay for classifier')
    parser.add_argument('--fold-idx', type=int, default=0,
                        help='the index of fold in 10-fold validation')
    parser.add_argument('--alternating', action='store_true',
                        help='use alternating training')
    parser.add_argument('--walk', action='store_true',
                        help='use walk instead of path')
    parser.add_argument('--use-cuda', action='store_true',
                        help='use cuda or not')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    ### Model visualization arguments
    parser.add_argument('--interpret', action='store_true',
                        help='interpret model')
    parser.add_argument('--graph-idx', type=int, default=-1,
                        help='graph to interpret')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='regularization parameter for extracting motifs')
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.use_cuda = True

    args.continuous = False
    degree_as_tag = False
    if args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
        # social network
        degree_as_tag = True
    elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
        # bioinformatics
        degree_as_tag = False
    elif args.dataset in ['BZR', 'COX2', 'ENZYMES', 'PROTEINS_full']:
        degree_as_tag = False
        args.continuous = True
    elif args.dataset in ['Mutagenicity']:
        # model visualization
        degree_as_tag = False
    else:
        raise ValueError("Unrecognized dataset!")
    args.degree_as_tag = degree_as_tag

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/sup'
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        if args.aggregation:
            outdir = outdir + '/aggregation'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except:
                    pass
        outdir = outdir + '/{}_{}_{}_{}_{}_{}'.format(
            args.path_size, args.hidden_size, args.pooling,
            args.global_pooling, args.sigma, args.weight_decay)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/cv{}'.format(args.fold_idx)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir
    return args

def train(epoch, model, data_loader, criterion, optimizer,
          lr_scheduler=None, alternating=False, use_cuda=False):
    if alternating or epoch == 0:
        model.eval()
        model.unsup_train_classifier(data_loader['train'], criterion, use_cuda=use_cuda)

    print('current LR: {}'.format(optimizer.param_groups[0]['lr']))
    df = {}
    for phase in ['train', 'val']:
        if phase not in data_loader:
            continue
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_acc = 0.0
        n = data_loader[phase].n

        tic = timer()
        for data in data_loader[phase].make_batch():
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            labels = data['labels']
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
                labels = labels.cuda()
            if phase == 'train':
                optimizer.zero_grad()
                output = model(
                    features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes})
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(
                        features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes})
                    loss = criterion(output, labels)

            pred = output.data.argmax(dim=1)
            running_loss += loss.item() * size
            running_acc += torch.sum(pred == labels).item()

        toc = timer()
        epoch_loss = running_loss / n
        epoch_acc = running_acc / n
        df[phase + "_loss"] = epoch_loss
        df[phase + "_acc"] = epoch_acc
        print('{} loss: {:.4f} acc: {:.4f} time: {:.2f}s'.format(
            phase, epoch_loss, epoch_acc, toc - tic))
    print()
    if lr_scheduler is not None:
        lr_scheduler.step()
    return df


def main():
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    if args.interpret:
        args.use_cuda = False
        main_motif(args)
        return

    graphs, nclass = load_data(args.dataset, '../dataset', degree_as_tag=args.degree_as_tag)
    if args.continuous:
        from sklearn.preprocessing import StandardScaler
        print("Dataset with continuous node attributes")
        node_features = np.concatenate([g.node_features for g in graphs], axis=0)
        sc = StandardScaler()
        sc.fit(node_features)
        for g in graphs:
            node_features = sc.transform(g.node_features)
            g.node_features = node_features / np.linalg.norm(node_features, axis=-1, keepdims=True).clip(min=1e-06)

    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    train_loader = PathLoader(train_graphs, max(args.path_size), args.batch_size,
                              True, dataset=args.dataset, walk=args.walk)
    test_loader = PathLoader(test_graphs, max(args.path_size), args.batch_size,
                             True, dataset=args.dataset, walk=args.walk)
    train_loader.get_all_paths()
    test_loader.get_all_paths()
    input_size = train_loader.input_size

    print('Unsupervised initialization...')
    model = GCKNet(nclass, input_size, args.hidden_size, args.path_size,
                   kernel_funcs=args.kernel_funcs, kernel_args_list=args.sigma,
                   pooling=args.pooling,
                   global_pooling=args.global_pooling, aggregation=args.aggregation,
                   weight_decay=args.weight_decay)
    model.unsup_train(train_loader, n_sampling_paths=args.sampling_paths)

    data_loader = {'train': train_loader, 'val': test_loader}
    if args.dataset == 'Mutagenicity':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LOSS['hinge'](nclass)

    if args.alternating:
        # optimizer = optim.Adam(model.features.parameters(), lr=args.lr)
        optimizer = optim.SGD(model.features.parameters(), lr=args.lr, momentum=0.9)
    else:
        #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = optim.Adam([
            {'params': model.features.parameters()},
            {'params': model.classifier.parameters(), 'weight_decay': args.weight_decay}
            ], lr=args.lr)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    print('Starting training...')
    if args.use_cuda:
        model.cuda()
    from collections import defaultdict
    table = defaultdict(list)
    tic = timer()
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)
        df = train(epoch, model, data_loader, criterion, optimizer,
                                  lr_scheduler, alternating=args.alternating,
                                  use_cuda=args.use_cuda)
        for name in df:
            table[name].append(df[name])
    toc = timer()
    ts = toc - tic
    if args.save_logs:
        df['ts'] = ts
        df = pd.DataFrame.from_dict(df, orient='index')
        df.to_csv(args.outdir + '/results.csv',
                  header=['value'], index_label='name')
        table = pd.DataFrame.from_dict(table)
        table.to_csv(args.outdir + '/epoch_results.csv', index=False)
        args.nclass = nclass
        args.input_size = input_size
        torch.save({'args': args, 'weights': model.state_dict()}, args.outdir + '/model.pt')

def load_model(datapath):
    model = torch.load(datapath)
    args = model['args']
    weights = model['weights']
    model = GCKNet(args.nclass, args.input_size, args.hidden_size, args.path_size,
                   kernel_funcs=args.kernel_funcs, kernel_args_list=args.sigma,
                   pooling=args.pooling,
                   global_pooling=args.global_pooling, aggregation=args.aggregation,
                   weight_decay=args.weight_decay)
    model.load_state_dict(weights)
    return model, args

def train_motif(epoch, mask, model, data_loader, criterion, optimizer,
                mu, use_cuda=False):
    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    # print(mask)
    for data in data_loader.make_batch(False):
        features = data['features']
        paths_indices = data['paths']
        n_paths = data['n_paths']
        n_nodes = data['n_nodes']
        labels = data['labels']
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
            labels = labels.cuda()

        def closure():
            optimizer.zero_grad()
            output = model(
                features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes, 'mask': mask})
            loss = criterion(output, labels)
            for m in mask:
                loss = loss + mu * m.abs().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        for m in mask:
            m.data.clamp_(min=0, max=1)
        output = model(
            features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes, 'mask': mask})
        loss = criterion(output, labels)
        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * size
        running_acc += torch.sum(pred == labels).item()

    toc = timer()
    epoch_loss = running_loss
    epoch_acc = running_acc
    print('loss: {:.4f} acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    print()

def evaluate(model, data_loader, use_cuda=False):
    pred_labels = torch.zeros(data_loader.n, dtype=torch.long)
    idx = 0
    for data in data_loader.make_batch(False):
        features = data['features']
        paths_indices = data['paths']
        n_paths = data['n_paths']
        n_nodes = data['n_nodes']
        labels = data['labels']
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
            labels = labels.cuda()
        output = model(
                features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes})
        # loss = criterion(output, labels)
        pred = output.data.argmax(dim=1)#.item()
        pred_labels[idx:idx+size] = pred
        idx += size
    return pred_labels

def main_motif(args):
    from gckn.data_io import get_motif, log_graph

    graphs, nclass = load_data(args.dataset, '../dataset', degree_as_tag=args.degree_as_tag)
    model, model_args = load_model(args.outdir + '/model.pt')
    if args.use_cuda:
        model.cuda()
    if args.graph_idx == -1:
        sizes = [len(g.neighbors) for g in graphs]
        indices = np.argsort(sizes)[::-1]
        indices = [i for i in indices if sizes[i] <= 100][:2]
    else:
        indices = [args.graph_idx]
    for idx in indices:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        args.graph_idx = idx

        graph = [graphs[args.graph_idx]]
        print(graph[0].label)

        train_loader = PathLoader(graph, max(model_args.path_size), 1,
                                  True, mask=True)
        train_loader.get_all_paths()
        label = evaluate(model, train_loader, args.use_cuda)
        print(label)
        if label.item() != 0 or graph[0].label != label.item():
            continue
        train_loader.labels = label

        mask = train_loader.mask_vec[0]
        if not isinstance(mask, list):
            mask = [mask]
        for i, m in enumerate(mask):
            if args.use_cuda:
                m = m.cuda()
                mask[i] = m
            m.fill_(1.)
            m.requires_grad_()

        model.eval()

        optimizer = optim.LBFGS(mask, lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch + 1, args.epochs))
            print('-' * 10)
            train_motif(epoch, mask, model, train_loader, criterion, optimizer, args.mu,
                        use_cuda=args.use_cuda)

        mask = [m.data for m in mask]
        print(mask)
        paths = train_loader.data['paths'][0]
        print(paths)
        motif_graph = get_motif(mask, paths, graph[0].g)
        # print(motif)
        outdir = args.outdir + '/motifs'
        try:
            os.makedirs(outdir)
        except:
            pass
        outdir = outdir + '/{}'.format(args.graph_idx)
        try:
            os.makedirs(outdir)
        except:
            pass
        log_graph(graph[0].g, outdir, 'origin{}.pdf'.format(args.graph_idx))
        log_graph(motif_graph, outdir, 'motif{}_{}.pdf'.format(args.graph_idx, label.item()))


if __name__ == "__main__":
    main()
