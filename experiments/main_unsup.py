# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from gckn.data import load_data, PathLoader
from gckn.models import GCKNetFeature

import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cyanure import LinearSVC


def load_args():
    parser = argparse.ArgumentParser(
        description='Unsupervised GCKN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--path-size', type=int, nargs='+', default=[2],
                        help='path sizes for layers')
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[64],
                        help='number of filters for layers')
    parser.add_argument('--pooling', type=str, default='sum',
                        help='local path pooling for each node')
    parser.add_argument('--global-pooling', type=str, default='sum',
                        help='global node pooling for each graph')
    parser.add_argument('--aggregation', action='store_true',
                        help='aggregate all path features until path size')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.5],
                        help='sigma of exponential (Gaussian) kernels for layers')
    parser.add_argument('--sampling-paths', type=int, default=300000,
                        help='number of paths to sample for unsupervised training')
    parser.add_argument('--walk', action='store_true',
                        help='use walk instead of path')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    args = parser.parse_args()
    args.continuous = False
    if args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
        # social network
        degree_as_tag = True
    elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
        # bioinformatics
        degree_as_tag = False
    elif args.dataset in ['BZR', 'COX2', 'ENZYMES', 'PROTEINS_full']:
        degree_as_tag = False
        args.continuous = True
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
        outdir = outdir + '/unsup'
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
        outdir = outdir + '/{}_{}_{}_{}_{}'.format(
            args.path_size, args.hidden_size, args.pooling,
            args.global_pooling, args.sigma)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir
    return args

def main():
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    graphs, n_class = load_data(args.dataset, '../dataset', degree_as_tag=args.degree_as_tag)
    if args.continuous:
        print("Dataset with continuous node attributes")
        node_features = np.concatenate([g.node_features for g in graphs], axis=0)
        sc = StandardScaler()
        sc.fit(node_features)
        for g in graphs:
            node_features = sc.transform(g.node_features)
            g.node_features = node_features / np.linalg.norm(node_features, axis=-1, keepdims=True).clip(min=1e-06)

    data_loader = PathLoader(graphs, max(args.path_size), args.batch_size,
                             True, dataset=args.dataset, walk=args.walk)
    print('Computing paths...')
    if args.dataset != 'COLLAB' or max(args.path_size) <= 2:
        data_loader.get_all_paths()
    input_size = data_loader.input_size

    print('Unsupervised training...')
    model = GCKNetFeature(input_size, args.hidden_size, args.path_size,
                          kernel_args_list=args.sigma, pooling=args.pooling,
                          global_pooling=args.global_pooling,
                          aggregation=args.aggregation)
    model.unsup_train(data_loader, n_sampling_paths=args.sampling_paths)

    print('Encoding...')
    features, labels = model.predict(data_loader)
    print(features)
    print(features.shape)
    print(labels)

    features = features.numpy()
    labels = labels.numpy()

    print('Cross validation')
    train_fold_idx = [np.loadtxt('../dataset/{}/10fold_idx/train_idx-{}.txt'.format(
        args.dataset, i)).astype(int) for i in range(1, 11)]
    test_fold_idx = [np.loadtxt('../dataset/{}/10fold_idx/test_idx-{}.txt'.format(
        args.dataset, i)).astype(int) for i in range(1, 11)]
    cv_idx = zip(train_fold_idx, test_fold_idx)

    C_list = np.logspace(-4, 4, 60)
    svc = LinearSVC(C=1.0)
    clf = GridSearchCV(make_pipeline(StandardScaler(), svc),
                       {'linearsvc__C' : C_list},
                       cv=cv_idx,
                       n_jobs=-1, verbose=0, return_train_score=True)
    
    clf.fit(features, labels)
    df = pd.DataFrame({'C': C_list, 
                       'train': clf.cv_results_['mean_train_score'], 
                       'test': clf.cv_results_['mean_test_score'],
                       'test_std': clf.cv_results_['std_test_score']}, 
                        columns=['C', 'train', 'test', 'test_std'])
    print(df)

    if args.save_logs:
        df.to_csv(args.outdir + "/results.csv")


if __name__ == "__main__":
    main()
