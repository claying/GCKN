# -*- coding: utf-8 -*-
import numpy as np
import os
import copy
import torch
from collections import defaultdict

from gckn.data import GraphLoader, convert_dataset
from gckn.models import GCKNetFeature
from torch_geometric import datasets
import torch_geometric.transforms as T

import pandas as pd
import argparse

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from timeit import default_timer as timer
from warnings import simplefilter
simplefilter('ignore', category=ConvergenceWarning)


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
    parser.add_argument('--kernel-funcs', type=str, nargs='+', default=None,
                        help='kernel functions')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.5],
                        help='sigma of exponential (Gaussian) kernels for layers')
    parser.add_argument('--sampling-paths', type=int, default=300000,
                        help='number of paths to sample for unsupervised training')
    parser.add_argument('--walk', action='store_true',
                        help='use walk instead of path')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    args = parser.parse_args()

    args.use_cuda = False
    if torch.cuda.is_available():
        args.use_cuda = True

    args.continuous = False
    if args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'COLLAB']:
        # social network
        degree_as_tag = True
    elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1']:
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
        outdir = outdir + '/gckn_unsup'
        outdir = outdir + '/{}'.format(args.dataset)
        if args.aggregation:
            outdir = outdir + '/aggregation'
        outdir = outdir + '/{}_{}_{}_{}_{}'.format(
            args.path_size, args.hidden_size, args.pooling,
            args.global_pooling, args.sigma)
        os.makedirs(outdir, exist_ok=True)
        args.outdir = outdir
    return args


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    data_path = '../datasets/TUDataset'

    dset_name = args.dataset

    dset = datasets.TUDataset(data_path, dset_name, use_node_attr=args.continuous,)
    nclass = dset.num_classes
    n_tags = None

    if args.continuous:
        node_features = dset.data.x.numpy()
        sc = StandardScaler().fit(node_features)

        def normalize_features(data):
            x_trans = sc.transform(data.x.numpy())
            x_trans /= np.linalg.norm(x_trans, axis=-1, keepdims=True).clip(min=1e-06)
            data.x = torch.from_numpy(x_trans)
            return data
        dset.transform = normalize_features

    graphloader = GraphLoader(args.path_size, args.batch_size, args.dataset, args.walk)

    converted_dset = convert_dataset(dset, n_tags)
    data_loader = graphloader.transform(converted_dset)
    input_size = data_loader.input_size

    model = GCKNetFeature(input_size, args.hidden_size, args.path_size,
                          kernel_funcs=args.kernel_funcs, kernel_args_list=args.sigma,
                          pooling=args.pooling, global_pooling=args.global_pooling,
                          aggregation=args.aggregation)

    model.unsup_train(data_loader, n_sampling_paths=args.sampling_paths)

    if args.use_cuda:
        model.cuda()

    print('Encoding...')
    tic = timer()
    X, y = model.predict(data_loader)
    toc = timer()

    print("Embedding finished, time: {:.2f}s".format(toc - tic))
    print(X)
    print(X.shape)
    print(y)

    X = X.numpy().astype('double')
    y = y.numpy().astype('double')

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    C_list = np.logspace(-4, 4, 60)
    lambda_list = 1. / (2. * X.shape[0] * C_list)
    param_grid = {'clf__lambda_1': lambda_list}
    param_grid = {
        # 'clf__max_depth': [None],
        'clf__min_samples_leaf': [1, 2],
        'clf__min_samples_split': [2, 5, 10],
        'clf__n_estimators': [25, 50, 100, 200]
    }

    scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rf = RandomForestClassifier(class_weight='balanced', random_state=args.seed)
        # svc = LinearSVC(lambda_1=1.0, verbose=False, random_state=42)
        clf = GridSearchCV(Pipeline(steps=[('scaler', StandardScaler()), ('clf', rf)]),
                           param_grid,
                           cv=5, scoring='accuracy',
                           n_jobs=1, verbose=0, refit=True)
    
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        scores.append(score)
        print("score: {:.6f}".format(score))

    print("Acc: {:.2f} ± {:.2f}".format(np.mean(scores) * 100, np.std(scores) * 100))


if __name__ == "__main__":
    main()
