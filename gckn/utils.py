# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

EPS = 1e-4


def normalize_(x, p=2, dim=-1, c2=1.):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS) / math.sqrt(c2))
    return x

def init_kmeans(x, n_clusters, norm=1., n_local_trials=None, use_cuda=False):
    n_samples, n_features = x.size()
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    clusters[0] = x[np.random.randint(n_samples)]

    closest_dist_sq = 2 * (norm - clusters[[0]].mm(x.t()))
    closest_dist_sq = closest_dist_sq.view(-1)
    current_pot = closest_dist_sq.sum().item()

    for c in range(1, n_clusters):
        rand_vals = np.random.random_sample(n_local_trials).astype('float32') * current_pot
        rand_vals = np.minimum(rand_vals, current_pot * (1.0 - EPS))
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1).cpu(), rand_vals)
        distance_to_candidates = 2 * (norm - x[candidate_ids].mm(x.t()))

        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq,
                                    distance_to_candidates[trial])
            new_pot = new_dist_sq.sum().item()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters

def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x kmer_size x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    if x.ndim == 3:
        n_samples, kmer_size, n_features = x.size()
    else:
        n_samples, n_features = x.size()
    if init == "kmeans++":
        print(init)
        if x.ndim == 3:
            clusters = init_kmeans(x.view(n_samples, -1), n_clusters, norm=kmer_size, use_cuda=use_cuda)
            clusters = clusters.view(n_clusters, kmer_size, n_features)
        else:
            clusters = init_kmeans(x, n_clusters, use_cuda=use_cuda)
    else:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf

    for n_iter in range(max_iters):
        # assign data points to clusters
        cos_sim = x.view(n_samples, -1).mm(clusters.view(n_clusters, -1).t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                # clusters[j] = x[random.randrange(n_samples)]
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm(dim=-1, keepdim=True).clamp(min=EPS)

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters
