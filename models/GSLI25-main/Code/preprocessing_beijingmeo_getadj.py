
# !!! Please run the processing_beijingmeo.csv before running this script
# Because we need to get beijingmeo_stations.csv 

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from sklearn.metrics.pairwise import haversine_distances

import math
from math import pi

def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM
    print(distances)
    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:36, :36])  
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_adj_beijingmeo():
    df = pd.read_csv("Data/beijingmeo_stations.csv")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values

    adj = get_similarity(res)

    return adj

adj = get_adj_beijingmeo()

np.savetxt("Data/beijingmeo/beijingmeo_adj.csv", adj, delimiter=",", fmt="%.6f")
