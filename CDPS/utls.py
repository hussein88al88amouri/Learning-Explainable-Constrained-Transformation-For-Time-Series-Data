"""
utls.py

This module contains basic function used in CDPS model.
"""
import numpy
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
import torch


def generate_random_indices(nsamples, batch_size, idx_exclude):
    """
    Generate Random indices for the input batch.
    The indices will be gerenated such that they are not repeated in A.
    Args:
        nsamples: total number of samples to generate indices for
        batchsize: size of the batch size
        idx_exclude: set of indices not to choose from
        (These indices will be found in ml or cl sets)
    """

    if nsamples < batch_size:
        raise ValueError("nsamples must >= batch_size.")

    if batch_size > nsamples:
        raise ValueError("batch_size must <= nsamples.")

    size = [batch_size, 2]
    random_indices = numpy.random.choice(
        nsamples, size=size, replace=False)
    f_duplicate = any(numpy.any((random_indices[:, None, :] ==
                                 idx_exclude).all(axis=-1), axis=-1))
    while f_duplicate:
        random_indices = numpy.random.choice(
            nsamples, size=size, replace=False)
        f_duplicate = any(numpy.any((random_indices[:, None, :] ==
                                     idx_exclude).all(axis=-1), axis=-1))
    return random_indices


def compute_dtw_indep_dep(time_series1, time_series2, type_='DEP'):
    """
    Calculate DTW either in depenedent or independet approach.
    check the article:
    Args:
        time_series1: multidimensional time series
        time_series2: multidimensional time series
        type_: "DEP" --> dependent approach
            "INDEP" --> indepent appraoch
    """

    if type_ == 'DEP':
        return dtw(time_series1, time_series2)
    elif type_ == 'INDEP':
        dtwi = 0
        for dim in range(time_series1.shape[-1]):
            dtwi += dtw(time_series1[:, dim], time_series2[:, dim])
        return dtwi
    else:
        raise ValueError("Unsupported value for 'type_'. "
                         "Choose either 'INDEP' or 'DEP'.")


def tslearn2torch(x_input, device="cpu"):
    """
    numpy array to a torch tensor
    Args:
        x_input (ndarry): input array representing a time series
        device (str, optional): load tensor on "cuda' or "cpu".
    Returns:
        torch.tensor: tensor
    """
    x_tensor = torch.Tensor(numpy.transpose(x_input, (0, 2, 1))).to(device)
    return x_tensor


def _kmeans_init_shapelets(x_input, n_shapelets, shp_len, n_draw=10000):
    """
    intialize the shapelets layer weights using k-means cluster
    centers as shapelets.

    Args:
        x_input (ndarry): _description_
        n_shapelets (int): _description_
        shp_len (int): _description_
        n_draw (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """

    n_ts, sz, d = x_input.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.random.choice(sz - shp_len + 1,
                                       size=n_draw,
                                       replace=True)
    subseries = numpy.zeros((n_draw, shp_len, d))
    for i in range(n_draw):
        t_idx1 = indices_time[i]
        t_idx2 = indices_time[i] + shp_len
        subseries[i] = x_input[indices_ts[i], t_idx1: t_idx2]
    return TimeSeriesKMeans(n_clusters=n_shapelets,
                            metric="euclidean",
                            verbose=False).fit(subseries).cluster_centers_
