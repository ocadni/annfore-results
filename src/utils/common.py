import warnings
import numpy as np

def normalize(arr, dim=-1):
    """
    Return array of the same shape, where the sum along dimension `dim` is 1
    
    Avoids division by zero when one of the sum is 0
    """
    sum_r = arr.sum(dim)
    if len(arr.shape) == 1:
        return arr/sum_r

    if dim < 0:
        dim = dim % len(arr.shape)
    sum_r = sum_r.reshape(list(arr.shape[:dim])+[1]+list(arr.shape[dim+1:]))
    if np.any(sum_r==0):
        warnings.warn("Some sums are zero on dimension {:d}".format(dim))
    sum_r=np.repeat(sum_r, arr.shape[dim], dim)
    g = np.where(sum_r != 0)
    xc = arr.copy()
    xc[g] /= sum_r[g]
    return xc

def sort_by_inf(M, t=0, sep=False):
    vals = M[:,t,1]
    idx = np.argsort(vals)[::-1]
    if sep:
        return idx, vals[idx]
    else:
        ### legacy format
        return np.stack((vals[idx],idx),1)