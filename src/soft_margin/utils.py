"""
Common utility functions

Author: Fabio Mazza
"""
import warnings
import numpy as np
import numba as nb

import forward_sim.propagate as propagate

def warn(msg):
    warnings.warn(msg, RuntimeWarning)

def get_random_binary(shape, p):
    return np.random.random(shape) < p 

def get_nb_float_dtype():
    t = np.empty((1,),dtype=np.float_)
    if t.itemsize ==4:
        return nb.float32
    elif t.itemsize ==8:
        return nb.float64
    else:
        raise ValueError("Cannot find dtype of float_")

def get_one_hot_numpy(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

@nb.njit()
def sample_prob(probabs):
    p = np.random.random()
    c = 0.
    for i in range(len(probabs)):
        c += probabs[i]
        if p < c:
            return i
    if c == 0:
        raise ValueError("Sum p is 0")
    raise ValueError("p>1??")

@nb.njit()
def count_elemts(array):
    n_pars = array.shape[0]
    N = array.max()+1
    counts = np.zeros((n_pars,N), dtype=np.int_)
    for i in range(n_pars):
        for j in range(array.shape[1]):
            counts[i][array[i,j]] += 1
    return counts

@nb.njit()
def convert_idx(counts):
    out = np.empty(counts.sum(), dtype=np.int_)
    c = 0
    for i in range(len(counts)):
        n = counts[i]
        out[c:(n+c)] = i
        c += n
    return out

def sample_probs_n(probs, n):
    """
    Sample N times from values (0,1,....M) with M different probabilities
    Multinomial draw with replacement
    Returns the index of the chosen probability
    """
    x = np.random.random((n, 1))
    r = np.argmax(x < np.cumsum(probs), 1)
    return r


def sample_probs_multi(probs, n_per_r):
    """
    Sample N times from probabilites probs, sampling on their last dimension
    """
    x = np.random.random(list(probs.shape[:-1])+[n_per_r, 1])
    check = np.abs(probs.sum(-1)-1)>1e-10
    if np.any(check):
        raise ValueError("Sum probability on last axis != 1")
    probs_sum = np.cumsum(probs, -1).reshape(
        list(probs.shape[:-1]) + [1, probs.shape[-1]])
    #print(x.shape, probs_sum.shape)
    r = np.argmax(x < probs_sum, -1)
    return r

@nb.njit(parallel=True)
def sample_probs_multi_nb(probs, n_per_r):
    """
    Sample n_per_r times from values (0,1,....M) with LxM different probabilities
    Multinomial draw with replacement on the second axis
    Returns the index of the chosen probability
    """
    first = probs.shape[0]
    n = probs.shape[1]
    vals = np.zeros((first,n_per_r),dtype=np.int64)
    for j in nb.prange(n_per_r):
        for i in nb.prange(first):
            if probs[i].sum() == 0:
                vals[i,j] = int(np.random.rand()*len(probs[i]))
                #if not warning_shown:
                #    print("WARNING: Got sum probabilities = 0, extracting at random")
                #    warning_shown = True
            else:
                vals[i,j] = sample_prob(probs[i])
    return vals

@nb.njit(parallel=True)
def sample_multi_1d_nb(probs, n_per_r):
    """
    Sample n_per_r times from values (0,1,....M) with LxM different probabilities
    Multinomial draw with replacement on the second axis
    Returns the index of the chosen probability
    """
    if len(probs.shape) > 1:
        raise ValueError("2D probability array not supported")
    if np.abs(probs.sum() -1.0) > 1e-11:
        raise ValueError("Sum of probabilities is not 1")
    
    n = probs.shape[0]
    vals = np.zeros(n_per_r,dtype=np.int_)
    if probs.sum() == 0:
        raise ValueError("Sum of prob is 0")
    for i in nb.prange(n_per_r):
        vals[i] = sample_prob(probs)
    return vals

def get_traj_one_hot(t_limit, inf_times, delays, n_states):

    n = inf_times.shape[0]
    if n != delays.shape[0]:
        raise ValueError("Infection times or delays missing")
    
    fin_state = np.zeros((t_limit+1, n, n_states), dtype=np.int8)

    fill_traj_one_hot_nb(fin_state, inf_times, delays)

    return fin_state

@nb.njit
def fill_traj_one_hot_nb(result, inf_times, delays):
    
    n_times = result.shape[0]
    n_nodes = result.shape[1]
    if n_nodes != delays.shape[0] or n_nodes != inf_times.shape[0]:
        raise ValueError("Wrong number of infection times or delays")
        
    for i in range(n_nodes):
        for t in range(n_times):
            st = int( propagate.state_numba(t,inf_times[i],delays[i])-1 )
            result[t,i,st] = 1

@nb.njit
def fill_traj_one_hot_nb_val(result, inf_times, delays, val):
    """
    Fill trajectory in one_hot way, but with value val
    """
    n_times = result.shape[0]
    n_nodes = result.shape[1]
    if n_nodes != delays.shape[0] or n_nodes != inf_times.shape[0]:
        raise ValueError("Wrong number of infection times or delays")
        
    for i in range(n_nodes):
        for t in range(n_times):
            st = int( propagate.state_numba(t,inf_times[i],delays[i])-1 )
            result[t,i,st] = val
