import numpy as np
from numba import njit, prange
from epigen import propagate
import soft_margin.utils as softm_utils

nbfloat = softm_utils.get_nb_float_dtype()


@njit()
def get_categ_I_R_numba(conf):
    """
    Create sets of infected and recovered nodes in conf
    """
    I_val=1
    R_val=2
    n = len(conf)
    seti = set()
    setr = set()
    for i in range(n):
        if conf[i] == I_val:
            seti.add(i)
        elif conf[i] == R_val:
            setr.add(i)
    return seti,setr

@njit
def jaccard_idx_nb(set1,set2):
    """
    Get jaccard index of two sets
    """
    unions = set1.union(set2)
    inters = set1.intersection(set2)
    if len(unions) > 0:
        j = len(inters)/float(len(unions))
    else:
        j = 1
    return j

@njit
def get_jacc_conf_val(v1,c2, val):
    """
    Compute jaccard index for value `val` in array
    v1 is the precomputed (c1 == val)
    """
    #v1 = (c1 == val)
    v2 = (c2 == val)
    un = v1 | v2
    n_un = un.sum()
    if n_un == 0:
        return 1
    inters = v1 & v2
    n_inter = inters.sum()
    return n_inter/n_un


@njit()
def calc_jacc_conf(c1, c2):
    v1 = (c1==1)
    val1 = get_jacc_conf_val(v1,c2,1)
    v1 = (c1==2)
    val2 = get_jacc_conf_val(v1,c2,2)
    return val1, val2

### LIKELIHOOD CALCULATION
def gaussian_weight(j, a):
    return np.exp(-(1-j)**2/a**2)


@njit()
def _gauss_weight(jaccs, a):
    """
    Compute the gaussian weight from similarity indexes
    """
    sumj =( (1-jaccs[0])**2 + (1-jaccs[1])**2 ) / 2
    if sumj == 0:
        return 1
    else:
        return np.exp(-sumj/a**2)


@njit(parallel=True)
def calc_g_weight_nb_2d(jaccs, a_pars):
    """
    Calculate gaussian weight from jaccard idcs
    Jaccs is 3D: (n_pars, n_sims, 2)
    """
    if len(a_pars.shape) > 1:
        raise ValueError("Only working for linear a_pars")
    if jaccs.shape[0] != len(a_pars):
        raise ValueError("Lenght of parameters not corresponding")
    n_sims = jaccs.shape[1]
    n_pars = len(a_pars)
    weights = np.empty((n_pars, n_sims))
    for i in prange(n_sims):
        for j in prange(n_pars):
            weights[j,i] = _gauss_weight(jaccs[j,i], a_pars[j])
    return weights

@njit(parallel=True)
def calc_g_weight_nb_1d(jaccs, a_pars):
    #jaccs is 2d
    if len(a_pars.shape) > 1 or len(jaccs.shape) > 2:
        raise ValueError("Only working for linear arrays")
    n_sims = jaccs.shape[0]
    n_pars = len(a_pars)
    weights = np.empty((n_pars, n_sims))
    for i in prange(n_sims):
        for j in prange(n_pars):
            weights[j,i] = _gauss_weight(jaccs[i], a_pars[j])
    return weights


def calc_likelihood(sidx, pars, sources, num_nodes, weight_fun=None):
    """
    Compute likelihood given the two jaccard indices
    sidx: [n_pars x n_sims x 2]
    """
    if weight_fun is None:
        if sidx.shape[0] == 1:
            weights = calc_g_weight_nb_1d(sidx[0], pars)
        else:
            weights = calc_g_weight_nb_2d(sidx, pars)
    else:
        weights = weight_fun(sidx, pars[:, np.newaxis])
    n_pars = len(pars)
    liklis = np.zeros((n_pars, num_nodes))
    if sources.shape[1] != sidx.shape[1]:
        raise ValueError("Incorrect number of sources")
    if sources.shape[0] != n_pars:
        sources = np.repeat(sources, n_pars, 0)
    ## sum the weights with numba
    compute_likelis_nb(liklis, weights, sources)
    return liklis

@njit(parallel=True)
def jaccard_sims_src_parall_a(sources, n, t_limit, contacts, mu, last_obs):
    """
    Calculate jaccard index for soft margin estimator
    summing the Jaccard Indices for I and R
    Sources with shape (n_pars x n_epi)
    """
    num_pars = sources.shape[0]
    num_epi = sources.shape[1]
    #end_conf = np.zeros(n,dtype=np.int8)
    vals_I = (last_obs == 1)
    vals_R = (last_obs == 2)
    #delays,epidemy,fathers = arrays

    jaccard_res = np.zeros((*sources.shape,2), dtype=np.float64)
    for j in prange(num_epi):
        for i in prange(num_pars):
            epidemy = np.empty(n)
            delays = np.empty(n)
            fathers = np.empty(n)
            epid_res = propagate.make_epidemy_inplace(sources[i, j], n, mu, contacts,
                                                      epidemy, delays, fathers)
            end_conf = propagate.get_status_t_numba(
                t_limit, epid_res[0], delays)

            jaccard_res[i, j][0] = get_jacc_conf_val(vals_I, end_conf,1)
            jaccard_res[i, j][1] = get_jacc_conf_val(vals_R, end_conf,2)

    return jaccard_res

@njit(parallel=True)
def run_sim_jaccard_src_single_a(sources, n, t_limit, contacts, mu, last_obs):
    """
    Calculate jaccard index for soft margin estimator
    summing the Jaccard Indices for I and R

    sources: array of sources, 1D, length: number of sims
    """
    num_epi = len(sources)
    vals_I = (last_obs == 1)
    vals_R = (last_obs == 2)

    jaccard_res = np.zeros((num_epi,2), dtype=np.float64)

    for i in prange(num_epi):
        epidemy = np.empty(n)
        delays = np.empty(n)
        fathers = np.empty(n)
        epid_res = propagate.make_epidemy_inplace(sources[i], n, mu, contacts,
                                                  epidemy, delays, fathers)
        end_conf = propagate.get_status_t_numba(t_limit, epid_res[0], delays)

        jaccard_res[i][0] = get_jacc_conf_val(vals_I, end_conf,1)
        jaccard_res[i][1] = get_jacc_conf_val(vals_R, end_conf,2)
        #p += np.exp(-(j-1)**2/a2)
    return jaccard_res

@njit(parallel=True)
def compute_likelis_nb(likls_arr, weights, sources):
    n_pars = likls_arr.shape[0]
    num_nodes = likls_arr.shape[1]
    n_epi = sources.shape[1]
    for i in prange(num_nodes):
        for l in prange(n_pars):
            count = 0
            tot = 0.
            for v in prange(n_epi):
                if sources[l, v] == i:
                    count += 1
                    tot += weights[l, v]
            if count != 0:
                likls_arr[l, i] = tot/count

#### SIMULATIONS FOR THE SPARSE OBSERVATIONS CASE ARE REMOVED