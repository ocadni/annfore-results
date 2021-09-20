import numpy as np
import numba as nb
from . import run_sim
import soft_margin.utils as softm_utils
from epigen import propagate
from utils.common import pretty_print_n, normalize

newax = np.newaxis

warn = softm_utils.warn

#categorize_obs = run_sim.get_categ_I_R_numba
get_jacc = run_sim.get_jacc_conf_val
jaccard_idx_nb = run_sim.jaccard_idx_nb
gaussian_weight_nb = run_sim._gauss_weight

def run_oneit_softm_margs(a_pars, p_sources, n_epi_per_a, contacts, inst, last_obs,
                              use_partial_obs=False, overw_log=False, prev_message=None):
    """
    a_pars: parameter for the weight, M parameters
    p_sources: source probabilities, MxN,  N number of nodes
    put same_p_sources=True if you do only one iteration of softmargin
    """

    def myprint(msg):
        if prev_message is not None:
            msg = prev_message + msg
        
        if overw_log:
            print(msg,end="\r")
        else:
            print(msg)

    if use_partial_obs:
        raise NotImplementedError("Running with sparse observations and single source is not allowed yet")


    fun_sample = softm_utils.sample_probs_multi
    n_pars = len(a_pars)

    if prev_message is not None:
         myprint(" "*len("Computing likelihoods and posterior"))
    
    myprint("Extracting sources...")
    sources = fun_sample(p_sources, n_epi_per_a)
    if np.any(np.isnan(sources)):
        warn("Sources are nan")

    myprint("Running simulations")
    one_hot_c_sum, weights = calc_weights_margs_nb(sources,  a_pars, inst.n, inst.t_limit,
                                         contacts, inst.mu, last_obs)
    weights_sum = weights.sum(1)
    if np.any(np.isnan(weights)):
        warn("Weights are nan")

    marginals = one_hot_c_sum / weights_sum[:, newax, newax, newax]
    
    myprint("Computing likelihoods and posterior")
        # adjust shape
    sources = sources[np.newaxis, :]
    sources = np.repeat(sources, n_pars, 0)
    p_sources = p_sources[np.newaxis, :]

    likelis = np.zeros((len(a_pars), inst.n))
    #if sources.shape[0] != len(a):
        
    #likelis = calc_likelihood_pars(simils_idx, a_pars, sources, inst.n)
    run_sim.compute_likelis_nb(likelis, weights, sources)
    if np.any(np.isnan(likelis)):
        warn("Likelihoods are nan")

    posters = normalize(likelis*p_sources, 1)
    if np.any(np.isnan(posters)):
        warn("Posteriors are nan")
    if overw_log and prev_message is None:
        print("")
        print("Done")

    return posters, marginals, weights_sum


@nb.njit(parallel=True)
def calc_weights_margs_nb(sources, a_pars, n, t_limit, contacts, mu, last_obs):
    """
    Calculate jaccard index for soft margin estimator
    summing the Jaccard Indices for I and R
    Sources with shape (n_pars x n_epi)

    one_hot_confs: array to save the one_hot encoded result
    dim (num_pars, t_limit+1, n, 3) (S, I, R)
    """
    num_pars = a_pars.shape[0]
    num_epi = sources.shape[0]
    #end_conf = np.zeros(n,dtype=np.int8)
    #set_true_I, set_true_R = categorize_obs(last_obs)
    vals_I = (last_obs == 1)
    vals_R = (last_obs == 2)
    #delays,epidemy,fathers = arrays

    final_conf = np.zeros((num_pars, t_limit+1, n, 3))
    #weights_sum = np.zeros(num_pars)
    weights_all = np.zeros((num_pars,num_epi))
    for j in nb.prange(num_epi):
        one_hot_c_sum = np.zeros_like(final_conf)
        ## INIT SIM
        epidemy = np.empty(n)
        delays = np.empty(n)
        fathers = np.empty(n)
        #one_hot_c = np.zeros((t_limit+1, n, 3))
        ## RUN SIM
        epid_res = propagate.make_epidemy_inplace(sources[j], n, mu, contacts,
                                                    epidemy, delays, fathers)
        end_conf = propagate.get_status_t_numba(
            t_limit, epid_res[0], delays)
        j_indics = (get_jacc(vals_I, end_conf,1), get_jacc(vals_R, end_conf, 2))

        ## save conf
        #softm_utils.fill_traj_one_hot_nb(one_hot_c,epid_res[0], delays)
        ## compute weight
        for i in nb.prange(num_pars):
            weight = gaussian_weight_nb(j_indics, a_pars[i])
            ## save conf directly
            softm_utils.fill_traj_one_hot_nb_val(one_hot_c_sum[i], epid_res[0], delays, weight)
            weights_all[i,j] = weight
        
        #weights_sum += m_weights
        final_conf += one_hot_c_sum

    return final_conf, weights_all


def run_softm_manysims(a_pars, p_sources, num_sims_all, contacts, inst, last_obs, 
            overw_log=False, prev_message=None):
    """
    Run softmargin on only one iteration, for different values 
    of the number of epidemies to simulate

    DOES NOT COMPUTE MARGINALS

    a_pars: parameter for the weight, M parameters
    p_sources: source probabilities, MxN,  N number of nodes
    num_sims_all: all the numbers of epidemies needed [1000 epis, 2000 epis, etc]
    put same_p_sources=True if you do only one iteration of softmargin
    """

    def myprint(msg):
        if prev_message is not None:
            msg = prev_message + msg
        
        if overw_log:
            print(msg,end="\r")
        else:
            print(msg)

    fun_similarity = run_sim.run_sim_jaccard_src_single_a

    fun_sample = softm_utils.sample_multi_1d_nb
    

    if prev_message is not None:
         myprint(" "*len("Computing likelihoods and posterior"))
    ## sort the requested number of sims
    num_sims_all=np.sort(num_sims_all).astype(int)
    ## biggest number of sims
    n_epi_per_a = num_sims_all[-1]
    
    myprint("Extracting sources...")
    sources = fun_sample(p_sources, n_epi_per_a)
    if np.any(np.isnan(sources)):
        warn("Sources are nan")

    myprint("Running simulations")
    simils_idx = fun_similarity(sources, inst.n, inst.t_limit,
                                         contacts, inst.mu, last_obs)
    if np.any(np.isnan(simils_idx)):
        warn("Jaccards are nan")
    
    np.savez("simils_idx", similarity=simils_idx)
    ### SINGLE ITER: simils is 1D array -> need to n_repeatbroadcast on zeroth dimension
    posteriors = []
    p_sources = p_sources[np.newaxis, :]
    for nsim in num_sims_all:
        myprint(f"Computing likelis and poster for {pretty_print_n(nsim)}")
        msimidx = simils_idx[np.newaxis, :nsim]
        msources = sources[np.newaxis, :nsim]
        
        likelis = run_sim.calc_likelihood(msimidx, a_pars, msources, inst.n)
        if np.any(np.isnan(likelis)):
            warn("Likelihoods are nan")
        #myprint(f"Likelihoods are {likelis.shape}")

        posters = normalize(likelis*p_sources, 1)
        if np.any(np.isnan(posters)):
            warn("Posteriors are nan")
        posteriors.append(posters)
    if overw_log and prev_message is None:
        print("")
        print("Done")

    return dict(zip(num_sims_all, posteriors))