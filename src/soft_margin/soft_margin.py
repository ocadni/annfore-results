"""
Additional soft margin methods and classes

Author: Fabio Mazza
"""
import warnings
from pathlib import Path

import numpy as np
from numba import njit, prange
import numba as nb

from epigen.propagate import make_epidemy_inplace
import soft_margin.utils as softm_utils
import io_m.libsaving as libsaving
import io_m.io_utils as io_utils
from utils.common import normalize

from .saving import SoftMarginSaver, load_data_softmargin
from . import run_sim
import soft_margin.singleiter as singleit

### imported methods from old similarity module
def build_last_obs_src(inst_run_data):

    last_obs = inst_run_data[1]
    true_init = inst_run_data[0]
    last_obs = np.array(last_obs)
    true_src = np.where(true_init)[0]
    return last_obs, true_src[0]



def calc_accu(poster_src, src):
    """
    Calculate accuracy curves
    """
    maxsrc = np.flip(poster_src.argsort(-1), -1)
    return np.cumsum(maxsrc == src, -1)



class SoftMarginRunner:
    """
    Class for running soft margin
    """
    STATE_S = 0
    KEY_TOT_SIMS_MAR = "tot_sims_marg"
    def __init__(self, instance, inst_data, contacts,
                 a_par_min=0.1, a_par_max=0.2, a_par_step=0.01,
                 num_iter=1, n_sims_per_a=10000, use_obs=False):
        self.inst = instance
        self.using_obs = use_obs
        if not isinstance(use_obs, bool):
            raise ValueError("Parameter use_obs has to be boolean")
        if use_obs:
            warnings.warn("Usage with partial observations from different times\
                 has yet to be checked to run correctly")
        self._create_last_obs(inst_data, use_obs)
        self.contacts = contacts

        self.set_a_pars(a_par_min, a_par_max, a_par_step)
        
       
        self.marginals = [tuple() for _ in range(self.num_inst)]
        self.n_sims_per_a = n_sims_per_a
        try:
            len(n_sims_per_a)
            self.manysims = True
        except:
            self.manysims = False
        if self.manysims:
            self.final_p_sources = {n: {} for n in n_sims_per_a}
        else:
            self.final_p_sources = {}
        self.n_iter = num_iter
        self.info = {}
        
        print(f"Setup for {instance.type_graph} {instance.n} d {instance.d} complete")

    def set_a_pars(self, a_par_min, a_par_max, a_par_step):
        # Self descriptive method
        self.a_pars = np.arange(a_par_min, a_par_max+a_par_step/4, a_par_step)

    def _create_last_obs(self, inst_data, use_obs):
        """
        Setup for the algorithm,
        creating the observations and the sources for each instance
        """
        self.l_obs = []
        self.sources = []
        dataset = inst_data["test"]
        if use_obs:
            obs = inst_data["obs"]
            if isinstance(obs, list):
                iterat = enumerate(obs)
            elif isinstance(obs,dict):
                iterat = obs.items()
            else:
                raise ValueError
            for _,o in iterat:
                self.l_obs.append(convert_obs_to_list(o))
            for init, _ in dataset:
                self.sources.append(np.where(init)[0][0])
        else:
            for obs, src in map(build_last_obs_src, dataset):
                self.l_obs.append(obs)
                self.sources.append(src)

        self.num_inst = len(self.sources)

    def run_softmargin(self, n_inst, find_margs=False, chain_msg=None):
        """
        Run soft margin algorithm
        """
        p_sources = np.full((self.inst.n), 1/self.inst.n)
        
        idx_sus = np.where(self.l_obs[n_inst] == self.STATE_S)[0]

        p_sources[idx_sus] = 0.
        p_sources /= p_sources.sum()  ## normalize p sources
        if find_margs and self.n_iter == 1:
            # bypass the calculation
            if self.manysims:
                raise ValueError("Cannot run marginals directly with multiple nsims")
            prev_msg = chain_msg+" - Single iter -" if chain_msg is not None else None
            p_src, margs, weights = singleit.run_oneit_softm_margs(self.a_pars,
                            p_sources, self.n_sims_per_a, self.contacts, self.inst,
                            self.l_obs[n_inst], overw_log=True, prev_message=prev_msg)
            self.final_p_sources[n_inst] = p_src
            self._check_margs(self.n_sims_per_a)
            self.marginals[n_inst] = margs
            return margs, weights

        elif self.n_iter == 1 and self.manysims:
            ## we are running on one iter, different values of nsims
            if chain_msg is None:
                prev_msg = f"Iterat 0: "
            else:
                prev_msg = chain_msg + f" - iter 0: "
            probsdict = singleit.run_softm_manysims(self.a_pars, p_sources,
                                self.n_sims_per_a, self.contacts, self.inst,
                                self.l_obs[n_inst], overw_log=True, prev_message=prev_msg)
            for nsims, probs in probsdict.items():
                self.final_p_sources[nsims][n_inst] = probs
            if find_margs:
                raise NotImplementedError("NOT extracting marginals for many values of nsims")
        else:
            for i in range(self.n_iter):
                if chain_msg is None:
                    prev_msg = f"Iterat {i}: "
                else:
                    prev_msg = chain_msg + f" - iter {i}: "

                same_sources = (i==0)
                p_sources = run_iteration_soft_margin(self.a_pars, p_sources, self.n_sims_per_a,
                                                    self.contacts, self.inst, self.l_obs[n_inst],
                                                    same_p_sources=same_sources,
                                                    use_partial_obs=self.using_obs, overw_log=True,
                                                    prev_message=prev_msg)
            self.final_p_sources[n_inst] = p_sources
            if find_margs:
                print("Extracting marginals")
                return self.extract_marginals(n_inst, saving=True)
        if chain_msg is None:
            print("\nFinished")

    
    def calc_accuracy_avg(self):
        """
        Calculate average accuracy on finished instances
        """
        done_inst_n = len(self.final_p_sources.keys())
        res = []
        print("Averaging on {} instances".format(done_inst_n))
        for inst, p_sources in self.final_p_sources.items():
            src = self.sources[inst]
            res.append(calc_accu(p_sources, src))
        res = np.stack(res)
        return res.mean(0)

    def run_soft_margin_all(self, calc_margs=False):
        """
        Run soft margin algorithm on all instances of the problem
        """
        
        for i in range(self.num_inst):
            ch_msg = "  Instance {}".format(i)
            #print(ch)
            self.run_softmargin(i, find_margs=calc_margs, chain_msg=ch_msg)

    def extract_marginals(self, inst_idx, n_sims=None, n_iter=1,
                              use_partial_obs=False, saving=False,
                              nsims_probs_sel=None):
        """
        a_pars: parameter for the weight, M parameters
        p_sources: source probabilities, MxN,  N number of nodes
        put same_p_sources=True if you do only one iteration of softmargin

        Marginals have shape (n_pars, t_limit+1, n, 3)
        """
        if nsims_probs_sel is not None:
            p_sources = self.final_p_sources[nsims_probs_sel][inst_idx]
        else:
            p_sources = self.final_p_sources[inst_idx]
        a_pars = self.a_pars
        if n_sims is None:
            if self.manysims:
                n_sims = self.n_sims_per_a[-1]
            else:
                n_sims = self.n_sims_per_a
        contacts = self.contacts
        overw_log = n_iter > 1

        margs, weights = extract_marginals(p_sources=p_sources, a_pars=a_pars, n_epi_per_a=n_sims, inst=self.inst,
                                contacts=contacts, last_obs=self.l_obs[inst_idx],
                                use_partial_obs=use_partial_obs,
                                overw_log=overw_log)
        newx = np.newaxis
        i=0
        for _ in range(1, n_iter):
            new_margs, new_w = extract_marginals(p_sources, a_pars, n_sims,
                                    self.inst, contacts, self.l_obs[inst_idx],
                                    use_partial_obs=use_partial_obs,
                                    overw_log=overw_log)
            i+=1
            newsum_w = new_w + weights
            margs = (margs * weights[:,newx, newx, newx] 
                        + new_margs *new_w[:, newx, newx, newx]) / newsum_w[:, newx, newx, newx]
            weights = newsum_w
        if i > 0:
            print("Finished", " "*24)
        if saving:
            self._check_margs(n_sims)
            self.marginals[inst_idx] = margs
        return margs, weights
    
    def _check_margs(self, tot_sims):
        """
        Check the number of simulations used for the
        marginals is consistent across different runs
        """
        if self.KEY_TOT_SIMS_MAR in self.info:
            if self.info[self.KEY_TOT_SIMS_MAR] != tot_sims:
                raise ValueError("Using a different number of simulations than before")
        else:
            self.info[self.KEY_TOT_SIMS_MAR] = tot_sims


            
class SoftMarginDirectRunner(SoftMarginRunner):
    """
    Version with custom file saving of SoftMarginRunner
    """
    #KEY_TOT_SIMS = "tot_sims"
    def save_data(self, file_name: str, overwrite: bool=False, extra_pars: dict=None, nsims_probs: int = None):
        """
        Save run data
        nsims_probs: value of nsims for which to save the data
        """
        pars = {
            "n_iter": self.n_iter,
            
        }
        if extra_pars is not None:
            pars.update(extra_pars)
        fname_pars = file_name +"_pars.json"
        fname_data = file_name + "_probs.npz"
        if self.KEY_TOT_SIMS_MAR in self.info:
            pars["n_epi_marginals"] = self.info[self.KEY_TOT_SIMS_MAR]

            fname_margs = file_name + "_margs.npz"
            SoftMarginSaver.save_margs_(self.marginals,
                fname_margs, self.num_inst, overwrite=overwrite)
        if not self.manysims:
            pars["epi_per_it"]=self.n_sims_per_a
            ## nsims_probs is ignored
            SoftMarginSaver.save_apars_srcprobs(self.a_pars,
                self.final_p_sources, fname_data, overwrite=overwrite)
        else:
            
            if nsims_probs not in self.final_p_sources.keys():
                raise ValueError("Cannot find the correct nsims")
            pars["epi_per_it"] = int(nsims_probs)
            SoftMarginSaver.save_apars_srcprobs(self.a_pars,
                self.final_p_sources[nsims_probs], fname_data, overwrite=overwrite)

        io_utils.save_json(fname_pars, pars)

    def calc_all_margs(self, n_sims=None, n_iter=1):
        """
        Compute all marginals running more sims
        """

        if n_sims is None:
            n_sims = self.n_sims_per_a
        tot_sims = n_sims * n_iter
        self._check_margs(tot_sims)
        for i in range(self.num_inst):
            self.extract_marginals(i, n_sims=n_sims, n_iter=n_iter, saving=True)


class SoftMarginRunnerSaver(SoftMarginRunner):
    def __init__(self, instance, inst_data, contacts, results_root_fold,
                 a_par_min=0.1, a_par_max=0.2, a_par_step=0.01,
                 num_iter=1, n_sims_per_a=10000, use_obs=False):

        super().__init__(instance, inst_data, contacts,
        a_par_min, a_par_max, a_par_step, num_iter,
        n_sims_per_a=n_sims_per_a, use_obs=use_obs)

        fold = Path(results_root_fold)
        if not fold.exists():
            warnings.warn(f"Folder: {fold.resolve().as_posix()} doesn't exist")

        self.saver = SoftMarginSaver(instance,
            results_root_fold, n_sims_per_a, num_iter)

    def run_softmargin(self, n_inst, chain_msg=None, autosave=False):

        super().run_softmargin(n_inst, chain_msg=chain_msg)

        if autosave:
            warnings.warn("Autosave on: overwriting old results")
            self.saver.save_src_probabilities(self.final_p_sources,
            self.a_pars, overwrite=True)
            self.saver.save_params()
        
    def run_soft_margin_all(self, autosave=True):
        """
        Run soft margin algorithm on all instances of the problem
        """
        
        for i in range(self.num_inst):
            print("\t Instance {}".format(i))
            self.run_softmargin(i, autosave=autosave)
        
        if not autosave:
            print("Saving results...")
            self.saver.save_src_probabilities(self.final_p_sources,
            self.a_pars, overwrite=True)
            self.saver.save_params()


@njit()
def make_epidemy_nb(src, n, mu, contacts):
    epidemy = np.empty(n)
    delays = np.empty(n)
    fathers = np.empty(n)
    epid_res = make_epidemy_inplace(src, n, mu, contacts,
                                              epidemy, delays, fathers)
    return epid_res, delays



def warn(msg):
    warnings.warn(msg, RuntimeWarning)
def run_iteration_soft_margin(a_pars, p_sources, n_epi_per_a, contacts, inst, last_obs,
                              same_p_sources=False,
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
        if same_p_sources:
            raise NotImplementedError("Running with sparse observations and single source is not allowed yet")
        fun_similarity = run_sim.correct_sims_src_parall_obs
    else:
        if same_p_sources:
            fun_similarity = run_sim.run_sim_jaccard_src_single_a
        else:
            fun_similarity = run_sim.jaccard_sims_src_parall_a

    if same_p_sources:
        #assert len(p_sources.shape) == 1
        fun_sample = softm_utils.sample_multi_1d_nb
    else:
        fun_sample = softm_utils.sample_probs_multi_nb
        if np.any(p_sources.sum(1) <1e-5):
            warn("p_sources with very low p")
            #print("\n", p_sources)

    if prev_message is not None:
         myprint(" "*len("Computing likelihoods and posterior"))

    if not same_p_sources and p_sources.shape[0] != len(a_pars):
        raise ValueError("Shape mismatch, p_sources are {}, but a_pars are {}"\
            .format(p_sources.shape, len(a_pars)))
    
    
    myprint("Extracting sources...")
    sources = fun_sample(p_sources, n_epi_per_a)
    if np.any(np.isnan(sources)):
        warn("Sources are nan")

    myprint("Running simulations")
    simils_idx = fun_similarity(sources, inst.n, inst.t_limit,
                                         contacts, inst.mu, last_obs)
    if np.any(np.isnan(simils_idx)):
        warn("Jaccards are nan")

    ### SINGLE ITER: simils is 1D array -> need to n_repeatbroadcast on zeroth dimension
    
    myprint("Computing likelihoods and posterior")
    if same_p_sources:
        # adjust shape
        simils_idx = simils_idx[np.newaxis, :]
        sources = sources[np.newaxis, :]
        p_sources = p_sources[np.newaxis, :]
    likelis = run_sim.calc_likelihood(simils_idx, a_pars, sources, inst.n)
    if np.any(np.isnan(likelis)):
        warn("Likelihoods are nan")

    posters = normalize(likelis*p_sources, 1)
    if np.any(np.isnan(posters)):
        warn("Posteriors are nan")
    if overw_log and prev_message is None:
        print("")
        print("Done")

    return posters


### OBSERVATIONS ###
def convert_obs_to_list(obs_json):
    states = {"S":0,"I":1,"R":2}
    r = []

    for st, tnodes in obs_json.items():
        for t,nodes in tnodes.items():
            for n in nodes:
                r.append((t,states[st],n))
    return sorted(r)


def extract_marginals(p_sources, a_pars, n_epi_per_a, inst, contacts, last_obs,
                            use_partial_obs=False, overw_log=False):
    """
    a_pars: parameter for the weight, M parameters
    p_sources: source probabilities, MxN,  N number of nodes
    
    Extract marginals from p_sources
    """
    def myprint(msg):
        if overw_log:
            print(msg,end="\r")
        else:
            print(msg)

    if use_partial_obs:
        raise NotImplementedError("Running with sparse observations is not allowed yet")

    fun_sample = softm_utils.sample_probs_multi_nb
    newx = np.newaxis

    shape_single_margs = (inst.t_limit+1, inst.n, 3)


    myprint(" "*len("Computing likelihoods and posterior"))
    if p_sources.shape[0] != len(a_pars):
        raise ValueError("Shape mismatch, p_sources are {}, but a_pars are {}"\
            .format(p_sources.shape, len(a_pars)))
    sump = p_sources.sum(1)
    if np.any(sump <1e-7):
        mask =sump<1e-7
        warn("p_sources with very low p: {} at {}".format(sump[mask], np.where(mask)[0]))
        
    
    myprint("Extracting sources...")
    sources = fun_sample(p_sources, n_epi_per_a)
    if np.any(np.isnan(sources)):
        warn("Sources are nan")

    myprint("Running simulations")
    final_margs, weights_sum = run_sim.calc_margs_softm(sources, a_pars,
                                inst.n, inst.t_limit, contacts, inst.mu, last_obs)
    final_margs /= weights_sum[:,newx, newx, newx]
    #final_margs = (confs_one_hot * weights[:,:,newx, newx, newx]).sum(1) / weights.sum(1)[:,newx,newx,newx]

    #weights_sum = weights_sum
    assert final_margs.shape == (len(a_pars), *shape_single_margs)
    if overw_log:
        myprint(" "*25)
        myprint("Done")

    return final_margs, weights_sum


## UNUSED FUNCTIONS
@njit(parallel=True)
def _compute_margs(final_margs, weights, epids_one_hot):
    n_pars = weights.shape[0]
    n_epis = weights.shape[1]

    if epids_one_hot.shape[:2] != (n_pars, n_epis):
        raise ValueError
    
    
    for j in prange(n_epis):
        c = np.zeros_like(epids_one_hot[:,j])# * weights[:,j]
        for i in prange(n_pars):
            c[i,:] = epids_one_hot[i,j,:] * weights[i,j]
        
        final_margs += c
