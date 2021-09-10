import numpy as np
import pandas as pd
from sklearn.metrics import auc

import utils.common as common_utils
import sys
sys.path.insert(0, '../')

sort_I = common_utils.sort_by_inf

def sources_finder(ress, data_, num_conf, nsims, keys=["regressive", "sib", "sm"], ):
    marginals = {}
    Is = []   
    for k in keys:
        marginals[k] = []
    marginals["true_source"] = []
    marginals["rnd_I"] = []
    marginals["rnd"] = []
    pos_sources = {}
    for k in marginals:
        pos_sources[k]=[]
    for seed in ress:
        #print(seed)
        for i in range(num_conf[seed]):
            source=np.argmin(data_[seed]["epidemy"][i][0][0])
            marginals["true_source"].append(source)
            Is.append(np.where(data_[seed]["epidemy"][i][0][0] != np.inf)[0])
            S = np.where(data_[seed]["epidemy"][i][0][0] == np.inf)[0]
            I_shuffle = Is[-1].copy()
            np.random.shuffle(I_shuffle)
            IS_rnd = np.concatenate((I_shuffle, S))
            IS_rnd_all = IS_rnd.copy()
            np.random.shuffle(IS_rnd_all)
            
            marginals["rnd_I"].append(IS_rnd)
            marginals["rnd"].append(IS_rnd_all)
            
            pos_sources["rnd_I"].append(np.argwhere(marginals["rnd_I"][-1]==source)[0][0])
            pos_sources["rnd"].append(np.argwhere(marginals["rnd"][-1]==source)[0][0])

            for k in keys:
                if k != "sm":
                    mm = ress[seed][i][k]["marginals"]
                    mm[S,0, 1] = -1 #removing S from sorting
                    marginals[k].append(common_utils.sort_by_inf(mm, 0))
                    pos_sources[k].append(np.argwhere(marginals[k][-1][:,1]==source)[0][0])
                else:
                    marginals[k].append({})
                    pos_sources[k].append({})
                    try:
                        for nsim in ress[seed][i][k]:
                            #print(nsim)
                            marginals[k][-1][nsim] = []
                            pos_sources[k][-1][nsim] = []
                            for alpha in range(ress[seed][i][k][nsim]["prob_zero"].shape[0]):
                                marg_sm_i = ress[seed][i]["sm"][nsim]["prob_zero"][alpha]
                                marg_sm=np.zeros((len(marg_sm_i),1, 2))
                                marg_sm[:,0,1] = marg_sm_i
                                marginals[k][-1][nsim].append(np.array(sort_I(marg_sm, 0)))
                                pos_sources[k][-1][nsim].append(
                                    np.argwhere(marginals[k][-1][nsim][-1][:,1]==source)[0][0])
                            pos_sources[k][-1][nsim] = np.array(pos_sources[k][-1][nsim])
                    except:
                        print(f" seed:{seed} k:{k} nsim:{nsim} not found")
    for k in pos_sources:
        if k != "sm":
            pos_sources[k] = np.array(pos_sources[k])
    
    Is_len = [len(x) for x in Is]
    pos_sources["Is"] = Is
    pos_sources["Is_len"] = Is_len

    
    return marginals, pos_sources


def count_values_arrs(data_array, max_instance=None):
    """
    Return counts of the values, row by row of the input matrix
    """

    if not isinstance(data_array,np.ndarray):
        data_array = np.array(data_array)
    
    counts = np.zeros((data_array.shape[0],data_array.max()+1), dtype=np.int)
    for i, arr in enumerate(data_array):
        vals_, counts_ = np.unique(arr,return_counts=True)
        for l in range(len(vals_)):
            counts[i,vals_[l]] = counts_[l]

    if max_instance is not None and isinstance(max_instance,int):
        return counts[:max_instance]
    else:
        return counts

def plot_patient_zero_roc(plt,
                          pos_sources, 
                          nsims, 
                          args,
                          alpha=5, 
                          bins=10,
                          range_=(0,1),
                          norm=1,
                          colors=None,
                         rnd=False):
    if colors == None:
        colors=plt.get_cmap("Greens")
    num_inst=len(pos_sources["regressive"])
    s_nn,edge = np.histogram(pos_sources["regressive"]/norm, bins=bins, range=range_)
    s_sib,edge = np.histogram(pos_sources["sib"]/norm, bins=bins, range=range_)
    s_rnd,edge = np.histogram(pos_sources["rnd"]/norm, bins=bins, range=range_)
    s_rnd_I,edge = np.histogram(pos_sources["rnd_I"]/norm, bins=bins, range=range_)

    y_nn = np.insert(np.cumsum(s_nn)/num_inst,0,0)
    y_sib= np.insert(np.cumsum(s_sib)/num_inst,0,0)
    y_rnd = np.insert(np.cumsum(s_rnd)/num_inst,0,0)
    y_rnd_I = np.insert(np.cumsum(s_rnd_I)/num_inst,0,0)

    x = edge
    cl=0
    if rnd: plt.plot(x, y_rnd, "--", label=f"random -- auc: {auc(x, y_rnd):.3f}", color="black")
    plt.plot(x, y_rnd_I, "--", label=f"random (only I) -- auc: {auc(x, y_rnd_I):.3f}", color="black")
    plt.plot(x, y_sib, "-.", label=f"sib -- auc: {auc(x, y_sib):.3f}", linewidth="2")
    plt.plot(x, y_nn, label=f"nn -- auc: {auc(x, y_nn):.3f}", linewidth="2")

    #try:
    for i, nsim in enumerate(nsims):
        pos_source_sm=[pos_sources["sm"][ii][nsim][alpha] for ii in range(len(pos_sources["sm"]))]
        s_sm,edge = np.histogram(np.array(pos_source_sm)/norm, bins=bins, range=range_)
        y_sm = np.insert(np.cumsum(s_sm)/num_inst,0,0)
        plt.plot(x, y_sm, ":",
                 label=f"sm - {args.a_min+alpha*args.a_step:.2f} -- auc {auc(x, y_sm):.3f} -- sims:{nsim:.0e}", 
                 color=colors(np.clip(i/len(nsims), 0.3, 0.9)), lw=2)
    #except:
    #    pass
    return plt


def find_accuracy_source(marginals, true_sources, num_n=20, source_p_index=1):
    """
    Compute the accuracy in finding the source from marginals
    """
    if len(marginals) != len(true_sources):
        raise ValueError("Need one source for each instance (first index in marginals is max_i)")
    if isinstance(marginals, list):
        ## try stacking the marginals (with np arrays)
        marginals = np.stack(marginals)
    ## marginals now  should be in shape MAX_I, N, NUM_STATES
    if len(marginals.shape) > 2:
        sources_p = marginals[:,:,source_p_index]
    else:
        sources_p = marginals

    indicis = np.argsort(sources_p)[:,::-1]
    
    corr = indicis == np.array(true_sources)[:,np.newaxis]

    ranks = corr.cumsum(-1).mean(0)*100

    return ranks[:num_n], indicis[:,:num_n], [pd.Series(pis) for pis in sources_p]


def get_ann_source_margs(data):
    """
    Give the source marginals from the `val_sources` given in
    `read_sources_nn`, that is, using pandas Series
    """
    return np.stack([v.sort_index().to_numpy().flatten() for v in data])

def plot_sources_sct(axis, x_src, y_src, label, mark="o", color=None, normed=False, ax_norm=-1):
    """
    Plot comparison of source probabilities,
    and put the norm 1 distance in the label
    """
    if ax_norm < 0:
        ax_norm += len(x_src.shape)
        print(ax_norm)
    if normed:
        x_src = common_utils.normalize(x_src, ax_norm)
        y_src = common_utils.normalize(y_src, ax_norm)
    d = np.abs(x_src - y_src).sum()
    axis.scatter(x_src.flatten(), y_src.flatten(), marker=mark, c=color, label=label+ f" - {d:3.2f}")

def read_sources_softmargin(res_folder,base_name,a_value,max_i,num_n_res,sources,par_format="{:3.2e}",name_probs="poster_all_a.npz"):
    poster_p = np.load(res_folder/(base_name+name_probs))
    indx = ""
    name_try = "a="+par_format.format(a_value)
    for i,name in enumerate(poster_p.files):
        #param = float(name[2:])
        
        if name_try == name:
            indx = name
            break
    if indx == "":
        print(poster_p.files)
        raise ValueError("Parameter value {} not found in the results with name {}".format(a_value,name_try))

    poster = poster_p[indx]
    ## poster is n_inst x n
    poster = poster[:max_i]
    n = poster.shape[-1]
    src_guess = np.flip(poster.argsort(-1),-1)
    src_true = np.array(sources)[:max_i]
    ROC = np.cumsum(src_guess==src_true[:,np.newaxis],-1)
    #print(src_guess.shape)
    curves = ROC.mean(0)
    #print(curves.shape)
    
    indics = np.repeat(np.arange(n)[:,np.newaxis],max_i).reshape(n,max_i).T
    probs_stacked = np.stack([indics,poster],-1)

    for i in range(probs_stacked.shape[0]):
        probs_stacked[i].view("f8,f8").sort(order="f1",axis=0)

    probs_stacked = np.flip(probs_stacked,1)
    #print(probs_stacked.shape)

    
    return curves[:num_n_res]*100, src_guess, probs_stacked

