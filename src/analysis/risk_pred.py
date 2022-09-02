"""
Epidemic risk predicition analysis tools
"""
from enum import Enum
import numpy as np

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from enum import Enum
class ResType(Enum):
    margs = "margs"
    risks = "risks"

def get_obs_idx(obs_df, states=(1,2)):
    """
    Get the index of the nodes which are observed in the mentioned states
    """
    r = None
    for l in states:
        if r is None:
            r = obs_df.obs == l
        else:
            r = r | (obs_df.obs == l)
    
    obs_i_r = set(obs_df.node[r])
    return obs_i_r

"""def get_err_margs(margs, nodes_idx, fin_conf):
    true_states = fin_conf[list(nodes_idx)]
    inf_r_p = margs[list(nodes_idx),-1]
    err_p = np.abs(1 - inf_r_p[np.arange(len(true_states)),true_states]).sum()
    err_p /= len(true_states)
    
    ndiff = sum(np.argmax(inf_r_p,-1) != true_states)
    return err_p, ndiff/len(true_states)"""

def get_err_rocs(margs, nodes_idx, fin_conf, states=(1,), t_obs=-1):
    """
    Calculate roc curve in finding nodes in state `state`
    with marginals `margs` (NxTxq)
    """
    true_states = fin_conf[list(nodes_idx)]
    inf_r_p = margs[list(nodes_idx),t_obs]
    tr = [true_states == s for s in states]
    if len(tr) > 1:
        valid = np.logical_or(*tr)
    else:
        valid=tr[0]
    prs = inf_r_p[:,states]
    if len(prs.shape) > 1:
        prs = prs.sum(-1)
    roc = roc_curve(valid.astype(np.int8), prs)
    
    return roc



"""def calc_margs_diff(margs, fin_conf, obs_all_df, instance):
    margs_diff =[]
    for i, conf_fin in enumerate(fin_conf):
        nidx = get_obs_idx(obs_all_df[i])
        sel_idx = set(range(instance.n)).difference(nidx)
        errs = get_err_margs(margs[i], sel_idx, conf_fin)
        margs_diff.append(errs)
        
    return margs_diff
"""
def calc_aucs(margs, fin_conf, obs_all_df, instance, t_obs=-1,
            st_exclude=(1,2), st_find=(1,)):
    """
    Calculate AUCs on the marginals, 
    excluding those with observed state in `st_exclude`,
    finding the nodes with state in `st_find` at time `t_obs`
    """
    resu =[]
    n_inst = min(len(fin_conf), len(margs))
    for i, conf_fin in enumerate(fin_conf[:n_inst]):
        nidx = get_obs_idx(obs_all_df[i], states=st_exclude)
        sel_idx = set(range(instance.n)).difference(nidx)
        errs = get_err_rocs(margs[i], sel_idx, conf_fin, states=st_find, t_obs=t_obs)
        resu.append(auc(errs[0],errs[1]))
        
    return resu

def calc_all_rocs(margs, fin_conf, obs_all_df, instance):
    """
    Calculate all roc curves
    """
    resu =[]
    n_inst = min(len(fin_conf), len(margs))
    for i, conf_fin in enumerate(fin_conf[:n_inst]):
        nidx = get_obs_idx(obs_all_df[i])
        sel_idx = set(range(instance.n)).difference(nidx)
        errs = get_err_rocs(margs[i], sel_idx, conf_fin)
        resu.append(np.array(errs))
        
    return resu

def get_err_rocs_risks(risks, nodes_idx, fin_conf, state=1):
    true_states = fin_conf[list(nodes_idx)]
    valid = (true_states == state)
    
    
    prs = risks[list(nodes_idx)]
    #print(prs.index)

    roc = roc_curve(valid.astype(np.int8), prs)
    
    return roc

def calc_rocs_risk(risks_arr, fin_conf, obs_all_df, instance,
            st_exclude=(1,2), st_find=(1,)):
    """
    Calculate ROCs on the marginals
    excluding those with observed state in `st_exclude`,
    finding the nodes with state in `st_find` at time `t_obs`
    """
    resu =[]
    if len(st_find) > 1:
        raise ValueError("Only one state possible")
    n_inst = min(len(fin_conf), len(risks_arr))
    for i, conf_fin in enumerate(fin_conf[:n_inst]):
        nidx = get_obs_idx(obs_all_df[i], states=st_exclude)
        sel_idx = set(range(instance.n)).difference(nidx)
        errs = get_err_rocs_risks(risks_arr[i], sel_idx, conf_fin, state=st_find[0])
        resu.append(errs)
        
    return resu

def calc_aucs_risks(risks_arr, fin_conf, obs_all_df, instance,
            st_exclude=(1,2), st_find=(1,)):
    """
    Calculate AUCs on the marginals
    excluding those with observed state in `st_exclude`,
    finding the nodes with state in `st_find` at time `t_obs`
    """

    rocs= calc_rocs_risk(risks_arr, fin_conf, obs_all_df,
        instance=instance, st_exclude=st_exclude, st_find=st_find)

    resu = [auc(r[0],r[1]) for r in rocs]
        
    return resu



def get_probs_inf(arr_data, fin_conf, obs_all_df, instance, what,
            st_exclude=(1,2), st_find=(1,), t_obs=-1):
    """
    Calculate accuracy on the marginals
    excluding those with observed state in `st_exclude`,
    finding the nodes with state in `st_find` at time `t_obs`
    """
    resu =[]
    if len(st_find) > 1 and what != ResType.margs:
        raise ValueError("Only one state possible")
    n_inst = min(len(fin_conf), len(arr_data))
    for i, conf_fin in enumerate(fin_conf[:n_inst]):
        ## i is the epidemic index
        nidx = get_obs_idx(obs_all_df[i], states=st_exclude)
        sel_idx = list(
            set(range(instance.n)).difference(nidx)
        )
        true_states = conf_fin[sel_idx]
        valid = sum(true_states == s for s in st_find)
        valid = np.array(valid,dtype=np.bool_)
        if what == ResType.risks:
            prs = arr_data[i][sel_idx]
        elif what == ResType.margs:
            prs = arr_data[i][sel_idx,t_obs]
            prs = prs[:,list(st_find)]
            if len(prs.shape) > 1:
                prs = prs.sum(1)
        resu.append((prs, valid))
        
    return resu

def count_found(risk_i):
    print(risk_i.shape)
    g = np.stack(risk_i,1)

    g.view("f8,f8").sort(axis=0, order="f0")

    return g[::-1].T[1].cumsum()

def calc_rocs_aucs(probs,
        npoints=101,
        ):
    """
    Compute ROCs and AUCs, and
    save rocs by linear interpolation
    (returning statistics)
    """
    aucs_exp = []
    rocs_exp=[]
    
    xinter = np.linspace(0,1,npoints)
    for inst_seed in probs:
        for d in inst_seed:
            trues=d[1]
            ps = d[0]
            fpr,tpr,_ = roc_curve(trues,ps)
            mauc = auc(fpr,tpr)
            if np.isnan(mauc):
                #print("Skipping for NaN")
                ## skip interpolation
                continue
            aucs_exp.append(mauc)
            rocs_exp.append(np.insert(
                np.interp(xinter, fpr, tpr), 0 ,0.))
    rocs_exp = np.array(rocs_exp)
    return np.array(aucs_exp), rocs_exp
    #np.stack(
    #    (rocs_exp.mean(axis=0),rocs_exp.std(axis=0))), np.nanquantile(
    #        rocs_exp, quantiles, axis=0
    #ù    )