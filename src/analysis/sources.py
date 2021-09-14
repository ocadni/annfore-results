import numpy as np

def get_source_candidates(all_data_epigen):
    """
    Find the source candidates (the ones who have not been found infected)
    Checks the final configurations (from data_["test"])
    """
    candids = {s:
     [np.where(np.array(c[1])!=0)[0] for c in mdata["test"] ]
           for s, mdata in all_data_epigen.items()}
    return candids

def get_src_posit_obs_margs(margs, msources, candids):
    """
    Get the source position (fraction of the number of candidates)
    Uses marginal distribution in shape N x T x q
    """
    psources = margs[:,0,1][candids]
    idx=candids[psources.argsort()[::-1]]
    #print(idx)
    pos = np.mean([np.argmax(idx == s) for s in msources])
    return pos/len(candids)
 