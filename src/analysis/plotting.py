import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from .generic import make_histo_cumsum
from utils.common import pretty_print_n


def plot_pzero_roc_new(pos_scaled_s, keys_plt, pos_scale_softm=None,
                    softm_alphas=None,softm_alpha_idx=14,
                    ax=None,
                    figsize=(7,5),
                    pars_histo=None,
                    cmap_softm="Greens",
                    cmap_lims=(0.4,0.8),
                    labels=False,
                    set_lims=True):

    if ax is None:
        f,ax = plt.subplots(figsize=figsize)
    if pars_histo is None:
        pars_histo = dict(nbins=100)

    for k,v in keys_plt.items():
        labl = v[0]
        style=v[1]
        d=make_histo_cumsum(pos_scaled_s[k],**pars_histo)
        if len(v) > 2:
            col = v[2]
        else:
            col = None
        ax.plot(*d, style, label=f"{labl} -- auc: {auc(*d):.3f}", linewidth="2", color=col)
    

    if pos_scale_softm!= None:
        nsims_softm = pos_scale_softm.keys()
        softm_alpha = softm_alphas[softm_alpha_idx]
        getcolor=plt.get_cmap(cmap_softm)
        colors=getcolor(np.linspace(cmap_lims[0], cmap_lims[1],len(nsims_softm)))
        c=0
        for k in nsims_softm:
            print(k)
            g=make_histo_cumsum(pos_scale_softm[k][..., softm_alpha_idx],)
            ax.plot(*g,"--", label="sm $a = {:.2f}$ - {} sims -- auc {:4.3f}".format(softm_alpha,pretty_print_n(k),
                                                                            auc(*g)),
                    color=colors[c] )
            c+=1

    #plt.legend(loc="lower right")
    if labels:
        plt.ylabel("Fraction of sources found (avg)", fontsize="large")
        plt.xlabel("Fraction of infected nodes considered",fontsize="large")
    if set_lims:
        ax.set_xlim((-0.01, 1.01))
        ax.set_ylim((-0.01, 1.01))

    return ax
