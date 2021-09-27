from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_json

#create dictorary containing data
def get_def_res(Ns, num_confs, seeds):
    ress = {}
    for i_seed, seed in enumerate(seeds):
        print(f"\nSEED: {seed}")
        ress[seed] = {}
        for cl in range(len(Ns)):
            N_cl = Ns[cl]
            ress[seed][Ns[cl]] = []
            for instance_num in range(num_confs[i_seed][N_cl]):
                print(f" N: {N_cl} instance {instance_num}", end="\r")
                ress[seed][Ns[cl]].append({})
    return ress

def get_other_inst_str(inst, height, pedge_float=False):
    name=f"N_{inst.n}_d_{inst.d}_h_{height}_T_{inst.t_limit}"
    name+=f"_lam_{inst.lambda_}_mu_{inst.mu}"
    if pedge_float:
        name+=f"_p_edge_{inst.p_edge:2.1f}_s_{inst.seed}"
    else:
        name+=f"_p_edge_{inst.p_edge}_s_{inst.seed}"
    return name
    
def _make_range_confs(in_range):
    if isinstance(in_range, range):
        return in_range
    
    if len(in_range) < 2:
        raise ValueError("Input start and finish")
    elif len(in_range) == 2:
        load_range = range(*in_range)
    else:
        load_range = in_range
    return load_range

def read_sib_data_def(fold,inst,h,prefix="", range_confs=(0,1), outprint=True):
    #name=f"{prefix}N_{inst.n}_d_{inst.d}_h_{h}_T_{inst.t_limit}"
    #name+=f"_lam_{inst.lambda_}_mu_{inst.mu}_p_edge_{inst.p_edge}_s_{inst.seed}"
    name = f"{prefix}"+ get_other_inst_str(inst,h)
    margs = []
    path = Path(fold)
    load_range = _make_range_confs(range_confs)
    if outprint:
        print(path.resolve().as_posix())
    for i in load_range:
        nam_f = name + f"_{i}_sib_margs.npz"
        d = np.load(path / nam_f)
        margs.append(d["marginals"])
        d.close()
    return margs
    
def read_margs_default(fold, inst, h, algo=None, prefix="", range_confs=(0,1), post_algo="", outprint=True, pedge_float=True):
    """
    Read marginals with default prefix system
    """
    name = f"{prefix}"+ get_other_inst_str(inst,h)
    if algo == "sib":
        post_algo = "sib_margs"
    elif algo == "ann":
        post_algo = "margs"
        name = f"{prefix}"+ get_other_inst_str(inst,h, pedge_float=pedge_float)
    elif len(post_algo) <= 0:
        raise ValueError("Cannot determine the algorithm")
    
    load_range = _make_range_confs(range_confs)
    margs = []
    path = Path(fold)
    if outprint:
        print(path.resolve().as_posix())
        print(load_range)
    for i in load_range:
        nam_f = name + f"_{i}_{post_algo}.npz"
        d = np.load(path / nam_f)
        margs.append(d["marginals"])
        d.close()
    return margs

def read_margs_inst(fold, inst, prefix="", name_margs="margs", range_confs=(0,1),algo=None, outprint=True):
    """
    Read marginals with instance naming
    """
    name = f"{prefix}"+ str(inst)
    if algo == "sib":
        name_margs = "sib_margs"
    elif algo == "ann":
        name_margs = "margs"
    
    margs = []
    path = Path(fold)
    if outprint:
        print(path.resolve().as_posix())

    load_range = _make_range_confs(range_confs)

    if outprint: print(load_range)
    for i in load_range:
        nam_f = name + f"_{i}_{name_margs}.npz"
        d = np.load(path / nam_f)
        margs.append(d["marginals"])
        d.close()
    return margs

def read_params_inst(fold, inst, prefix="", name_pars="args", range_confs=(0,1),algo=None, outprint=True):
    """
    Read marginals with instance naming
    """
    name = f"{prefix}"+ str(inst)
    
    margs = []
    path = Path(fold)
    if outprint:
        print(path.resolve().as_posix())

    load_range = _make_range_confs(range_confs)

    if outprint: print(load_range)
    for i in load_range:
        nam_f = name + f"_{i}_{name_pars}.json"
        d = load_json(path / nam_f)
        margs.append(d)
    return margs


def read_risk_inst(fold, inst, ranker, prefix="", range_confs=(0,1), outprint=True):
    """
    Read marginals with instance naming
    """
    name = f"{prefix}"+ str(inst)+f"_rk_{ranker}"
    
    ranking = []
    path = Path(fold)
    if outprint:
        print(path.resolve().as_posix())
    
    load_range = _make_range_confs(range_confs)
    if outprint: print(load_range)

    for i in load_range:
        nam_f = name + f"_{i}_rank.npz"
        d = np.load(path / nam_f)
        rep = d["ranking"]
        ser = pd.Series(index=rep["idx"], data=rep["risk"])
        ranking.append(ser)
        d.close()
    return ranking
