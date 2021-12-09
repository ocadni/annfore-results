#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import time
import warnings
from multiprocessing.pool import Pool as MultiProcPool
from pathlib import Path

import numpy as np
import pandas as pd

path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent/"src"))


from utils.script_utils import create_parser, create_data_, get_name_file_instance, get_base_name_file
try:
    import crisp_sir
except ImportError:
    print("crisp_sir is not installed. Install from ")
    sys.exit(1)



def add_arg_parser(parser):
    # sib options
    parser.add_argument('--p_source', type=float, default=-1, dest="p_source", help="p_source")
    parser.add_argument('--p_sus', type=float, default=0.5, dest="p_sus", help="p_sus")
    parser.add_argument('--p_autoinf', type=float, default=1e-7, help="prob of from exogenous sources")
    parser.add_argument('--n_steps', type=int, default=None, help="number of MC steps")
    parser.add_argument("--seed_mc", type=int, default=None)
    parser.add_argument('--p_wrong_obs', type=float, default=1e-6, help="prob of wrong observations")
    
    parser.add_argument('--n_burnin', type=int, default=100, help="number of MC to ignore (burn in)")

    parser.add_argument('--n_proc', type=int, default=None, help="number of processes")
    parser.add_argument("--rep_range", type=int, nargs=2, help="repetition range", default=None)
    parser.add_argument("--n_steps_list", type=int, nargs="*", help="list of number of MC steps to do (can be empty)")

    return parser

def make_run_args(data_, instance_num, base_name_file, args, instance,
        contacts, nsteps, seed):

    out_data = {}
    #print(data_.keys())
    out_data["test"] = data_["test"]
    if "observ_df" in data_:
        out_data["observ_df"] = data_["observ_df"]
    out_data["contacts"] = contacts
    out_data["instance_num"] = instance_num
    out_data["name_file"] = get_name_file_instance(args, base_name_file, instance)
    out_data["args"] = args
    out_data["epInstance"] = instance
    out_data["n_steps"] = nsteps
    out_data["seed_mc"] = seed

    return out_data



def run_crisp_save(data_in):
    inst_num = data_in["instance_num"]
    name_file = data_in["name_file"]
    args = data_in["args"]
    epInstance = data_in["epInstance"]
    contacts=data_in["contacts"]
    t_limit = epInstance.t_limit
    nsteps = data_in["n_steps"]
    seed_mc = data_in["seed_mc"]
    last_obs = data_in["test"][inst_num][1]
    real_src = data_in["test"][inst_num][0]
    print("Real source:",np.where(real_src)[0])
    name_file_instance = name_file + "_" + str(inst_num)
    mat_obs = crisp_sir.make_mat_obs(args.p_wrong_obs)

    pars_crisp = crisp_sir.make_params(N, T, pautoinf=args.p_autoinf, 
            p_source=args.p_source, lamda=epInstance.lambda_, mu=epInstance.mu, p_sus=args.p_sus)
    if not args.sparse_obs:
        obs_list = []
        
        for i, s in enumerate(last_obs):
            obs_list.append([i,s,t_limit])
        
    else:
        obs_df = data_in["observ_df"][inst_num]
        
        obs_list = obs_df[["node","obs_st","time",]].to_records(index=False)
        print(obs_list)
        obs_init_list =[(i,-1,t) for t in range(t_limit+1) for i in range(N) ]
        obs_init_list.extend(obs_list)
        #print(obs_list)
        #print(json.dumps(data_["observ_dict"][instance_num],), )
        obs_df.to_csv(name_file_instance+"_obs_sparse.csv",index=False)
        obs_list = obs_init_list
    
    obs_list.sort(key=lambda tup: tup[0])

    t_v = time.time()
    if seed_mc is not None:
        '''
        crisp_sir.set_numba_seed(seed_mc)
        from crisp_sir.crisp_sir import sample
        ptry = np.ones(10000)
        ptry /= ptry.sum()
        print("Seed: {}, Rand sample: {}    ".format(seed_mc, sample(ptry)))
        '''
        print("\n\tSetting seed ", seed_mc)

    ecc1, stats, ecc = crisp_sir.run_crisp(pars_crisp, observ=obs_list,
        contacts=contacts,
        num_samples=nsteps,
        mat_obs=mat_obs,
        burn_in=args.n_burnin,
        seed=seed_mc,
        start_inf=True,
        )
    taken_t = int(time.time() -  t_v)
    print("Max num source", stats[:,0,1].max())
    
    print("Done, saving marginals and script arguments...")

    all_args = vars(args)
    all_args["running_time"] = taken_t 
    
    with open(name_file_instance+"_args.json","w") as mfile:
        json.dump(all_args,mfile, indent=1)
    
    margs = stats / stats.sum(-1)[...,np.newaxis]
    init_c = margs[:,0].sum(0)
    
    print("Init state: "," ".join(f"{cc}:{vv:.3f}" for cc,vv in zip(["S","I","R"], init_c)))
    

    np.savez_compressed(name_file_instance+"_margs.npz", marginals=margs)

if __name__ == "__main__":
    parser = create_parser()
    parser = add_arg_parser(parser)
    args = parser.parse_args()
    print("arguments:")
    print(args)

## ************* set algorithm specific parameters *********** 
    mu = args.mu
    p_source = args.p_source
    p_sus = args.p_sus
    
    
        

    if args.n_steps is not None and args.n_steps_list is not None:
        raise ValueError("Cannot give both `n_steps` and `n_steps_list` arguments")

    if args.n_steps is not None:
        N_STEPS_MC = [args.n_steps]
    elif args.n_steps_list is not None:
        N_STEPS_MC = args.n_steps_list
    else:
        raise ValueError("Give either `n_steps` or `n_steps_list` argument")

    if args.rep_range is not None:
        REPS = range(*args.rep_range)
    else:
        REPS = None
    

    
## ************ GENERATE EPIDEMIES ************

    data_, _, INSTANCE = create_data_(args, give_instance=True,
        use_inst_name=True)
    if data_ == None:
        quit()
    
    confs = np.array(data_["test"])
    print("We have {} epidemies".format(len(confs)))
    
    if args.p_source <= 0:
        print("Setting p source to 1/N={:.4f}".format(1/INSTANCE.n))
        args.p_source = 1/INSTANCE.n
    

## ************ RUN INFERENCE ALGORITHMS ************

    
        
    t_limit = INSTANCE.t_limit
    N = INSTANCE.n
    T = t_limit + 1

    contacts = data_["contacts"]

    mat_obs = crisp_sir.make_mat_obs(args.p_wrong_obs)

    base_name_file = get_base_name_file(args)
    #crisp_sir.set_numba_seed(seed)
    #np.random.seed(seed)

    SEED_MC = args.seed_mc
    
    if REPS is None and len(N_STEPS_MC) <= 1:
        for instance_num in range(args.start_conf, args.num_conf):
        ### determine name files
        
            ## simple
            myargs = make_run_args(data_, instance_num, base_name_file, args,
            INSTANCE, contacts, nsteps=args.n_steps, seed=args.seed_mc)

            run_crisp_save(myargs)
    else:
        ALL_ARGS = []
        for instance_num in range(args.start_conf, args.num_conf):
            if args.n_proc is None:
                warnings.warn("Using the number of cores for the number of processes, set `n_proc` instead")
            ## do not have to add the repetitions
            
            m_args = lambda nst, nam_file, s : make_run_args(data_, instance_num, nam_file, args,
                    INSTANCE, contacts, nsteps=nst, seed=s
                                    )
            for n_s in N_STEPS_MC:
                #args_c = dict(def_args)
                m_name_file = base_name_file
                if len(N_STEPS_MC) > 1:
                    m_name_file += f"nsteps_{n_s}_"
                if REPS is None:
                    ALL_ARGS.append(m_args(n_s,m_name_file, SEED_MC))
                else:
                    for r in REPS:
                       
                        if SEED_MC is not None:
                            s = SEED_MC + r
                        mname = m_name_file+ f"rep_{r}_"
                        ALL_ARGS.append(m_args(n_s, mname, s))
            

        ## run with multiprocessing
        print("ARGS: ", len(ALL_ARGS))
        with MultiProcPool(processes=args.n_proc) as pool:
            #res = pool.imap(run_crisp_save, ALL_ARGS, chunksize=10)
            results = [pool.apply_async(run_crisp_save, args=(w,)) for w in ALL_ARGS]
            results = [p.get() for p in results]
