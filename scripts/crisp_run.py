#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent/"src"))


from utils.script_utils import create_parser, create_data_
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
    parser.add_argument('--n_step', type=int, default=5000, help="number of MC steps")
    parser.add_argument("--seed_mc", type=int, default=None)
    parser.add_argument('--p_wrong_obs', type=float, default=1e-6, help="prob of wrong observations")
    
    parser.add_argument('--n_burnin', type=int, default=100, help="number of MC to ignore (burn in)")

    return parser

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

    
## ************ GENERATE EPIDEMIES ************

    data_, name_file, INSTANCE = create_data_(args, give_instance=True,
        use_inst_name=True)
    if data_ == None:
        quit()
    
    confs = np.array(data_["test"])
    print("We have {} epidemies".format(len(confs)))
    

## ************ RUN INFERENCE ALGORITHMS ************

    
        
    t_limit = INSTANCE.t_limit
    N = INSTANCE.n
    T = t_limit + 1

    contacts = data_["contacts"]

    mat_obs = crisp_sir.make_mat_obs(args.p_wrong_obs)


    #crisp_sir.set_numba_seed(seed)
    #np.random.seed(seed)
    
    params_crisp = crisp_sir.make_params(N, T, pautoinf=args.p_autoinf, 
            p_source=args.p_source, lamda=INSTANCE.lambda_, mu=INSTANCE.mu, p_sus=args.p_sus)
    
    for instance_num in range(args.start_conf, args.num_conf):
        last_obs = data_["test"][instance_num][1]
        real_src = data_["test"][instance_num][0]
        print("Real source:",np.where(real_src)[0])
        name_file_instance = name_file + "_" + str(instance_num)

        if not args.sparse_obs:
            obs_list = []
            
            for i, s in enumerate(last_obs):
                obs_list.append([i,s,t_limit])
            
        else:
            obs_df = data_["observ_df"][instance_num]
            
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
        if args.seed_mc is not None:
            seed_mc = args.seed_mc
            print("Setting seed ", seed_mc)
        else:
            seed_mc = None
    
        ecc1, stats, ecc = crisp_sir.run_crisp(params_crisp, observ=obs_list,
            contacts=contacts,
            num_samples=args.n_step,
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
