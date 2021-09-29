#!/usr/bin/env python
# coding: utf-8

import sys, os
import time
from pathlib import Path
import json
import numpy as np
import torch


import epigen.generators as generat
from epigen.epidemy_gen import make_stats_observ

from annfore.net import nn_sir_path_obs
from annfore.utils.graph import find_neighs
from annfore.models import common as en_common
from annfore.models import sir_model_N_obs


from annfore.learn.losses import loss_fn_coeff, loss_fn_coeff_p_sus
from annfore.learn.train import train_beta, train_saving_marginals, make_training_step_local
from annfore.learn.opt import make_opt

from annfore.learn.train_params import opt_param_init, train_beta_params
from annfore.learn.train_params import learn_lamb_mu, learn_gamma_mu
from annfore.learn.l_utils import make_beta_sequence_three as make_beta_seq

path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent / "src"))
from utils.script_utils import create_parser, create_data_
from io_m.io_utils import save_json

N_LAYS_DEFAULT = 3

def add_arg_parser(parser):
    # neural network options
    parser.add_argument('--p_source', type=float, default=-1, dest="p_source", help="p_source")
    parser.add_argument("--p_rec_t0",type=float, default=-1, help="Prob of node = R at time 0")
    parser.add_argument('--p_mc', type=float, default=-1, dest="p_mc", help="p_mc")
    parser.add_argument('--p_obs', type=float, default=-1, dest="p_obs", help="p_obs")
    parser.add_argument("--no_bias", action="store_true", help="Don't use bias in the networks")

    parser.add_argument('--num_samples', type=int, default=10000, dest="num_samples", help="num_samples of the learning process")
    parser.add_argument('--lr', type=float, default=1e-3, dest="lr", help="learning rate")
    parser.add_argument('--t_obs', type=int, default=0, dest="t_obs", help="time to compute marginals")
    parser.add_argument('--step', type=float, default=1e-4, dest="step", help="step value of the beta annealing between (0,1)")
    parser.add_argument("--n_beta_steps", type=int, default=-3,
                    help="Set the number of total steps for beta annealing (instead of the step size)")
    parser.add_argument('--iter_marginals', type=int, default=100, dest="iter_marginals", help="number of steps for computing marginals")
    parser.add_argument('--continuous', action="store_true", help="use continuos model")
    parser.add_argument("--n_hidden_layers", type=int, default=-1, help="Number of hidden layers to use, default 3")
    parser.add_argument('--lay_deeper', action="store_true", help="use continuos model")
    parser.add_argument('--lay_less_deep', action="store_true",
        help="Hidden layers of 1/2, 1/2, 1")
    parser.add_argument("--lay_std_deep", action="store_true",
        help="Fixed hidden layers of 3/2, 1")
    parser.add_argument('--lay_deep_eq', action="store_true", help="Hidden layers of 1, 1")
    parser.add_argument('--lay_deep_sc', action="store_true", help="Hidden layers of 1 ,1/2, 1")
    parser.add_argument('--opt', type=str, default="adam", dest="opt_algo", help="name of the optimization algorithm")
    parser.add_argument('--num_threads', type=int, default=None, dest="num_threads", help="set num_threads")
    parser.add_argument('--num_end_iter', type=int, default=100, dest="num_end_iter", help="number of itareations at the end of annealing process ")
    parser.add_argument('--lr_param',  action="store_true", help="learning params of the propagation model [lambda, mu]")
    parser.add_argument('--lr_gamma',  action="store_true", help="learning rate of infection instead of probability of the propagation model [gamma]")
    parser.add_argument('--save_net', action="store_true", help="save neural net")
    parser.add_argument('--only_neigh', action="store_true", help="consider only nearest neighs")
    parser.add_argument('--MF', action="store_true", help="consider no neighs [MeanField version]")
    parser.add_argument("--all_graph_neighs", action="store_true",
                    help="Set all individuals as neighbors, no k-th neighbor approximation")

    parser.add_argument("--no_net_obs", action="store_true", help="do not pass observations to NN")
    parser.add_argument("--p_fin_bal", action="store_true", help="Use the prior for balanced final state, like psus in training")
    #parser.add_argument("--p_sus_nodes", action="store_true", help="Calculate psus separately for the nodes")
    parser.add_argument("--psus_beta_exp", type=float, default=1., help="Exponent for beta descent of psus")
    #parser.add_argument("--p_sus_forced",action="store_true", help="Force to use psus regardless of balanced final state prior")
    parser.add_argument("--p_sus", type=float, default=-1, help="Parameter p_sus for the model")
    parser.add_argument("--debug")
    parser.add_argument("--init", type=str, default="xavier", help="Initialization method for the weights")
    parser.add_argument("--lin_net_pow", type=float, default=2., help="Power to which scale the linear layers")


    parser.add_argument("--exp_like_beta",action="store_true", help="beta annealing with more exponential like sequence of beta")
    return parser



if __name__ == "__main__":
    parser = create_parser()
    parser = add_arg_parser(parser)
    args = parser.parse_args()
    print("arguments:")
    print(args)

## ************* set algorithm specific parameters *********** 
    device = args.device
    dtype = torch.float
    p_source = args.p_source
    p_mc = args.p_mc
    p_obs = args.p_obs
    #p_sus_sparse_obs = 1.-1/((args.t_limit+1)*(args.t_limit+1))
    USE_PSUS_FINALT = args.sparse_obs
    USE_LOSS_PSUS_BETA = args.sparse_obs

    #P_SUS_FIXED = args.p_sus_fixed if args.p_sus_fixed > 0 else None

    #FORCE_PSUS = args.p_sus_forced

    #if (args.p_sus_fixed > 0) and not args.p_fin_bal:
    #    USE_LOSS_PSUS_BETA = False
    if (args.init =="uniform" or args.init=="xavier"):
        INIT_METHOD = args.init
    else:
        raise ValueError(f"Init method for the weights {args.init} is invalid")
    
    NEXT_NEAR_NEIGH = True if not args.only_neigh else False
    
    func_layers =["none"]*args.n_hidden_layers
    if (args.n_hidden_layers != -1) and (
        args.lay_deeper or args.lay_std_deep or args.lay_less_deep or args.lay_deep_eq or args.lay_deep_sc):
        raise ValueError("Use either '--n_hidden_layers' or a fixed layer flag")
    
    if (args.n_hidden_layers == -1):
        args.n_hidden_layers=N_LAYS_DEFAULT

    if args.lay_deeper:
        func_layers = [3/2, 3/2, 1]
    elif args.lay_deep_eq:
        func_layers = [1, 1]
    elif args.lay_deep_sc:
        func_layers = [2/3, 1/3, 2/3, 1/3]
    elif args.lay_less_deep:
        func_layers = [1/2, 1/2, 1]
        #raise NotImplementedError
    elif args.lay_std_deep:
        func_layers = [3/2,1]
    else:
        func_layers = [-2]*args.n_hidden_layers

    print("Func layers:", func_layers)

    opt_algo = args.opt_algo

    if args.num_threads != None:
        torch.set_num_threads(args.num_threads)

    esp_p =2
    c = 1/(1-np.exp(-esp_p))
    #beta_inv_ann_coeff = lambda b: c*np.exp(-esp_p*b)+1-c
    beta_inv_ann_coeff = lambda b: (1.-b)**args.psus_beta_exp
    #beta_inv_ann_coeff = None
    #psus_mult=args.p_sus_mult

## ************ GENERATE EPIDEMIES ************

    data_, name_file, INSTANCE = create_data_(args,give_instance=True,
                                    use_inst_name=True)
    if data_ == None:
        quit()
    if args.sparse_obs:
        ## make statistics on the observations
        df_all_obs = make_stats_observ(np.array(data_["test"]),  data_["observ_df"])

        print(df_all_obs[args.start_conf:])

## ************ SET data-dependent ALGORITHM PARAMETER ************

    #+1 t_limit times || +1 obs after contacts || +1 for susceptible
    contacts = data_["contacts"]
    N = int(max(contacts[:, 1]) + 1)
    t_limit = args.t_limit
    mu = args.mu
    print("PSUS: ", args.p_sus)

    BIAS_NETS = not args.no_bias
    print("BIAS: ", BIAS_NETS)

    
    if p_source < 0:
        p_source = 1/N
    if p_mc < 0:
        p_mc = np.exp(np.log(p_source) - args.t_limit)
    if p_obs < 0:
        p_obs = p_mc
    
    if args.p_rec_t0 < 0:
        p_rec_t0 = p_source
    else:
        p_rec_t0 = args.p_rec_t0

    nfeat = int(max(contacts[:, 0]) + 3)
    if args.all_graph_neighs and args.MF:
        raise ValueError("Cannot put `all_graph_neighs` and `MF` together!")

    if args.all_graph_neighs:
        neighs = [range(0,k) if k != 0 else [] for k in range(N)]
    elif not args.MF:
        neighs = find_neighs(contacts,N=N,only_minor=True, next_near_neigh=NEXT_NEAR_NEIGH)
    else:
        neighs = [[] for n in range(N)]

    #print(neighs)

## ************ RUN INFERENCE ALGORITHMS ************

    print("Begin training")

    for instance_num in range(args.start_conf, args.num_conf):
        print("Instance", instance_num)
        last_obs = data_["test"][instance_num][1]
        init_conf = data_["test"][instance_num][0]
        
        print("Real source:",np.where(init_conf)[0])
        name_file_instance = name_file + "_" + str(instance_num)
        if not args.sparse_obs:
            obs_list = []
            obs_list_sib=[]
            for i, s in enumerate(last_obs):
                obs_list.append([i,t_limit,s])
                obs_list_sib.append([i,s,t_limit])
        else:
            obs_df = data_["observ_df"][instance_num]
            obs_list = obs_df[["node","time","obs_st"]].to_records(index=False)
            obs_list_sib = obs_df[["node","obs_st","time"]].to_records(index=False)
            #print(obs_list)
            #print(json.dumps(data_["observ_dict"][instance_num],), )
            obs_df.to_csv(name_file_instance+"_obs_sparse.csv",index=False)

        
        all_script_args = vars(args)    
        
        all_script_args["extra"]={
            "func_layers": func_layers
        }
        extra_saving_args = all_script_args["extra"]
        print("Observations: ", obs_list_sib)
        
        if args.no_net_obs:
            obs_list_net = []
        elif not args.lr_param and mu==0:
            obs_list_net = list(obs_list_sib)
            for i in range(INSTANCE.n):
                obs_list_net.append((i,1,t_limit+10))
            #print(obs_list_net)
        else:
            obs_list_net = obs_list_sib

        my_net = nn_sir_path_obs.SIRPathColdObs(neighs,
                    t_limit+1,
                    obs_list=obs_list_net,
                    hidden_layer_spec=func_layers,
                    dtype=dtype,
                    device = device,
                    in_func=torch.nn.LeakyReLU(),
                    bias=BIAS_NETS,
                    lin_scale_power=args.lin_net_pow
                    )

        my_net.init(method=INIT_METHOD)
        #in_func=torch.nn.LeakyReLU()
        masks = my_net.masks.numpy()
        for n in my_net.dimensions():
            print(n)
        ## check if we have a fixed psus
        """if args.p_sus_fixed > 0:
            p_sus_sparse_obs = args.p_sus_fixed
            p_sus_max = args.p_sus_fixed
            print("Fixed p sus for the whole training")
        else:
            p_sus_arr = en_common.calc_psus_masks_nodes(masks, my_net.N, t_limit)
            p_sus_max = p_sus_arr.mean()
            p_sus_sparse_obs = (p_sus_max - .5) * psus_mult + 0.5
            extra_saving_args["p_sus_max"] = p_sus_max
            extra_saving_args["p_sus_final"] = p_sus_sparse_obs
        """

        extra_saving_args["num_parameters"] = my_net.nparams
        print("Num parameters: ", my_net.nparams)
        
        
        if args.p_fin_bal:
            print("Using Sir Model bal")
            ModelClass = sir_model_N_obs.SirModel_bal
            #if FORCE_PSUS:
            #    print("P sus max:", p_sus_max, "p sus final:", p_sus_sparse_obs)
        else:
            ModelClass = sir_model_N_obs.SirModel
            #print("P sus max:", p_sus_max, "p sus final:", p_sus_sparse_obs)
        """
        elif args.p_sus_nodes:
            p_sus_sparse_obs = torch.tensor(
                en_common.calc_psus_masks_nodes(masks, my_net.N, t_limit),
                device=device,
                dtype=dtype,
            )
            print("PSUS ALL: ",p_sus_sparse_obs)
            modelclass = sir_model_N.SirModel_susSep
        """
        
            
        
        #p_sus = (0.5,0) if (not USE_PSUS_FINALT) or args.p_fin_bal else (p_sus_sparse_obs, args.t_limit)
        model = ModelClass(contacts,
                                mu = mu,
                               device = device,
                                  p_source=p_source,
                                  p_obs=p_obs,
                                  p_sus=args.p_sus,
                                  p_w=p_mc)
        model.set_obs(obs_list)
        if args.p_fin_bal:
            model.set_masks(my_net.masks.numpy(),logmult=1.)
            """
            if FORCE_PSUS:
                ## we want to use psus too
                print("Forcing psus with p_fin_bal, p_sus_fixed: ", P_SUS_FIXED)
                model.set_forced_psus(multipl=psus_mult, psus_val=P_SUS_FIXED, use_psus_energy=(args.p_sus_fixed > 0))
            """
        num_samples = args.num_samples
        lr=args.lr
        optimizer = []
        for i in range(N):
            if len(my_net.params_i[i]):
                optimizer.append(make_opt(my_net.params_i[i], lr=lr))
            else:
                optimizer.append([])
        t_obs = args.t_obs
        step = args.step
        if args.n_beta_steps > 0:
            if args.exp_like_beta:
                betas = make_beta_seq(args.n_beta_steps, mid=(0.25,0.5),highmid=(0.6,0.85))
            else:
                betas = np.linspace(0,1,args.n_beta_steps+1)[:-1]
        else:
            betas = np.arange(0.,1,step)

        JSON_STAT_FILE=name_file_instance+"_args.json"
        #with open(JSON_STAT_FILE,"w") as f:
        #    json.dump(all_script_args,f, indent=1)

        all_script_args["timing"] = {}
        all_script_args["timing"]["start"] = int(time.time())

        save_json(JSON_STAT_FILE, all_script_args, indent=1)
        
        if not args.lr_param:
            if USE_LOSS_PSUS_BETA:
                loss_fun = loss_fn_coeff_p_sus
                #betas = np.concatenate((np.zeros(50),betas))
                print("USE PSUS LOSS")
                extra_args = {"fun_beta_inv":beta_inv_ann_coeff}
            else:
                loss_fun = loss_fn_coeff
                extra_args = None
            
            results = train_beta(my_net, optimizer,
                        model, name_file_instance,
                        loss_fun, t_obs,
                        num_samples=num_samples,
                        train_step = make_training_step_local,
                        betas=betas, save_every=200,
                        loss_extra_args=extra_args)
        else:
            betas = np.append(betas, [1]*args.num_end_iter)
            
            mu_param, mu_opt = opt_param_init(model, 
                                              param_init=args.mu_init_param, 
                                              name="mu", dtype=dtype, device=device, lr=lr)
            if args.lr_gamma:
                model.create_deltas_tensor(data_["deltas"])
                lamb_param, lamb_opt = opt_param_init(model, 
                                                      param_init=args.lambda_init_param, 
                                                      name="lamb", dtype=dtype, device=device, lr=lr/10)
                learn_params_fnc=learn_gamma_mu
            else:
                contacts[:,3]=1-1e-6
                model.create_lambdas_tensor(contacts)
                lamb_param, lamb_opt = opt_param_init(model, 
                                                  param_init=args.lambda_init_param, 
                                                  name="lamb", dtype=dtype, device=device)
                learn_params_fnc=learn_lamb_mu
            
            params=[lamb_param, mu_param]
            opt_params=[lamb_opt, mu_opt]

            model.extra_params["max_Z"]=-100000000
            
            loss_fun = loss_fn_coeff
            results = train_beta_params(my_net, optimizer, 
                        model, name_file_instance, 
                        loss_fun, t_obs, 
                        params, opt_params,
                        num_samples=num_samples,
                        train_step = make_training_step_local,
                        learn_params=learn_params_fnc,
                        betas=betas, save_every=200)

        print("computing marginals")
        iter_marginals = args.iter_marginals

        train_saving_marginals(my_net, optimizer,
                    model, name_file_instance,
                        loss_fun, t_obs,
                    results=results,
                    num_samples=num_samples,
                    train_step = make_training_step_local,
                    num_iter=iter_marginals)
        if args.save_net:
            torch.save(my_net, name_file_instance + "_full_net.pt" )

        all_script_args["timing"]["end"] = int(time.time())


        save_json(JSON_STAT_FILE, all_script_args, indent=1)