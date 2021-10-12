import argparse
from pathlib import Path
import warnings

import numpy as np

from epigen import epidemy_gen_new, EpInstance
import epigen.generators as generat
from epigen import gen_observ


def create_parser():
    # read arguments
    parser = argparse.ArgumentParser(description="Run a simulation and don't ask.")

    # generic options
    parser.add_argument('-N', type=int, default=100, dest="N", help='network size')
    parser.add_argument('-d', type=int, default=3, dest="d", help='network degree')
    parser.add_argument('-height', type=int, default=3, dest="h", help='network height (for TREE only)')
    parser.add_argument('-scale', type=float, default=1, dest="scale", help='network scale (for proximity only)')
    parser.add_argument('-p_edge', type=float, default=1, dest="p_edge", help='probability of temporal contacts')
    parser.add_argument('-T', type=int, default=10, dest="t_limit", help='total time')
    parser.add_argument('-s', '--seed', type=int, default=1, dest="seed", help='rng seed')
    parser.add_argument('--type_graph', type=str, default="TREE", dest="type_graph", help="type_graph")
    parser.add_argument('--small_lambda_limit', type=float, default=900, dest="small_lambda_limit", help='small time cut for real contacts data')
    parser.add_argument('--gamma', type=float, default=5e-4, dest="gamma", help="gamma (rate of infection for real data)")
    parser.add_argument('--path_contacts', type=str, default="i_bird_contacts.npz", dest="path_contacts", help="path real data contacts")
    parser.add_argument('--dir', type=str, default="results/", dest="output_dir", help='output directory')
    parser.add_argument('--lambda', type=float, default=0.5, dest="lambda_", help="lambda")
    parser.add_argument('--mu', type=float, default=1e-10, dest="mu", help="mu")
    parser.add_argument('--device', type=str, default="cpu", dest="device", help="device")
    parser.add_argument('--init_name_file', type=str, default="", dest="str_name_file", help="str_name_file")
    parser.add_argument('--path_dir', type=str, default="not_setted", dest="path_dir", help="path_dir")
    parser.add_argument('--num_conf', type=int, default=10, dest="num_conf", help="num_conf with observations")
    parser.add_argument("--n_sources", type=int, default=1, help="Number of sources (seeds) for the epidemic cascades")
    parser.add_argument('--start_conf', type=int, default=0, dest="start_conf", help="starting number of the range of configurations to be computed")
    parser.add_argument('--lambda_init_param', type=float, default=0.1, dest="lambda_init_param", help="lambda starting value of the learning")
    parser.add_argument('--mu_init_param', type=float, default=0.1, dest="mu_init_param", help="mu starting value of the learning")
    parser.add_argument("--ninf_min", type=float, default=1, dest="ninf_min", help="""minimum number of infected in the generated epidemies. \
        If < 1 but > 0, it is considered the fraction of the population""")
    parser.add_argument("--ninf_max", type=float, default=None, dest="ninf_max", help="""maximum number of infected in the generated epidemies. \
        If < 1 it is considered the fraction of the population""")
    parser.add_argument("--unique_numinf",  action="store_true", help="Make epidemies with unique number of final infected and recovered")
    parser.add_argument("--no_ver_gen", action="store_false", dest="verbose_gen", 
            help="Don't be too verbose in the generation of epidemies")
    ###Observations args
    parser.add_argument("--sparse_obs", action="store_true", 
            help="Generate and run with sparse observations")
    parser.add_argument("--sparse_rnd_tests", type=int, default=-1, dest="sparse_n_test_time", 
            help="Number of random daily tests for each time")

    parser.add_argument("--sparse_obs_last", action="store_true",
            help="Only observe the infected, at the last time instant. No random observations are done")
    parser.add_argument("--sp_obs_min_tinf", type=int, default=-2)
    
    parser.add_argument("--pr_sympt", type=float, default=0., dest="sp_p_inf_symptoms", 
        help="probability for each infected individual to be symptomatic and get tested")
    parser.add_argument("--delay_test_p", type=float, nargs="*", dest="sp_p_test_delay", 
        help="List of probabilities for the time delay in testing symptomatic individuals, starting from 0. Enter one probability after the other, they will get normalized")
    parser.add_argument("--save_data_confs", action="store_true", 
            help="Save data of generated epidemies")
    


    return parser
    
def create_data_(args, give_instance=False, use_inst_name=False):
    seed = args.seed
    type_graph = args.type_graph
    N = args.N
    d = args.d
    h = args.h
    scale = args.scale
    t_limit = args.t_limit # Numbers of epoch of our epidemics spreading [0,1,...,T_limit-1]
    lambda_ = args.lambda_ # probability of infection
    mu = args.mu # probability of recovery
    if mu < 0:
        raise ValueError("Negative value of mu")
    if lambda_ < 0:
        raise ValueError("Negative value of lambda")
    p_edge = args.p_edge
    num_conf = args.num_conf
    data_gen = vars(args)
    data_gen.update({
        "start_time":0,
        "shift_t":True,
    })
    if "gamma1" and "gamma2" and "fraction_nodes1" in args:
        data_gen.update({
            "gamma1":args.gamma1,
            "gamma2":args.gamma2,
            "fraction_nodes1":args.fraction_nodes1
        })

    path_dir = args.path_dir
    if args.path_dir == "not_setted":
        path_dir = type_graph

    data_ = epidemy_gen_new(
                    type_graph = type_graph,
                    t_limit=t_limit,
                    mu=mu, 
                    lim_infected=args.ninf_min,
                    max_infected=args.ninf_max,
                    seed=seed,
                    num_conf=num_conf,
                    num_sources=args.n_sources,
                    data_gen=data_gen,
                    unique_ninf=args.unique_numinf,
                    verbose=args.verbose_gen
                       )

    
    if data_ == None:
        if give_instance:
            return data_, "", None
        else:
            return data_, ""
    # Generate observations
    if args.sparse_obs:
        ##check args
        print("Generating sparse observations...")
        ntests = args.sparse_n_test_time
        pr_sympt = args.sp_p_inf_symptoms
        p_test_delay = args.sp_p_test_delay
        if (p_test_delay is None and ntests < 0):
            raise ValueError("In order to run sparse observations you have to put the number of tests and the test delay")
        if p_test_delay is None:
            p_test_delay = np.array([1.])
        else:
            p_test_delay = np.array(p_test_delay)/sum(p_test_delay)
        ## get full epidemies
        if args.sparse_obs_last:
            obs_df, obs_json = gen_observ.make_sparse_obs_last_t(data_,
                t_limit, pr_sympt=pr_sympt, seed=seed, verbose=args.verbose_gen,
                )
        else:
            obs_df, obs_json = gen_observ.make_sparse_obs_default(data_,
                    t_limit, ntests=ntests, pr_sympt=pr_sympt,
                    p_test_delay=p_test_delay, seed=seed, verbose=args.verbose_gen,
                    min_t_inf=args.sp_obs_min_tinf)
        for df in obs_df:
            df["obs_st"] = tuple(gen_observ.convert_obs_list_numeric(df["obs"]))
        data_["observ_df"] = obs_df
        data_["observ_dict"] = obs_json
        #print(obs_df[0])
        print("DONE.")


    path_save =Path(path_dir)
    if not path_save.exists():
        warnings.warn("SAVING FOLDER DOES NOT EXIST")
        
    contacts = data_["contacts"]
    N = int(max(contacts[:, 1]) + 1)

    inst = EpInstance(type_graph=type_graph, n=N, d=d,
        t_limit=t_limit, lamda=lambda_, mu=mu, seed=seed, p_edge=p_edge,
        n_source=args.n_sources)
    
    ## ************ CREATE NAME FILE  ************

    name_file = path_dir + "/" + args.str_name_file
    if use_inst_name:
        name_file += str(inst)
    else:
        name_file += f"N_{N}_d_{d}_h_{h}_T_{t_limit}_lam_{lambda_}_mu_{mu}_p_edge_{p_edge}"
        name_file += f"_s_{seed}"
    #name_file += f"_h_source_{h_source}_h_log_p_{h_log_p}_h_obs_{h_obs}"
    #print(name_file)
    if give_instance:
        return data_, name_file, inst
    else:
        return data_, name_file