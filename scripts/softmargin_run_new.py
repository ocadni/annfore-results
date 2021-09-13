#!/usr/bin/env python
# coding: utf-8
"""
Old script to launch the softmargin.
"""

import sys, os
import argparse
import warnings
from pathlib import Path
import numpy as np
import numba as nb
##FIND CORRECT DIR
path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent))


import src.soft_margin.soft_margin as soft_margin
from src.utils.script_utils import create_parser, create_data_
from epigen import propagate

def add_arg_parser(parser):

    # soft margin options
    parser.add_argument('--a_min', type=float, default=0.1, dest="a_min",
        help="Minimum value of parameter a")
    parser.add_argument('--a_step', type=float, default=0.01, dest="a_step",
        help="Stepping value of parameter a")
    parser.add_argument('--a_max', type=float, default=0.4, dest="a_max",
        help="Maximum value of parameter a")

    parser.add_argument("-sms","--softm_seed", type=int, default=None,
        dest="softm_seed")

    parser.add_argument('--niter', type=int, default=1, dest="n_iter",
        help="Number of iterations of softmargin algorithm")
    parser.add_argument("--nsims", type=int, default=2000, dest="n_sims",
            help="Number of simulations to do at each iteration of softmargin")

    parser.add_argument("--ncores", type=int, default=-1, dest="n_cores",
        help="Number of cores to run the simulations with")

    parser.add_argument("--nsims_margs", type=int, default=None, dest="n_sims_margs",
        help="Number of simulations to do at each iteration for the marginals")
        
    parser.add_argument("--niter_margs", type=int, default=1, dest="n_iter_margs",
        help="Number of iterations for the computation of marginals")
    parser.add_argument("--nsims_steps", type=int, default=1, dest="n_sims_steps",
        help="Number of simulations steps")
    
    parser.add_argument("-mg","--marginals", action="store_true",
        help="Calculate marginals together with the source probability")
    parser.add_argument("-nr","--nrepeat", type=int, default=1,
        dest="nrepeat", help="Number of times to repeat the same inst")
    parser.add_argument("--overwrite",  action="store_true", help="overwrite save data")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    parser = add_arg_parser(parser)
    args = parser.parse_args()
    print("arguments:")
    print(args)
    
        

    ## ************ GENERATE EPIDEMIES ************
    h = args.h
    data_, name_file, INSTANCE = create_data_(args, give_instance=True)
    if data_ == None:
        sys.exit(-2)

    if h < 3:
        x = np.array(data_["test"])
        print(x)
    ## ************* set algorithm specific parameters *********** 

    num_conf = args.num_conf
    type_graph = args.type_graph
    path_dir = args.path_dir
    if args.path_dir == "not_setted":
        path_dir = type_graph
    path_dir = Path(path_dir)
    name_file = args.str_name_file + str(INSTANCE)
    name_file +=f"_nc_{num_conf}"


    ## ************ RUN INFERENCE ALGORITHMS ************
    if args.softm_seed is not None:
        seed_r = args.softm_seed
        propagate.set_seed_numba(seed_r)
        np.random.seed(seed_r)
    n_sims_margs = args.n_sims_margs
    n_sims = args.n_sims
    if args.marginals:
        GET_MARGS = True
        print("Running with marginals")
    else:
        GET_MARGS = False

    if args.n_sims_margs is not None and n_sims_margs != n_sims:
        RUN_MARGS_W_SRC = False
    else:
        RUN_MARGS_W_SRC = True

    contacts = data_["contacts"]
    if args.n_cores > 0:
        n_cores = min(args.n_cores, os.cpu_count())
        nb.set_num_threads(n_cores)

    softmRunner = soft_margin.SoftMarginDirectRunner(
        INSTANCE, data_,
        contacts,
        a_par_max=args.a_max,
        a_par_min=args.a_min,
        a_par_step=args.a_step,
        num_iter=args.n_iter,
        n_sims_per_a=args.n_sims
    )

    softmRunner.run_soft_margin_all(calc_margs=(RUN_MARGS_W_SRC and GET_MARGS))
    if GET_MARGS and not RUN_MARGS_W_SRC:
        softmRunner.calc_all_margs(args.n_sims_margs, args.n_iter_margs)

    name_file+=f"_softm_nsims_{args.n_sims}_nit_{args.n_iter}"

    if GET_MARGS and not RUN_MARGS_W_SRC:
        name_file+=f"_mgns_{args.n_sims_margs}"
        name_file+=f"_mgnit_{args.n_iter_margs}"

    name_file_softm = (path_dir / name_file).as_posix()

    softmRunner.save_data(name_file_softm, overwrite=args.overwrite)
