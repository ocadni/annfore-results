#!/usr/bin/env python
# coding: utf-8

import sys, os
import argparse
import warnings
import time
from pathlib import Path
import numpy as np
import numba as nb

from epigen import propagate
##FIND CORRECT DIR
path_script = Path(sys.argv[0]).parent.absolute()
sys.path.append(os.fspath(path_script.parent+"/src"))

import soft_margin.soft_margin as soft_margin
import io_m.libsaving as libsaving
from io_m import io_utils
from io_m import mlogging
from utils.script_utils import create_parser, create_data_
from utils.common import pretty_print_n

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
    parser.add_argument("--nsims_start", type=int, default=-3, dest="n_sims_start",
            help="Number of simulations to do at each iteration of softmargin")
    parser.add_argument("--nsims_steps", type=int, default=1, dest="n_sims_steps",
        help="Number of steps in the simulation escalation (2^x)")

    parser.add_argument("--nsims_probs", type=int, nargs="*", dest="n_sims_probs",
        help="""Number of simulations to do, many arguments accepted.
        Cannot be used with nsims_start and nsims_steps""")

    parser.add_argument("--ncores", type=int, default=-1, dest="n_cores",
        help="Number of cores to run the simulations with")

    parser.add_argument("--nsims_margs", type=int, default=None, dest="n_sims_margs",
        help="Number of simulations to do at each iteration for the marginals")
        
    parser.add_argument("--niter_margs", type=int, default=1, dest="n_iter_margs",
        help="Number of iterations for the computation of marginals")

    
    parser.add_argument("-mg","--marginals", action="store_true",
        help="Calculate marginals together with the source probability")
    parser.add_argument("-nr","--nrepeat", type=int, default=1,
        dest="nrepeat", help="Number of times to repeat the same inst")
    parser.add_argument("--rep_start", type=int, default=0,
        dest="rep_start", help="Repetition index to start from")
    
    parser.add_argument("--overwrite",  action="store_true", help="overwrite save data")


    parser.add_argument("--ninf_run", action="append", type=int,
        help="""Run with this particular number of infected. Can epeat flag with different values""")
    return parser

def calc_set_seed(args: argparse.Namespace, rep:int, rep_start:int):
    m_seed = args.softm_seed is not None
    if args.softm_seed is not None:
        seed_r = args.softm_seed + 2*(rep)
    else:
        seed_r = args.seed + 2*rep
    if args.nrepeat+rep_start > 1 or m_seed or rep !=0:
        propagate.set_seed_numba(seed_r)
        np.random.seed(seed_r)
        print("Set seed: {}".format(seed_r))
    return seed_r

def run_softmargin_single(INSTANCE, rundata, contacts, args, nsims_p, rep, run_inst, seed_r):
    """
    Run the softmargin for set
    """
    tstart = time.time()
    softmRunner = soft_margin.SoftMarginDirectRunner(
        INSTANCE, rundata,
        contacts,
        a_par_max=args.a_max,
        a_par_min=args.a_min,
        a_par_step=args.a_step,
        num_iter=args.n_iter,
        n_sims_per_a=nsims_p
    )
    print("#### NSIM", nsims_p, "REP", rep)
    n_sims_margs = args.n_sims_margs
    MANYSIMS = softmRunner.manysims
    if MANYSIMS:
        RUN_MARGS_W_SRC = False
    elif args.n_sims_margs is not None and n_sims_margs != nsims_p:
        RUN_MARGS_W_SRC = False
    else:
        RUN_MARGS_W_SRC = True
    calc_margs_w_src = (RUN_MARGS_W_SRC and GET_MARGS)
    calc_margs_after = GET_MARGS and not RUN_MARGS_W_SRC
    if GET_MARGS and MANYSIMS:
        warnings.warn("Running with many sims for probs, and calculating marginals")
    if run_inst is None:
        softmRunner.run_soft_margin_all(calc_margs=calc_margs_w_src)
        if calc_margs_after:
            softmRunner.calc_all_margs(args.n_sims_margs, args.n_iter_margs)
    else:
        for c,inst_idx in enumerate(run_inst):
            prev_msg = "{:2d}/{:2d}".format(c+1,len(run_inst),inst_idx)
            softmRunner.run_softmargin(inst_idx, calc_margs_w_src, chain_msg=prev_msg)
            if calc_margs_after:
                softmRunner.extract_marginals(inst_idx,
                                            n_sims=args.n_sims_margs,
                                            n_iter=args.n_iter_margs,
                                            saving=True)

    if not MANYSIMS:
        nsims_p = [nsims_p]
    for num_sim in np.sort(nsims_p).astype(int):
        m_file=NAME_FILE + f"_softm_nsims_{num_sim}"
        if args.n_iter > 1:
            m_file += f"_nit_{args.n_iter}"
        m_file+=f"_rep_{rep}"

        if GET_MARGS and not RUN_MARGS_W_SRC:
            m_file+=f"_mgns_{args.n_sims_margs}"
            m_file+=f"_mgnit_{args.n_iter_margs}"

        name_file_softm = (SAVE_PATH_DIR / m_file).as_posix()

        extra_p = {"seed": seed_r}
        if run_inst is not None:
            extra_p["instances_run"]=io_utils.convert_dtypes_py(list(run_inst))
        else:
            extra_p["instances_run"]=list(range(len(softmRunner.l_obs)))

        softmRunner.save_data(name_file_softm, extra_pars=extra_p,
                    overwrite=args.overwrite, nsims_probs=num_sim)
    del softmRunner
    
    tend = time.time()
    print(f"Time taken: {mlogging.format_time(tend-tstart)}")


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
    confs = np.array(data_["test"])
    np.save(Path(args.path_dir) / (args.str_name_file+f"_{INSTANCE}_confs.npy"), confs)

    ## ************* set algorithm specific parameters *********** 

    type_graph = args.type_graph
    SAVE_PATH_DIR = args.path_dir

    num_conf = args.num_conf
    if h < 3 and num_conf<3:
        x = np.array(data_["test"])
        print(x)
    if args.path_dir == "not_setted":
        SAVE_PATH_DIR = type_graph
    SAVE_PATH_DIR = Path(SAVE_PATH_DIR)
    NAME_FILE = args.str_name_file + str(INSTANCE)
    NAME_FILE +=f"_nc_{num_conf}"


    ## ************ RUN INFERENCE ALGORITHMS ************
    contacts = data_["contacts"]
    
    if args.n_cores > 0:
        n_cores = min(args.n_cores, os.cpu_count())
        nb.set_num_threads(n_cores)
        print(f"Set to use {n_cores} cores")
    else:
        print("Not setting cores to use")
    
    n_sims_margs = args.n_sims_margs
    if args.marginals:
        GET_MARGS = True
        print("Running with marginals")
    else:
        GET_MARGS = False

    run_inst = None
    if args.ninf_run is not None:
        ## Filter by num infected
        ninf=(confs[:,1]>0).sum(-1)
        maskinf = np.zeros(len(confs), np.bool_)
        for i in args.ninf_run:
            maskinf |= (ninf == i)

        run_inst = np.where(maskinf)[0]
        print("Running only with ninf: ", args.ninf_run)
        print("Instances: ", run_inst)

    NSIMS_PROBS = args.n_sims_probs
    RUN_STEPPED_NSIMS = args.n_sims_start > 0
    if RUN_STEPPED_NSIMS and NSIMS_PROBS is not None:
        raise ValueError("Flags nsims_start and nsims_probs cannot be used together")
    if not RUN_STEPPED_NSIMS and NSIMS_PROBS is None:
        raise ValueError("Use either nsims_start or nsims_probs flags")
    if RUN_STEPPED_NSIMS:
        NSIMS_PROBS = [args.n_sims_start*2**s for s in np.arange(args.n_sims_steps)]
    
    if len(NSIMS_PROBS) <= 1:
        ## equivalent to run either way
        ## RUN THE OLD WAY
        RUN_STEPPED_NSIMS = True
    print("#### NSIMS TO DO: ", [pretty_print_n(ns) for ns in NSIMS_PROBS])
    rep_start = args.rep_start
    if RUN_STEPPED_NSIMS:
        for nsim in NSIMS_PROBS:
            for rep in range(rep_start, args.nrepeat+rep_start):
                
                seed_r = calc_set_seed(args, rep, rep_start)
                    
                run_softmargin_single(INSTANCE, data_, contacts, args,
                    int(nsim), rep, run_inst, seed_r)
    else:
        for rep in range(rep_start, args.nrepeat+rep_start):
            seed_r = calc_set_seed(args, rep, rep_start)

            run_softmargin_single(INSTANCE, data_, contacts, args,
                    NSIMS_PROBS, rep, run_inst, seed_r)
