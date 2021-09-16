#!/bin/bash
<<EOF

RUN SIB WITH MANY SOURCES
#num_conf=50

type_graph="RRG"
seed=8
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=15
p_edge=1.


EOF
<<PROXIMITY

type_graph="proximity"
seed=1
N=1000
d=10
height=3
lambda=0.025
mu=0.02
t_limit=15
p_edge=1
nsrc=10
PROXIMITY

type_graph="i_bird"
seed=9
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=12
p_edge=1
nsrc=2

##used by proximity
scale=2
## used by work and i_bird
gamma=1e-3
small_lambda_limit=0
path_contacts="../../results/patient_zero/work/work_13_contacts.npz"
path_dir="$(pwd)/sib/${type_graph}_${nsrc}src/"

p_source=1e-4
p_sus=0.6

num_conf=20
start_conf=0
SCRIPT="-u ../../results/script/sib_run_new.py"

#sparse observ
pr_sympt=0.5
n_test_rnd=0
delay_test_p="0. 1. 1. 1. 1."


init_name="${type_graph}_${n_test_rnd}rnd_tinf5_psus60_"
#init_name="${type_graph}_tinf10_psus60_"


GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge --num_conf $num_conf --start_conf $start_conf -scale $scale  --n_sources $nsrc"

EXTRA_GEN="--gamma $gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --sp_obs_min_tinf 5"


EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir"
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p"
TRAINING="--step $step --iter_marginals $iter_marginals --p_source $p_source --num_samples $num_samples --device $device"

SIB_PARS="--p_source $p_source --maxit 1000 --p_sus $p_sus --sib_tol 1e-4"

mkdir -p $path_dir

#python $SCRIPT $GEN_GRAPH --seed $seed $EXTRA_FLAGS $SPARSE_OBS $SIB_PARS --nthreads 20
#for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for dummy in 1
do
    python $SCRIPT $GEN_GRAPH --seed $seed $EXTRA_FLAGS $SPARSE_OBS $SIB_PARS --nthreads 10 $EXTRA_GEN
done