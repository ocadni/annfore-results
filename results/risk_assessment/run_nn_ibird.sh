#!/bin/bash

type_graph="i_bird"
N=100
d=10
height=3
lambda=0.04
mu=0.
t_limit=14
p_edge=1
seed=3
scale=1
nsrc=1

gamma=2e-4
small_lambda_limit=300
path_contacts="../patient_zero/i_bird/i_bird_contacts.npz"


#num_conf=50
#small_lambda_limit=0
path_dir="$(pwd)/ann/good_new/${type_graph}_${nsrc}src"

# nn parameters
p_source=1e-4
#step=1e-4
#p_rec_t0=1e-7

iter_marginals=100
device="cuda:1"
num_samples=10000
num_betas=10000


SCRIPT="-u ../../scripts/nn_run_redo.py"
#SCRIPT="-u ../results/script/nn_run_new.py"
#init_name="psrc4_3rnd_pfinNew08_"
#sparse observ
pr_sympt=0.5
n_test_rnd=0
delay_test_p="0. 0.8 1. 1. 1. 1."

#init_name="psrc4_${n_test_rnd}rnd_newfin06_"
#init_name="8kst8ks_lastobs_psus06DSC_6xpr_"
#init_name="10kst10ks_${n_test_rnd}rnd_psus06_4lpow2_nn_"
init_name="10kst10ks_lastobs_psus06_4lpow2_"

GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge --n_sources $nsrc"

EXTRA_GEN="--gamma $gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"

EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir --sparse_obs_last"
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p"
TRAINING="--iter_marginals $iter_marginals --p_source $p_source --num_samples $num_samples --device $device --n_beta_steps $num_betas"
#--p_rec_t0 $p_rec_t0 
EXPER="--p_sus 0.6 --p_fin_bal --t_obs $t_limit --n_hidden_layers 3 --lin_net_pow 2"

num_conf=1
#start_conf=0

mkdir -p $path_dir
for st_conf in 2
do
    n_conf=$(( $st_conf+$num_conf ))
    python $SCRIPT $GEN_GRAPH $EXTRA_GEN --seed $seed $EXTRA_FLAGS $SPARSE_OBS $TRAINING --start_conf $st_conf --num_conf $n_conf --num_threads 10 $EXPER
done