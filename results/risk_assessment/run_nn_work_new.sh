#!/bin/bash

type_graph="i_bird"
N=100
d=10
height=3
lambda=0.03
mu=0.
t_limit=12
p_edge=1
## 1 src last obs -> seed 6
#seed=4
#nsrc=2
seed=6
nsrc=1

gamma=1e-3
small_lambda_limit=0
path_contacts="../../results/patient_zero/work/work_13_contacts.npz"


#num_conf=50
#small_lambda_limit=0
path_dir="$(pwd)/ann/good_new/${type_graph}_${nsrc}src"

# nn parameters
p_source=1e-2
#step=1e-4
#p_rec_t0=1e-7

iter_marginals=100
device="cuda:0"
num_samples=10000
num_betas=10000


SCRIPT="-u ../../results/script/nn_run_redo.py"
#SCRIPT="-u ../results/script/nn_run_new.py"

#sparse observ
pr_sympt=0.5
#n_test_rnd=3
n_test_rnd=0
delay_test_p="0. 0.4 0.5 0.6"

#init_name="psrc4_${n_test_rnd}rnd_newfin06_"
#init_name="10kst10ks_${n_test_rnd}rnd_4lpow_"
#init_name="10kst10ks_lastobs_4lpow_psusPr_"
#init_name="10kst10ks_lastobs_4lpow_ps06_l0try_"
init_name="10kst10ks_lastobs_4lpow_psus07_psrc2_"


GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge --n_sources $nsrc"

EXTRA_GEN="--gamma $gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"

EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir --sparse_obs_last" #--layer_norm batch --no_bias" 
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p"
TRAINING="--iter_marginals $iter_marginals --p_source $p_source --num_samples $num_samples --device $device --n_beta_steps $num_betas"
#--p_rec_t0 $p_rec_t0 
EXPER="--p_sus 0.7 --t_obs $t_limit --n_hidden_layers 4 --p_fin_bal" # --lin_net_pow 0.5" 

num_conf=5
start_conf=75
echo $start_conf

mkdir -p $path_dir
for st_conf in $start_conf
do
    n_conf=$(( $st_conf+$num_conf ))
    python $SCRIPT $GEN_GRAPH $EXTRA_GEN --seed $seed $EXTRA_FLAGS $SPARSE_OBS $TRAINING --start_conf $st_conf --num_conf $n_conf --num_threads 10 $EXPER
done
