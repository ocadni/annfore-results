#!/bin/bash
<<EOF

RUN NN RRG
type_graph="RRG"
seed=7
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=15
p_edge=1

EOF

type_graph="i_bird"
seed=9
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=12
p_edge=1


gamma=1e-3
small_lambda_limit=0
path_contacts="../../results/patient_zero/work/work_13_contacts.npz"



#num_conf=50
#small_lambda_limit=0
path_dir="$(pwd)/ann/good_new/${type_graph}"

# nn parameters
p_source=1e-4
#step=1e-4
#p_rec_t0=1e-7

iter_marginals=100
device="cuda:1"
num_samples=6000
num_betas=6000


SCRIPT="-u ../../results/script/nn_run_redo.py"
#SCRIPT="-u ../results/script/nn_run_new.py"
#init_name="psrc4_3rnd_pfinNew08_"
#sparse observ
pr_sympt=0.5
#n_test_rnd=3
n_test_rnd=0
delay_test_p="0. 0.6 0.4"

#init_name="psrc4_${n_test_rnd}rnd_newfin06_"
init_name="6kst6ks_lastobs_newfin06DSCxa_expNd_"

GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge"

EXTRA_GEN="--gamma $gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"

EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir --sparse_obs_last"
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p"
TRAINING="--iter_marginals $iter_marginals --p_source $p_source --num_samples $num_samples --device $device --n_beta_steps $num_betas"
#--p_rec_t0 $p_rec_t0 
EXPER="--p_sus 0.6 --p_fin_bal --t_obs $t_limit --deeper --init xavier"

num_conf=5
#start_conf=0

mkdir -p $path_dir
for st_conf in 5
do
    n_conf=$(( $st_conf+$num_conf ))
    python $SCRIPT $GEN_GRAPH $EXTRA_GEN --seed $seed $EXTRA_FLAGS $SPARSE_OBS $TRAINING --start_conf $st_conf --num_conf $n_conf --num_threads 10 $EXPER
done