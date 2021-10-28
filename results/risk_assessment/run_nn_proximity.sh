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

type_graph="proximity"
seed=1
N=100
d=10
height=3
#lambda=0.026
lambda=0.03
mu=0.
t_limit=15
p_edge=1

scale=2
nsrc=1
#num_conf=50
#gamma=1e-3
#path_contacts="../patient_zero/work/work_13_contacts.npz"
#small_lambda_limit=0
path_dir="$(pwd)/ann/${type_graph}_${nsrc}src/sparse_t/"

# nn parameters
p_source=1e-2
#step=1e-4
#p_rec_t0=1e-7

iter_marginals=100
device="cuda"
num_samples=10000
num_betas=10000


SCRIPT="-u ../../scripts/nn_run_redo.py"
#SCRIPT="-u ../results/script/nn_run_new.py"
#init_name="psrc4_3rnd_pfinNew08_"
#sparse observ
pr_sympt=0.5
#n_test_rnd=3
n_test_rnd=1
delay_test_p="0. 0.2 0.5 0.6"

#init_name="10kst10ks_${n_test_rnd}rnd_"
init_name="10kst10ks_3lpow1_psus07_psrc2_LN_"

GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge -scale $scale"

EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir"
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p" # --sparse_obs_last
TRAINING="--iter_marginals $iter_marginals --p_source $p_source --num_samples $num_samples --device $device --n_beta_steps $num_betas"
#--p_rec_t0 $p_rec_t0 
EXPER="--p_sus 0.7 --p_fin_bal --t_obs $t_limit --n_hidden_layers 3 --lin_net_pow 1 --layer_norm"

num_conf=1
st_conf=0

mkdir -p $path_dir
for seed in $(seq 0 30)
do
    echo $seed
    n_conf=$(( $st_conf+$num_conf ))
    python $SCRIPT $GEN_GRAPH --seed $seed $EXTRA_FLAGS $SPARSE_OBS $TRAINING --start_conf $st_conf --num_conf $n_conf --num_threads 10 $EXPER
done