#!/bin/bash

#SBATCH --time=5:10:00 
#SBATCH --ntasks=1
#SBATCH --partition=cuda 
#SBATCH --gres=gpu:1
#SBATCH --job-name=work_params 
#SBATCH --mem=3GB
#SBATCH --mail-type=ALL
#SBATCH --output=/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/work/run_new.out
#SBATCH --err=/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/work/run_new.err


#interaction graph
type_graph="data_deltas"
N=95
d=10
height=3

#epidemic parameters

lambda=0.5
mu=0.02
t_limit=12
scale=2
gamma=1e-3
path_contacts="/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/work/work_13_contacts.npz"
small_lambda_limit=0

#configurations
start_conf=0
num_conf=1
ninf_min=5
seed=0

#learning params
lambda_init_param=0.01
mu_init_param=0.01

# bias source learning rae
p_source=1e-4

#saving path
path_dir="/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/work/data/"
init_name_file="eq_05_"
#python bin
#python="python3"
python="/home/ibiazzo/miniconda3/bin/python3"


# sib paramters
iter_learn=1
lr_param_sib=2e-6
maxit=20

# nn parameters
lr=1e-3
step=1e-4
iter_marginals=100
device="cuda"
#device="cpu"
num_samples=10000
num_threads=1
num_end_iter=100
beta_start_learn=0.5
lin_net_pow=1

GEN_EPI="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu --gamma $gamma"
CONFS=" --num_conf $num_conf --start_conf $start_conf --ninf_min $ninf_min"

ANN_CONF="--p_source $p_source --step $step --device $device --iter_marginals $iter_marginals --lambda_init_param $lambda_init_param --mu_init_param $mu_init_param --lr_param --num_samples $num_samples --lr $lr --num_threads $num_threads --num_end_iter $num_end_iter --init_name_file $init_name_file --beta_start_learn $beta_start_learn --lr_gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"
ANN_LAYERS="--lay_deep_eq"
#ANN_LAYERS="--lay_deep_sc"
#ANN_LAYERS="--lin_net_pow $lin_net_pow"

SIB_CONF="--p_source $p_source --lambda_init_param $lambda_init_param --mu_init_param $mu_init_param --iter_learn $iter_learn --lr_param $lr_param_bp --maxit $maxit --lr_gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"

cd ../../scripts

#for seed in {1..20}
#do
    #$python ./sib_run_new.py  $GEN_EPI $CONFS $SIB_CONF --seed $seed --path_dir $path_dir 
$python ./nn_run_redo.py  $GEN_EPI $CONFS $ANN_CONF $ANN_LAYERS --seed $seed --path_dir $path_dir 
#done
