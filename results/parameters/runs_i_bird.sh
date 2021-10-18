#!/bin/bash

#SBATCH --time=72:00:00 
#SBATCH --ntasks=1
#SBATCH --partition=cuda 
#SBATCH --gres=gpu:1
#SBATCH --job-name=i_bird_params 
#SBATCH --mem=3GB
#SBATCH --mail-type=ALL
#SBATCH --output=/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/i_bird/run_new.out
#SBATCH --err=/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/i_bird/run_new.err


#interaction graph
type_graph="data_deltas_2_gamma"
N=95
d=10
height=3

#epidemic parameters

lambda=0.5
mu=0.02
t_limit=12
scale=2
gamma1=1.5e-4
gamma2=3e-5
path_contacts="/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/i_bird/i_bird_contacts.npz"
#path_contacts="../results/parameters/i_bird/i_bird_contacts.npz"
small_lambda_limit=300

#configurations
start_conf=0
num_conf=1
ninf_min=40
seed=0

#learning params
gamma1_init_param=1e-3
gamma2_init_param=1e-3

# bias source learning rae
p_source=1e-4

#saving path
#path_dir="../results/parameters/i_bird/data/"
path_dir="/home/ibiazzo/git/ann_results/nnepi-results/results/parameters/i_bird/data/"
init_name_file="eq_05_lr_"
#python bin
#python="python3"
python="/home/ibiazzo/miniconda3/bin/python3"


# sib paramters
iter_learn=500
lr_param_sib=3e-6
maxit=20

# nn parameters
lr=3e-4
step=5e-5
iter_marginals=100
device="cuda"
#device="cpu"
num_samples=10000
num_threads=1
num_end_iter=100
beta_start_learn=0.5
lin_net_pow=1

GEN_EPI="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --gamma1 $gamma1 --gamma2 $gamma2"
CONFS=" --num_conf $num_conf --start_conf $start_conf --ninf_min $ninf_min"

ANN_CONF="--p_source $p_source --step $step --device $device --iter_marginals $iter_marginals --num_samples $num_samples --lr $lr --num_threads $num_threads --num_end_iter $num_end_iter --init_name_file $init_name_file  --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"
ANN_CONF_LEARN="--beta_start_learn $beta_start_learn --gamma1_init_param $gamma1_init_param --gamma2_init_param $gamma2_init_param "

ANN_LAYERS="--lay_less_deep"
#ANN_LAYERS="--lay_deep_sc"
#ANN_LAYERS="--lin_net_pow $lin_net_pow"

SIB_CONF="--p_source $p_source --lambda_init_param $lambda_init_param --mu_init_param $mu_init_param --iter_learn $iter_learn --lr_param $lr_param_sib --maxit $maxit --lr_gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"

cd ../../scripts

#for seed in {1..10}
#do
    #$python ./sib_run_new.py  $GEN_EPI $CONFS $SIB_CONF --seed $seed --path_dir $path_dir 
$python ./nn_run_two_params.py  $GEN_EPI $CONFS $ANN_CONF $ANN_CONF_LEARN $ANN_LAYERS --seed $seed --path_dir $path_dir 
#done
