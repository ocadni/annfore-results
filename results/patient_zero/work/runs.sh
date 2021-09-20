#!/bin/bash

type_graph="i_bird"
gamma=1e-3
mu=0.02
start_conf=0
num_conf=50
t_limit=12
small_lambda_limit=0

path_contacts="$(pwd)/work_13_contacts.npz"

#saving path
path_dir="$(pwd)/data/ann"

#python bin
python="python3"

# nn parameters
p_source=1e-5
n_beta_steps=10000
iter_marginals=100
device="cuda:1"

#sm parameters
niter=1
ncores=10
a_min=0.01
a_max=0.3
a_step=0.01
nsims="1_000_000 10_000_000 100_000_000"
#nsims="1_000"

GEN_EPI="--type_graph $type_graph -T $t_limit --mu $mu"
CONFS=" --num_conf $num_conf --start_conf $start_conf"
EXTRA_GEN="--path_contacts $path_contacts --small_lambda_limit $small_lambda_limit  --gamma $gamma"

SCRIPTFOLD="../../../scripts"
mkdir -p $path_dir
cd ../../../scripts/

#for seed in 1 2
for seed in 1
do
    python ./nn_run_redo.py $GEN_EPI $CONFS $EXTRA_GEN --seed $seed --p_source $p_source --path_dir $path_dir --n_beta_steps $n_beta_steps --device $device --iter_marginals $iter_marginals --lay_less_deep

    #python ./sib_run_new.py $GEN_EPI $CONFS $EXTRA_GEN --seed $seed --p_source $p_source --path_dir $path_dir"/sib"


    #python ./softmargin_run_multi.py $GEN_EPI $CONFS $EXTRA_GEN --path_dir $path_dir"/sm" --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite

done