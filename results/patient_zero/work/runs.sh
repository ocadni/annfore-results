#!/bin/bash

type_graph="i_bird"
gamma=1e-3
mu=0.02
start_conf=0
num_conf=50
t_limit=12
small_lambda_limit=0
path_contacts="../patient_zero/work/work_13_contacts.npz"

#saving path
path_dir="../patient_zero/work/data"

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


cd ../../script

#for seed in 1 2
for seed in 2
do
    #python ./nn_run_redo.py --type_graph $type_graph -T $t_limit --gamma $gamma --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir"/ann" --n_beta_steps $n_beta_steps --num_conf $num_conf --device $device --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --iter_marginals $iter_marginals --start_conf $start_conf

    #python ./sib_run_new.py --type_graph $type_graph -T $t_limit --gamma $gamma --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir"/sib" --num_conf $num_conf --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf


    python ./softmargin_run_multi.py  --type_graph $type_graph -T $t_limit --gamma $gamma  --mu $mu --seed $seed  --path_dir $path_dir"/sm" --num_conf $num_conf --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf

done