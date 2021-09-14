#!/bin/bash

seed=0
start_conf=0
num_conf=1

type_graph="i_bird"
N=100
d=10
height=3
#lambda=0.04
mu=0.02
t_limit=14
gamma=2e-4
path_contacts="../patient_zero/i_bird/i_bird_contacts.npz"
small_lambda_limit=300

#saving path
path_dir="../patient_zero/i_bird/data"

#python bin
python="python3"

# nn parameters
p_source=1e-5
step=1e-4
iter_marginals=100
device="cuda"

#sm parameters
niter=1
ncores=30
nsims=100000000
a_min=0.01
a_max=0.3
a_step=0.01


cd ../../script

#$python ./nn_run_new.py --type_graph $type_graph -T $t_limit -N $N -d $d --gamma $gamma --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir --step $step --num_conf $num_conf --device $device --iter_marginals $iter_marginals --start_conf $start_conf --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --less_deeper

for seed in 16
do
    python ./softmargin_run_new.py  --type_graph $type_graph -T $t_limit --gamma $gamma  --mu $mu --seed $seed  --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf
done