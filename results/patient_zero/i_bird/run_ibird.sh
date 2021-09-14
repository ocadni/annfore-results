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
num_samples=10000
#sm parameters
niter=1
ncores=20
nsims=1000000
a_min=0.01
a_max=0.22
a_step=0.01


cd ../../script
#27 28 31 34 45 50
#for seed in  66 72 75 76 80 82 85 89
#do
#    $python ./nn_run_new.py --type_graph $type_graph -T $t_limit -N $N -d $d --gamma $gamma --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir --step $step --num_conf $num_conf --device $device --iter_marginals $iter_marginals --start_conf $start_conf --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --less_and_more_deep --num_samples $num_samples
#done
for seed in {0..99}
do
    python ./sib_run_new.py --type_graph $type_graph -T $t_limit -N $N -d $d --gamma $gamma --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir --num_conf $num_conf --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf
done

#python ./softmargin_run.py  --type_graph $type_graph -T $t_limit --gamma $gamma  --mu $mu --seed $seed  --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims_margs $nsims_margs  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf