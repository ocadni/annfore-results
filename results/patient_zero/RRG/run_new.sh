#!/bin/bash


type_graph="RRG"
seed=1
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=15
#gamma=1e-3
#path_contacts="../patient_zero/work/work_13_contacts.npz"
#small_lambda_limit=0

#saving path
#path_dir="/home/ibiazzo/nn_epidemics/results/patient_zero/RRG/data"
#path_dir="../patient_zero/RRG/data"
path_dir="$(pwd)/data/"
#python bin
#python="/home/ibiazzo/miniconda3/bin/python3"
python="python3"

# nn parameters
p_source=1e-5
n_beta_steps=10000
iter_marginals=100
device="cuda:0"
start_conf=0
num_conf=10

#sm parameters
niter=1
ncores=20
nsims="1_000_000 10_000_000 100_000_000"
a_min=0.01
a_max=0.3
a_step=0.01

GEN_EPI="--type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu"
CONFS=" --num_conf $num_conf --start_conf $start_conf"


SCRIPTFOLD="../../../scripts"
mkdir -p $path_dir
cd ../../../scripts/
for seed in {0..9}
do
    #$python ./nn_run_redo.py $GEN_EPI $CONFS --seed $seed --p_source $p_source --path_dir $path_dir --n_beta_steps $n_beta_steps --device $device --iter_marginals $iter_marginals --lay_deep_eq
    $python ./sib_run_new.py $GEN_EPI $CONFS --seed $seed --path_dir $path_dir"sib" --p_source $p_source
    #$python ./softmargin_run_multi.py $GEN_EPI $CONFS --seed $seed --path_dir $path_dir --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite
    #done
done


#python ./softmargin_run.py  --type_graph $type_graph -T $t_limit --gamma $gamma  --mu $mu --seed $seed  --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims_margs $nsims_margs  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit --start_conf $start_conf
