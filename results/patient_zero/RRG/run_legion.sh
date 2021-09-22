#!/bin/bash


type_graph="RRG"
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
path_dir="$(pwd)/data/ann_3lpow1"
#python bin
#python="/home/ibiazzo/miniconda3/bin/python3"
python="python3 -u"

# nn parameters
p_source=1e-5
n_beta_steps=10000
iter_marginals=100
device="cuda"
# confs
start_conf=0
num_conf=10

#sm parameters
niter=1
ncores=20
nsims="1_000_000 10_000_000 100_000_000"
a_min=0.01
a_max=0.3
a_step=0.01
GEN_EPI="--type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu "
CONFS="--num_conf $num_conf --start_conf $start_conf"

EXTRA_NN="--n_hidden_layers 3 --lin_net_pow 1"
SCRIPTFOLD="../../../scripts"
mkdir -p $path_dir
cd ../../../scripts/
seed_end=$(( $seed_st + $n_seed -1 ))
echo "From seed $seed_st to $seed_end"
for seed in $(seq $seed_st $seed_end)
do
    $python ./nn_run_redo.py $GEN_EPI $CONFS --seed $seed --p_source $p_source --path_dir $path_dir --n_beta_steps $n_beta_steps --device $device --iter_marginals $iter_marginals $EXTRA_NN
    #$python ./sib_run_new.py --type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu --seed $seed --path_dir $path_dir --num_conf $num_conf --start_conf $start_conf --p_source $p_source
    #$python ./softmargin_run_multi.py --type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu --seed $seed --path_dir $path_dir"/sm" --num_conf $num_conf --start_conf $start_conf --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite
    #done
done

