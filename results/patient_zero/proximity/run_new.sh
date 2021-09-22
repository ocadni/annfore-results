#!/bin/bash


type_graph="proximity"
N=100
d=10
height=3
lambda=0.03
mu=0.02
t_limit=15
scale=2
#gamma=1e-3
#path_contacts="../patient_zero/work/work_13_contacts.npz"
#small_lambda_limit=0
start_conf=0
num_conf=1


#saving path
path_dir="$(pwd)/data/ann"

#python bin
python="python3 -u"

# nn parameters
p_source=1e-5
n_beta_steps=10000
iter_marginals=100
#device="cpu"
device="cuda"
num_threads=2
num_samples=10000

#sm parameters
niter=1
ncores=20
a_min=0.01
a_max=0.3
a_step=0.01
nsims="1_000_000 10_000_000 100_000_000"
#nsims="1_000"

GEN_EPI="--type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu "
CONFS=" --num_conf $num_conf --start_conf $start_conf"
EXTRA_GEN=" -scale $scale"

NN_TRAIN="--n_beta_steps $n_beta_steps --device $device --iter_marginals $iter_marginals --num_threads $num_threads --num_samples $num_samples"

SCRIPTFOLD="../../../scripts"
mkdir -p $path_dir
cd ../../../scripts/

<<REMOVE
for seed in {0..99}
do
    
    #$python ./sib_run_new.py $GEN_EPI $CONFS $EXTRA_GEN --seed $seed --p_source $p_source --path_dir $path_dir"/sib"  2>&1 | tee $path_dir"/sib/sib_run_new_"$seed".log"
    echo ""
done
REMOVE

for seed in 1 #{0..99}
do
    $python ./nn_run_redo.py $GEN_EPI $CONFS $EXTRA_GEN --seed $seed --p_source $p_source --path_dir $path_dir $NN_TRAIN  --all_graph_neighs --n_hidden_layers 3 --lin_net_pow 1 # --lay_less_deep
    #2>&1 > $path_dir"/ann/ann_run_new_"$seed".log" | tee -a $path_dir"/sm/softmargin_run_multi_"$seed".log" &
    echo ""
    #
done
<<DONE

for seed in {90..99}
do
    #echo $seed
    #$python ./softmargin_run_multi.py $GEN_EPI $CONFS $EXTRA_GEN --seed $seed --path_dir $path_dir"/sm" --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite 2>&1 | tee $path_dir"/sm/softmargin_run_multi_"$seed".log"
done

DONE