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
path_dir="../patient_zero/proximity/data"

#python bin
python="python3"

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


cd ../../script



for seed in {0..99}
do
    
    #$python ./sib_run_new.py --type_graph $type_graph -T $t_limit -N $N -d $d -scale $scale --lambda $lambda --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir"/sib" --num_conf $num_conf --start_conf $start_conf 2>&1 | tee $path_dir"/sib/sib_run_new_"$seed".log"
    echo
done

for seed in {0..99}
do
    #$python ./nn_run_redo.py --type_graph $type_graph -T $t_limit -N $N -d $d -scale $scale --lambda $lambda --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir"/ann" --n_beta_steps $n_beta_steps --num_conf $num_conf --device $device --iter_marginals $iter_marginals --start_conf $start_conf --num_threads $num_threads --num_samples $num_samples 
    #2>&1 > $path_dir"/ann/ann_run_new_"$seed".log" | tee -a $path_dir"/sm/softmargin_run_multi_"$seed".log" &
    echo
done

for seed in {90..99}
do
    echo $seed
    $python ./softmargin_run_multi.py --type_graph $type_graph -T $t_limit -N $N -d $d --lambda $lambda --mu $mu -scale $scale --seed $seed --path_dir $path_dir"/sm" --num_conf $num_conf --start_conf $start_conf --ncores $ncores --nsims_probs $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --overwrite 2>&1 | tee $path_dir"/sm/softmargin_run_multi_"$seed".log"
done

