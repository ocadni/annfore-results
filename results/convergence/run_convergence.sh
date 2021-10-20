#!/bin/bash

mu=1e-10
type_graph="TREE"
height=6
num_conf=1
t_limit=15
d=4


#saving path
path_dir="../convergence/data"

#python bin
python="python3"

# nn parameters
p_source=1e-5
num_samples=1000
iter_marginals=100
device="cuda:0"
device="cpu"
num_threads=1
lr=5e-4
#python bin
python="python3"

cd ../script/

for seed in 0 1 2
do
    for lambda in 0.1 0.2 0.3 0.4 0.5 0.6
    do
        #$python ./sib_run_new.py -d $d -height $height -T $t_limit --lambda $lambda --mu $mu --seed $seed --p_source $p_source --path_dir $path_dir --num_conf $num_conf 
        echo
    done
done

for seed in 1 2
do
    for step in 3.05175781e-05
#1.52587891e-05 7.62939453e-06 3.81469727e-06 1.90734863e-06
    do
        for lambda in 0.4 0.5 0.6
        do
        #echo $step
            $python ./nn_run_new.py -d $d -height $height -T $t_limit --lambda $lambda --mu $mu --seed $seed --init_name_file $step --p_source $p_source --path_dir $path_dir --step $step --num_conf $num_conf --device $device --iter_marginals $iter_marginals --num_samples $num_samples --num_threads $num_threads --init_name_file $step --lr $lr #>> "../convergence/log/script_$seed\_$lambda\_$step.log" 2>&1 &
        done
    done
done
