#!/bin/bash

mu=1e-10
type_graph="TREE"
height=6
num_conf=4
t_limit=15
d=4


#saving path
path_dir="$(pwd)/data/crisp_test"

#python bin
python="python3"

# nn parameters
p_source=5e-5

## crisp parameters
#n_steps=100000
n_burnin=1000
p_autoinf=1e-8
p_wrong_obs=1e-9


#python bin
python="python3"

num_proc=10

name="test_"

CRISP_PARS="--p_autoinf $p_autoinf  --p_wrong_obs $p_wrong_obs --n_burnin $n_burnin --p_source $p_source"

SCRIPT="-u ../../scripts/crisp_run.py"
N_STEPS_MC="20000 50000 100000 200000 500000"

PARALL="--n_steps_list $N_STEPS_MC --n_proc $num_proc --rep_range 0 10" #  --

GEN=" -d $d -height $height -T $t_limit  --mu $mu --seed $seed --path_dir $path_dir " 

mkdir -p $path_dir



for seed in 0 1 2
do
    for lam in 0.1 0.2 0.3 0.4 0.5 0.6
    do
        echo "LAMBDA: $lam"
        seed_mc=$((seed*59+3))
        
        echo "$name"
         
        $python $SCRIPT --num_conf $num_conf --lambda $lam --init_name_file $name $CRISP_PARS $PARALL $GEN --seed_mc $seed_mc 

        
    done
done
