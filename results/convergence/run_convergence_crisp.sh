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

CRISP_PARS="--p_autoinf $p_autoinf  --p_wrong_obs $p_wrong_obs --n_burnin $n_burnin --p_source $p_source"

SCRIPT="-u ../../scripts/crisp_run.py"
mkdir -p $path_dir
N_STEPS_MC="20000 50000 100000 200000 500000"
#N_STEPS_MC="100 200"
PARALL="--n_steps_list $N_STEPS_MC --n_proc $num_proc --rep_range 0 10" #  --n_steps $n_steps
#n_steps=10000
for seed in 0 1 2
do
    for lambda in 0.1 0.2 0.3 0.4 0.5 0.6
    do
        echo "LAMBDA: $lambda"
        seed_mc=$((seed*59+3))
        name="nsteps_${n_steps}_"
        echo "$name"
         
        $python $SCRIPT -d $d -height $height -T $t_limit --lambda $lambda --mu $mu --seed $seed --path_dir $path_dir --num_conf $num_conf  --init_name_file $name $CRISP_PARS $PARALL --seed_mc $seed_mc 

        
    done
done
