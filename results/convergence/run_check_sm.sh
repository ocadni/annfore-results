#!/bin/bash

mu=1e-10
type_graph="TREE"
path_dir="$(pwd)/trial"
niter=1
ncores=20
#nsims=100000
a_min=0.01
a_max=0.22
a_step=0.01

# cd ..

seed=1
num_conf=1
SCRIPT="../script/softmargin_run_multi.py"
EXTRA_FLAGS=" --a_min $a_min --a_max $a_max --a_step $a_step  --overwrite"
mkdir -p $path_dir

for nsims in 4000
do
    python $SCRIPT -N 10 -d 4 -height 3 -T 15 --lambda 0.2 --mu $mu --seed $seed --init_name_file $nsims --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims_start $nsims $EXTRA_FLAGS
    #python $SCRIPT -N 10 -d 4 -height 4 -T 15 --lambda 0.3 --mu $mu --seed $seed  --init_name_file $nsims --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --margs_w_src true --overwrite
    #python $SCRIPT -N 10 -d 4 -height 5 -T 15 --lambda 0.4 --mu $mu --seed $seed  --init_name_file $nsims --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --margs_w_src true --overwrite
    #python $SCRIPT -N 10 -d 4 -height 6 -T 15 --lambda 0.5 --mu $mu --seed $seed  --init_name_file $nsims --path_dir $path_dir --num_conf $num_conf --ncores $ncores --nsims $nsims  --a_min $a_min --a_max $a_max --a_step $a_step --margs_w_src true --overwrite
done

