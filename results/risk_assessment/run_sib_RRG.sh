#!/bin/bash
<<EOF

RUN NN RRG


type_graph="RRG"
seed=4
N=100
d=10
height=3
lambda=0.04
mu=0.02
t_limit=15
p_edge=1

type_graph="RRG"
seed=7
N=100
d=10
height=3
lambda=0.03
mu=0.
t_limit=15
p_edge=1
nsrc=2

EOF

type_graph="RRG"
N=100
d=10
height=3
lambda=0.035
mu=0.
t_limit=15
p_edge=1
nsrc=1

#num_conf=50
#gamma=1e-3
#path_contacts="../patient_zero/work/work_13_contacts.npz"
#small_lambda_limit=0
path_dir="$(pwd)/sib/${type_graph}_${nsrc}src"

# nn parameters
p_source=1e-2
#step=1e-4
p_sus=0.7


SCRIPT="-u ../../scripts/sib_run_new.py"

#sparse observ
pr_sympt=0.5
#n_test_rnd=3
n_test_rnd=1
delay_test_p="0. 0.4 0.5 0.6"

#init_name="${type_graph}_${n_test_rnd}rnd_psus06_"
init_name="${type_graph}_lastobs_psus07_psrc2_"


num_conf=1
st_conf=0


GEN_GRAPH="--type_graph $type_graph -N $N -d $d -height $height -T $t_limit --lambda $lambda --mu $mu -p_edge $p_edge --num_conf $num_conf --start_conf $st_conf  --n_sources $nsrc"

EXTRA_GEN="--gamma $gamma --path_contacts $path_contacts --small_lambda_limit $small_lambda_limit"


EXTRA_FLAGS=" --init_name_file $init_name --path_dir $path_dir --sparse_obs_last"
SPARSE_OBS="--sparse_obs --sparse_rnd_tests $n_test_rnd --pr_sympt $pr_sympt --delay_test_p $delay_test_p"


SIB_PARS="--p_source $p_source --maxit 1000 --p_sus $p_sus --sib_tol 1e-5"

mkdir -p $path_dir
for seed in $(seq 0 100)
do
    echo $seed
    n_conf=$(( $st_conf+$num_conf ))
    python $SCRIPT $GEN_GRAPH --seed $seed $EXTRA_FLAGS $SPARSE_OBS $SIB_PARS --nthreads 50 #$EXTRA_GEN
done