#!/bin/bash
name="TREE"
for seed in {1..10}
do 
	sed -E "/^seed=/ s/(.{5}).{1}/\1$seed /" runs_$name.sh > temp.sh
	sleep 0.1
        sed -i "s/run_new.out/run_new_$seed.out/g" temp.sh
	sed -i "s/run_new.err/run_new_$seed.err/g" temp.sh
	sed -i "s/job-name=$name_params/job-name=$name_params_$seed/g" temp.sh
	chmod +x temp.sh
	#./temp.sh
	sbatch temp.sh
	sleep 0.5
done
