runfile=$1
nseed=5
type_graph="proxim"

echo "Launching with $nseed seeds (cases)"
cat << EOF > temp.sbatch 
#!/bin/bash
#SBATCH --job-name=nnepi_${type_graph}_$2
#SBATCH --partition=cuda
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --mem=1800MB
#SBATCH --mail-type=END,FAIL
module load nvidia/cudasdk/10.1
module load intel/libraries/tbb/2019.4.243

source /home/fmazza/lib/mambaforge/bin/activate
conda activate pytorch

seed_st=$2
n_seed=$nseed
EOF

cat $1 >> temp.sbatch

#sh temp.sbatch
sbatch temp.sbatch
