runfile=$1
num_conf=5

echo "Launching from conf $2 for $num_conf conf"
cat << EOF > temp.sbatch 
#!/bin/bash
#SBATCH --job-name=nnepi_${num_conf}_work_${2}
#SBATCH --partition=cuda
#SBATCH --time=25:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --mem=1900MB
#SBATCH --mail-type=END,FAIL
module load nvidia/cudasdk/10.1
module load intel/libraries/tbb/2019.4.243
source /home/fmazza/lib/mambaforge/bin/activate
conda activate pytorch

start_conf=$2
num_conf=$num_conf


EOF

cat $1 >> temp.sbatch

sbatch temp.sbatch
#sh temp.sbatch
