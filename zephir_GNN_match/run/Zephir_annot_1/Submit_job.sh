#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=GNN5T
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:p100:1 
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=deng.h@northeastern.edu
#SBATCH --output=zephir.out

path=/work/venkatachalamlab/Hang/GNN_matching_git/code/02_GNN_match/

# source ~/miniconda3/bin/activate 
module load anaconda3/2022.05  
module load cuda/11.8
source activate deeplearning-cuda11_8


# Variables to pass to Python script
use_GNN=False
file_name="annotations.h5"
t_initil_list="[444]"
python3 ${path}run_GNN_zephir.py $use_GNN "$file_name" "$t_initil_list"

# cp /work/venkatachalamlab/Hang/GNN_matching_git/code/02_GNN_match/ZM9624/annotations.h5 $(dirname "$0")









