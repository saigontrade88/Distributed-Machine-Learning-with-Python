#! /bin/bash
#SBATCH --job-name="Chap2"
#SBATCH --output=out_chap_%j.txt
#SBATCH --error=err_chap2_%j.txt
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-user=longdang@usf.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

source $HOME/.bashrc

module add apps/cuda/11.3.1

# Activate your environment
conda activate torch_171

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python3 main.py

#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short

conda deactivate





