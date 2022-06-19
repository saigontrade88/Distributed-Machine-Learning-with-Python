#! /bin/bash
#SBATCH --job-name="Chap3"
#SBATCH --output=out_chap3_dataParallel_mp_%j.txt
#SBATCH --error=err_chap3_dataParallel_mp_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
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

python3 single_layer.py

#SBATCH --partition=simmons_itn18 
#SBATCH --qos=preempt_short
#SBATCH --partition=snsm_itn19  
#SBATCH --qos=snsm19_special

conda deactivate





