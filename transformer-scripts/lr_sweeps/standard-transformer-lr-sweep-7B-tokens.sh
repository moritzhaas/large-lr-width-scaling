#!/bin/bash
#SBATCH --time=3-00:00:00  # Runtime in D-HH:MM:SS    
#SBATCH --output=/mnt/lustre/work/luxburg/shared_data/moritz_sebastian_2025/logs/standard-transformer-lr-sweep-7B-tokens/%x_%A_%a.out  
#SBATCH --error=/mnt/lustre/work/luxburg/shared_data/moritz_sebastian_2025/logs/standard-transformer-lr-sweep-7B-tokens/%x_%A_%a.err   
#SBATCH --open-mode=append
#SBATCH --job-name=litsweep  
#SBATCH --partition=a100-galvani 
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=128G          
#SBATCH --cpus-per-task=8

# script parameters
LR=${1:-0.001}
WIDTH=${2:-256} 
WARMUP=${3:-700}
MBS=${4:-64}
ID=${5:-0}

scontrol show job ${SLURM_JOB_ID}
nvidia-smi
cd $WORK
export NCCL_TIMEOUT=1800000
export WANDB__SERVICE_WAIT=6000

cd litgpt/pretrain-experiment
source activate tp-theory-new

python pretrain-experiment.py \
    --run_name "width=${WIDTH}-lr=${LR}-warmup=${WARMUP}" \
    --run_id $ID \
    --experiment_name "standard-transformer-lr-sweep-7B-tokens" \
    --model "pythia-14m" \
    --qk_norm \
    --width $WIDTH \
    --data_dir "/mnt/lustre/work/luxburg/shared_data/dclm-baseline-1.0-tokenized" \
    --max_tokens 7000000000 \
    --max_seq_length 512 \
    --global_batch_size 256 \
    --micro_batch_size $MBS \
    --output_dir "/mnt/lustre/work/luxburg/shared_data/moritz_sebastian_2025/" \
    --lr $LR \
    --warmup_steps $WARMUP \
    --precision "bf16-mixed" \
    --seed 42 \
    --no-monitor \
    --compile \
    --resume
