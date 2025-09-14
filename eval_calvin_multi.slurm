#!/bin/bash
#SBATCH --job-name=seer_calvin_d_eval_multi   # Job name
#SBATCH --nodes=4                             # Number of nodes (same as training)
#SBATCH --gres=gpu:1                          # GPUs per node
#SBATCH --cpus-per-task=8                     # CPU cores per task
#SBATCH --mem=64G                             # Memory per node
#SBATCH --time=48:00:00                       # Walltime (shorter than training)
#SBATCH --partition=general                   # Partition/queue name
#SBATCH --output=/home/sreyasv/Projects/Seer/logs/%x_%j.out   # Stdout log
#SBATCH --error=/home/sreyasv/Projects/Seer/logs/%x_%j.err    # Stderr log

# Activate your conda environment
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate seer

# Paths (reuse from training)
calvin_dataset_path="/data/user_data/sreyasv/calvin_dataset/task_D_D"
save_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/seer_calvin_d"
vit_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/mae_pretrain_vit_base.pth"
calvin_conf_path="/home/sreyasv/calvin/calvin_models/conf"

# Update this to the checkpoint you want to evaluate
resume_from_checkpoint="${save_checkpoint_path}/pretrain_seer_calvin_d_multi/19.pth"

# For logging eval runs
IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="${save_checkpoint_path}/pretrain_seer_calvin_d_multi/eval_logs/$run_name"
mkdir -p "$log_folder"
log_file="$log_folder/evaluate_$log_name.log"

# Multi-node launch parameters
GPUS_PER_NODE=1
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# Launch evaluation
srun --ntasks=${WORLD_SIZE} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --gpus-per-task=1 \
     python -u -m torch.distributed.run \
       --nnodes=${SLURM_NNODES} \
       --nproc_per_node=${GPUS_PER_NODE} \
       --rdzv_backend=c10d \
       --rdzv_endpoint=${MASTER_ADDR}:29501 \
       eval_calvin.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset ${calvin_dataset_path} \
        --calvin_conf_path ${calvin_conf_path} \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --num_obs_token_per_image 9 \
        --run_name ${run_name} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --transformer_layers 24 \
        --hidden_dim 384 \
        --transformer_heads 12 \
        --phase "evaluate" \
        --finetune_type "calvin" \
        --action_pred_steps 3 \
        --sequence_length 10 \
        --future_steps 3 \
        --window_size 13 \
        --obs_pred \
        --resume_from_checkpoint ${resume_from_checkpoint} | tee ${log_file} \
         
