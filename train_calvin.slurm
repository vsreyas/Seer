#!/bin/bash
#SBATCH --job-name=seer_calvin_d         # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gres=gpu:1                     # GPUs per node (change if node has >1 GPU)
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=48:00:00                  # Walltime (hh:mm:ss)
#SBATCH --partition=general              # Partition/queue name
#SBATCH --output=/scratch/sreyasv/logs/%x_%j.out   # Stdout log
#SBATCH --error=/scratch/sreyasv/logs/%x_%j.err    # Stderr log

# Activate your conda environment
source /home/sreyasv/miniconda3/etc/profile.d/conda.sh
conda activate seer

# Paths
calvin_dataset_path="/data/user_data/sreyasv/calvin_dataset/task_D_D"
save_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/seer_calvin_d"
vit_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/mae_pretrain_vit_base.pth"

mkdir -p ${save_checkpoint_path}

# Auto-detect GPU count per node
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# Launch torchrun
torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_port=10211 \
  train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 10 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --finetune_type "calvin" \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name pretrain_seer_calvin_d \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "pretrain" \
    --action_pred_steps 3 \
    --sequence_length 14 \
    --future_steps 3 \
    --window_size 17 \
    --obs_pred \
    --loss_image \
    --loss_action \
    --atten_goal 4 \
    --atten_goal_state \
    --atten_only_obs \
    --save_checkpoint \
    --report_to_wandb \
    # --except_lang \

