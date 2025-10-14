#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=l40s-48:2


set -x

module load anaconda3/2024.10-1 
conda activate seer

# ==== CONFIGURE PATHS ====
save_checkpoint_path="/ocean/projects/cis250181p/venkatas/seer_libero_finetune_depth/checkpoints"
root_dir="/ocean/projects/cis250181p/venkatas/Seer/data/LIBERO"
libero_path="/ocean/projects/cis250181p/venkatas/Seer/LIBERO"
finetune_from_pretrained_ckpt="/ocean/projects/cis250181p/venkatas/seer_libero_pretrain_depth/checkpoints/libero_pretrain_depth_clip_resnet_action_only/19.pth"
### ================================== ###
mkdir -p ${save_checkpoint_path}

GPUS_PER_NODE=2
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))


srun --ntasks=${WORLD_SIZE} \
     --ntasks-per-node=${GPUS_PER_NODE} \
      torchrun --nproc_per_node=2 --standalone \
       train.py \
         --traj_cons \
         --rgb_pad 10 \
         --gripper_pad 4 \
         --gradient_accumulation_steps 8 \
         --workers 16 \
         --lr_scheduler cosine \
         --save_every_iter 10000 \
         --num_epochs 50 \
         --seed 42 \
         --batch_size 48 \
         --precision fp32 \
         --learning_rate 1e-3 \
         --save_checkpoint \
         --finetune_type libero_finetune \
         --root_dir ${root_dir} \
         --wandb_project seer \
         --weight_decay 1e-4 \
         --num_resampler_query 3 \
         --run_name libero_finetune_depth_clip_resnet_action_only \
         --save_checkpoint_path ${save_checkpoint_path} \
         --transformer_layers 6 \
         --transformer_heads 6 \
         --hidden_dim 192 \
         --phase "finetune" \
         --sequence_length 7 \
         --action_pred_steps 3 \
         --future_steps 3 \
         --window_size 11 \
         --loss_action \
         --gripper_width \
         --reset_action_token \
         --warmup_epochs 3 \
         --libero_path ${libero_path} \
         --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
         --report_to_wandb \
         --use_text \
         --use_state \
         --use_depth \
         --model_size tiny \
         --seer_mini
