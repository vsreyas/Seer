#!/bin/bash

### ===== CONFIGURE THESE PATHS ===== ###
save_checkpoint_path="checkpoints/"
root_dir="/home/venky/Projects/Seer/data/LIBERO"
libero_path="/home/venky/Projects/Seer/LIBERO"
finetune_from_pretrained_ckpt="/home/venky/Projects/Seer/checkpoints/libero_pretrain_depth_clip_resnet_action_only/19.pth"
### ================================== ###

node=1
node_num=1

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 8 \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 10000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 3 \
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
