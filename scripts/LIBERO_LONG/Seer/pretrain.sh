#!/bin/bash

### NEED TO CHANGE ###
save_checkpoint_path="/data/user_data/sreyasv/seer_libero"
root_dir="/data/hf_cache/datasets/LIBERO"
vit_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/mae_pretrain_vit_base.pth"
libero_path="/home/sreyasv/Projects/Seer/LIBERO"
### NEED TO CHANGE ###
calvin_dataset_path="~/calvin/dataset/task_D_D"
mkdir -p ${save_checkpoint_path}

node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 8 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 30 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_pretrain \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_pretrain \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "pretrain" \
    --obs_pred \
    --sequence_length 11 \
    --action_pred_steps 3 \
    --future_steps 3 \
    --atten_goal 4 \
    --window_size 11 \
    --loss_image \
    --loss_action \
    --gripper_width \
    --atten_only_obs \
    --atten_goal_state \
    --mask_l_obs_ratio 0.5 \
    --warmup_epochs 1 \
    --libero_path ${libero_path} \
    # --report_to_wandb \

    #Default worker is 16