#!/bin/bash

### NEED TO CHANGE ###
save_checkpoint_path="/data/user_data/sreyasv/seer_libero"
root_dir="/data/hf_cache/datasets/LIBERO"
vit_checkpoint_path="/data/user_data/sreyasv/seer_d/checkpoints/mae_pretrain_vit_base.pth"
libero_path="/home/sreyasv/Projects/Seer/LIBERO"
finetune_from_pretrained_ckpt="/data/user_data/sreyasv/seer_libero/checkpoints/seer_finetuned_libero.pth"
### NEED TO CHANGE ###
calvin_dataset_path="~/calvin/dataset/task_D_D"

node=8
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train_flow.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 1 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name libero_finetune \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_action \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 25 \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path ${libero_path} \
    # --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    # --report_to_wandb \
    # --reset_action_token \
    # --reset_obs_token \