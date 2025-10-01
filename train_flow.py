import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.base_decoder_diffusion import SeerAgentFlow
from models.diffusion_flow import GoalGaussianDiffusionSeerFlow, TrainerFlow
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, get_droid_dataset, get_libero_pretrain_dataset, get_libero_finetune_dataset, get_real_finetune_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  
from accelerate import Accelerator

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return total_params, trainable_params
@record
def main(args):
    # accelerator = Accelerator(mixed_precision="fp16" if args.precision == "fp16" else "no")
    # device = accelerator.device
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps

    args.run_name = args.run_name.replace(
        "Seer", f"Seer_ptbs{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}"
    )
    print("run_name:", args.run_name)
    print("training batch size:", ptbs)

    seer_model = SeerAgentFlow(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
    )
    # Freeze modules
    seer_model.clip_model.requires_grad_(False)
    seer_model.vision_encoder.requires_grad_(False)
    # Precision casting
    if args.precision == "bf16":
        seer_model = seer_model.bfloat16()
    elif args.precision == "fp16":
        seer_model = seer_model.half()
    elif args.precision == "fp32":
        seer_model = seer_model.float()
        if 'vision_encoder' in args.bf16_module:
            seer_model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            seer_model.transformer_backbone.bfloat16()
    
    total_params, trainable_params = count_parameters(seer_model)
    if args.rank == 0:
        print("total_params: {} M".format(total_params/1024/1024))
        print("trainable_params: {} M".format(trainable_params/1024/1024))
    device_id = args.rank % torch.cuda.device_count()
    seer_model = seer_model.to(device_id)
    seer_model._init_model_type()
    if os.path.exists("/data/user_data/sreyasv/seer_libero/checkpoints/transformer_pretrained_seer_libero.pth"):
        from utils.multimodal_encoder_utils import load_seer_backbone
        load_seer_backbone(seer_model, "/data/user_data/sreyasv/seer_libero/checkpoints/transformer_pretrained_seer_libero.pth")


    # ===== Load pretrained Seer weights if requested =====
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")    
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")

        if args.reset_action_token and "module.action_pred_token" in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"]["module.action_pred_token"]
        if args.reset_obs_token and "module.obs_tokens" in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"]["module.obs_tokens"]
        if args.reset_mask_token and "module.mask_token" in checkpoint["model_state_dict"]:
            del checkpoint["model_state_dict"]["module.mask_token"]

        # Resize position embedding if mismatch
        ckpt_pe = checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"]
        if ckpt_pe.shape != seer_model.transformer_backbone_position_embedding.shape:
            checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"] = ckpt_pe[
                :, :args.sequence_length, :, :
            ]

        if args.rank == 0:
            print("loading pretrained weights :", checkpoint["model_state_dict"].keys())

        seer_model.load_state_dict(checkpoint["model_state_dict"], strict=False)


    # ===== Wrap with diffusion =====
    diffusion_model = GoalGaussianDiffusionSeerFlow(model=seer_model, args=args)
    # ===== Dataset Init (same as before) =====
    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, seer_model.image_processor, clip, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "droid":
        calvin_dataset = get_droid_dataset(args, seer_model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_pretrain":
        calvin_dataset = get_libero_pretrain_dataset(args, seer_model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_finetune":
        calvin_dataset = get_libero_finetune_dataset(args, seer_model.image_processor, clip, epoch=0)
    elif args.finetune_type == "real":
        calvin_dataset = get_real_finetune_dataset(args, seer_model.image_processor, clip, epoch=0)
    elif args.finetune_type == "oxe":
        calvin_dataset = get_oxe_dataset(args, seer_model.image_processor, clip, epoch=0)

    train_set = calvin_dataset.dataloader
    valid_set = [0]  # unused
    # ===== Optimizer + LR scheduler =====
    diffusion_model = diffusion_model.to(device_id)
    print("checking device id again: ",device_id)
    ddp_model = DDP(diffusion_model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir,exist_ok=True)
    trainer = TrainerFlow(
        diffusion_model=ddp_model,
        args=args,
        opt=optimizer,
        lr_scheduler=lr_scheduler,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        gradient_accumulate_every=args.gradient_accumulation_steps,
        results_folder=ckpt_dir
    )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        resume_from_epoch = trainer.load(milestone=args.resume_from_checkpoint)
        
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    # ===== Training Loop =====
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        trainer.train_one_epoch(epoch, wandb_logger=wandb if args.report_to_wandb else None, args=args)

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
        