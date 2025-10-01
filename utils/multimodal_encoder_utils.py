import torch

def save_seer_backbone(original_seer_model, save_path="seer_backbone.pth"):
    """
    Extracts relevant components from the original Seer model
    and saves them to a checkpoint for reuse.
    """
    checkpoint = {}

    # === text ===
    checkpoint["text_projector"] = original_seer_model.text_projector.state_dict()

    # === state encoders ===
    checkpoint["arm_state_encoder"] = original_seer_model.arm_state_encoder.state_dict()
    checkpoint["gripper_state_encoder"] = original_seer_model.gripper_state_encoder.state_dict()
    checkpoint["state_projector"] = original_seer_model.state_projector.state_dict()

    # === action encoders ===
    checkpoint["action_pose_encoder"] = original_seer_model.action_pose_encoder.state_dict()
    checkpoint["action_gripper_position_encoder"] = original_seer_model.action_gripper_position_encoder.state_dict()
    checkpoint["action_projector"] = original_seer_model.action_projector.state_dict()

    # === vision encoder (frozen MAE) ===
    checkpoint["vision_encoder"] = original_seer_model.vision_encoder.state_dict()

    # === resampler & projectors ===
    checkpoint["perceiver_resampler"] = original_seer_model.perceiver_resampler.state_dict()
    checkpoint["image_primary_projector"] = original_seer_model.image_primary_projector.state_dict()
    checkpoint["cls_token_primary_projector"] = original_seer_model.cls_token_primary_projector.state_dict()
    checkpoint["image_wrist_projector"] = original_seer_model.image_wrist_projector.state_dict()
    checkpoint["cls_token_wrist_projector"] = original_seer_model.cls_token_wrist_projector.state_dict()

    # === special tokens ===
    if hasattr(original_seer_model, "action_pred_token"):
        checkpoint["action_pred_token"] = original_seer_model.action_pred_token.detach().cpu()
    if hasattr(original_seer_model, "obs_tokens"):
        checkpoint["obs_tokens"] = original_seer_model.obs_tokens.detach().cpu()

    # === embedding + mask ===
    checkpoint["embedding_layer_norm"] = original_seer_model.embedding_layer_norm.state_dict()
    checkpoint["attention_mask"] = original_seer_model.attention_mask.detach().cpu()
    checkpoint["transformer_backbone_position_embedding"] = (
        original_seer_model.transformer_backbone_position_embedding.detach().cpu()
    )

    # === transformer backbone (GPT2) ===
    checkpoint["transformer_backbone"] = original_seer_model.transformer_backbone.state_dict()

    torch.save(checkpoint, save_path)
    print(f"✔ Saved Seer backbone to {save_path}")


def load_seer_backbone(diffusion_seer_model, load_path="seer_backbone.pth"):
    """
    Loads backbone weights from saved Seer checkpoint
    into the diffusion-Seer model.
    """
    checkpoint = torch.load(load_path, map_location="cpu")

    diffusion_seer_model.text_projector.load_state_dict(checkpoint["text_projector"])

    diffusion_seer_model.arm_state_encoder.load_state_dict(checkpoint["arm_state_encoder"])
    diffusion_seer_model.gripper_state_encoder.load_state_dict(checkpoint["gripper_state_encoder"])
    diffusion_seer_model.state_projector.load_state_dict(checkpoint["state_projector"])

    diffusion_seer_model.action_pose_encoder.load_state_dict(checkpoint["action_pose_encoder"])
    diffusion_seer_model.action_gripper_position_encoder.load_state_dict(checkpoint["action_gripper_position_encoder"])
    diffusion_seer_model.action_projector.load_state_dict(checkpoint["action_projector"])

    diffusion_seer_model.vision_encoder.load_state_dict(checkpoint["vision_encoder"])

    diffusion_seer_model.perceiver_resampler.load_state_dict(checkpoint["perceiver_resampler"])
    diffusion_seer_model.image_primary_projector.load_state_dict(checkpoint["image_primary_projector"])
    diffusion_seer_model.cls_token_primary_projector.load_state_dict(checkpoint["cls_token_primary_projector"])
    diffusion_seer_model.image_wrist_projector.load_state_dict(checkpoint["image_wrist_projector"])
    diffusion_seer_model.cls_token_wrist_projector.load_state_dict(checkpoint["cls_token_wrist_projector"])

    if "action_pred_token" in checkpoint:
        diffusion_seer_model.action_pred_token.data.copy_(checkpoint["action_pred_token"])
    if "obs_tokens" in checkpoint:
        diffusion_seer_model.obs_tokens.data.copy_(checkpoint["obs_tokens"])

    diffusion_seer_model.embedding_layer_norm.load_state_dict(checkpoint["embedding_layer_norm"])

    # frozen/non-trainable stuff
    diffusion_seer_model.attention_mask.data.copy_(checkpoint["attention_mask"])
    diffusion_seer_model.transformer_backbone_position_embedding.data.copy_(
        checkpoint["transformer_backbone_position_embedding"]
    )

    diffusion_seer_model.transformer_backbone.load_state_dict(checkpoint["transformer_backbone"])

    print(f"✔ Loaded Seer backbone from {load_path}")
    return diffusion_seer_model
