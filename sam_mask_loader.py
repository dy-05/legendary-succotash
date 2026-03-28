# utils/sam_mask_loader.py
import numpy as np
import torch
import torch.nn.functional as F
import os
from pathlib import Path

def load_sam_mask_for_frame(img_path: str, 
                           mask_dir: str,
                           target_size: tuple = (336, 336),
                           orig_img_size: tuple = (1080, 1280)) -> torch.Tensor:
    """
    Load SAM2 mask (npy) and align with ProxyCLIP input geometry.
    
    Args:
        img_path: Path to input image (e.g., '.../0100.png')
        mask_dir: Directory containing SAM2 masks (e.g., '.../sam2_masks/')
        target_size: ProxyCLIP input size (H, W)
        orig_img_size: Original UAVid frame size (H, W)
    
    Returns:
        sam_masks: torch.Tensor [1, N, H_target, W_target] binary masks
                   or None if mask file not found
    """
    # 1. Derive mask filename from image path
    img_name = Path(img_path).name
    # Try parsing UAVid format: seq21_000000_0_1080_0_1280.png -> 000000
    parts = img_name.split('_')
    if len(parts) >= 2 and parts[0].startswith('seq'):
        frame_id = parts[1]
    else:
        # Fallback to simple filename
        frame_id = img_name.split('.')[0]
        
    mask_path = os.path.join(mask_dir, f"mask_{frame_id}.npy")
    
    if not os.path.exists(mask_path):
        # print(f"DEBUG: Mask not found at {mask_path}")
        return None
    
    # 2. Load SAM2 mask (instance ID map)
    mask_id_map = np.load(mask_path)  # [H_orig, W_orig], dtype=uint16
    
    # Validate format
    if mask_id_map.shape != orig_img_size:
        raise ValueError(f"SAM2 mask shape {mask_id_map.shape} != expected {orig_img_size}")
    
    # 3. Convert to binary masks [N, H_orig, W_orig]
    instance_ids = np.unique(mask_id_map)
    instance_ids = instance_ids[instance_ids != 0]  # Skip background (ID=0)
    
    if len(instance_ids) == 0:
        return None
    
    binary_masks = []
    for inst_id in instance_ids:
        binary_mask = (mask_id_map == inst_id).astype(np.float32)  # [H_orig, W_orig]
        binary_masks.append(binary_mask)
    
    masks_np = np.stack(binary_masks)  # [N, H_orig, W_orig]
    
    # 4. Align geometry: resize to target_size (critical for coordinate alignment)
    masks_tensor = torch.from_numpy(masks_np).float()  # [N, H_orig, W_orig]
    masks_tensor = masks_tensor.unsqueeze(1)  # [N, 1, H_orig, W_orig]
    masks_tensor = F.interpolate(
        masks_tensor,
        size=target_size,
        mode='nearest'  # Preserve binary property
    ).squeeze(1)  # [N, H_target, W_target]
    
    # 5. Convert to batch format [B=1, N, H, W]
    masks_tensor = masks_tensor.unsqueeze(0)  # [1, N, H_target, W_target]
    
    return masks_tensor