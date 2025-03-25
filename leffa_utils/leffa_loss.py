import torch
import torch.nn.functional as F

def compute_attention_map(Q, K, tau=2.0):
    """
    Compute scaled dot-product attention map with temperature scaling.
    Args:
        Q: Query tensor of shape (B, H, N, D)
        K: Key tensor of shape (B, H, N, D)
        tau: Temperature coefficient
    Returns:
        A: Attention map of shape (B, H, N, N)
    """
    d = Q.shape[-1]  # Feature dimension
    A = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5 / tau)
    A = F.softmax(A, dim=-1)
    return A

def compute_flow_field(A, coords):
    """
    Compute the flow field by multiplying the attention map with the coordinate map.
    Args:
        A: Averaged attention map (B, N, N)
        coords: Coordinate map (B, N, 2)
    Returns:
        Flow field (B, N, 2)
    """
    F_l = torch.matmul(A, coords)  # Weighted sum of coordinates
    return F_l

def upsample_flow_field(F_l, H, W):
    """
    Upsample the flow field to match the target image resolution.
    Args:
        F_l: Flow field at latent resolution (B, h, w, 2)
        H, W: Target image resolution
    Returns:
        Upsampled flow field (B, H, W, 2)
    """
    B, h, w, _ = F_l.shape
    F_l = F_l.permute(0, 3, 1, 2)  # (B, 2, h, w)
    F_l_up = F.interpolate(F_l, size=(H, W), mode='bilinear', align_corners=False)
    F_l_up = F_l_up.permute(0, 2, 3, 1)  # Back to (B, H, W, 2)
    return F_l_up

def warp_image(I_ref, F_l_up):
    """
    Warp the reference image using the upsampled flow field.
    Args:
        I_ref: Reference image (B, C, H, W)
        F_l_up: Flow field (B, H, W, 2)
    Returns:
        Warped image (B, C, H, W)
    """
    B, C, H, W = I_ref.shape
    grid = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    grid = torch.stack(grid, dim=-1).to(I_ref.device)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    warped_grid = grid + F_l_up  # Add the flow field
    warped_image = F.grid_sample(I_ref, warped_grid, align_corners=False)
    return warped_image

def leffa_loss(I_tgt, I_ref, A, coords, mask, H, W, lambda_leffa=1e-3):
    """
    Compute the Leffa Loss.
    Args:
        I_tgt: Target image (B, C, H, W)
        I_ref: Reference image (B, C, H, W)
        A: Attention map (B, N, N)
        coords: Normalized coordinate map (B, N, 2)
        mask: Binary mask for valid regions (B, C, H, W)
        H, W: Target image dimensions
        lambda_leffa: Weight for Leffa loss
    Returns:
        Leffa loss value
    """
    # Compute flow field
    F_l = compute_flow_field(A, coords)  # (B, N, 2)
    F_l = F_l.view(I_tgt.shape[0], H // 32, W // 32, 2)  # Reshape to (B, h, w, 2)
    
    # Upsample to target resolution
    F_l_up = upsample_flow_field(F_l, H, W)
    
    # Warp the reference image
    I_warp = warp_image(I_ref, F_l_up)
    
    # Compute L2 loss only on masked regions
    loss = torch.norm((I_tgt * mask - I_warp * mask), p=2, dim=(1, 2, 3)).mean()
    return lambda_leffa * loss
