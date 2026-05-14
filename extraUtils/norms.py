import torch

def AdaIn(content:torch.Tensor, style:torch.Tensor)->torch.Tensor:
    
    """
    Do adaptive instance normalization
    content: [B, C, L] or [B, C, H, W]
    style:   [B, C, L] or [B, C, H, W]
    """
    
    if len(content.shape) == 4:
        sd = (2,3)
    elif len(content.shape) == 3:
        sd = (2,)
    
    meanc = content.mean(dim=sd, keepdim=True)
    stdc = torch.sqrt(content.var(dim=sd, keepdim=True) + 1e-8)
    
    content = (content - meanc) / stdc
    
    means = style.mean(dim=sd, keepdim=True)
    stds = torch.sqrt(style.var(dim=sd, keepdim=True) + 1e-8)
    
    return content * stds + means