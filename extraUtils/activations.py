import torch
from torch import nn

class Snake(nn.Module):
    
    """
    Snake activation class
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self,x):
        
        alpha = torch.where(self.alpha.abs() < 1e-6, 
                                1e-6 * self.alpha.sgn(), 
                                self.alpha)
        alpha = torch.where(alpha == 0, 1e-7, alpha)
        
        exact_res = x + (1 - torch.cos(2 * alpha * x)) / (2 * alpha)
        taylor_res = x + self.alpha * (x**2)
        
        return torch.where(self.alpha.abs() < 1e-6, taylor_res, exact_res)