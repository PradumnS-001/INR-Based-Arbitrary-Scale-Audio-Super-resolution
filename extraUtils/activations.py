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
    
class getActivation(nn.Module):
    
    def __init__(self, act:str | None='relu'):
        super().__init__()
        self.activationFunction = lambda x : x
        act = act.lower()
        
        if act=='gelu': self.activationFunction = nn.GELU(approximate='tanh')
        elif act=='lrelu': self.activationFunction = nn.LeakyReLU(negative_slope=0.2)
        elif act=='elu': self.activationFunction = nn.ELU()
        elif act=='prelu': self.activationFunction = nn.PReLU()
        elif act=='sigmoid': self.activationFunction = nn.Sigmoid()
        elif act=='tanh': self.activationFunction = nn.Tanh()
        elif act=='relu': self.activationFunction = nn.ReLU()
        elif act=='silu': self.activationFunction = nn.SiLU()
        elif act=='softplus': self.activationFunction = nn.Softplus()
        elif act=='snake': self.activationFunction = Snake()
        
    def forward(self, x)->torch.Tensor:
        return self.activationFunction(x)