import torch
import torch.nn.functional as F

def generator_adv_losses(
    logits_fake: list[torch.Tensor], 
    fmaps_real: list[list[torch.Tensor]], 
    fmaps_fake: list[list[torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the standard unbounded Hinge Loss and Relative Feature Matching Loss 
    for the generator across the multi-scale outputs of the EnCodec Discriminator.
    """
    # 1. Unbounded Generator Hinge: -mean(D(fake))
    loss_hinge = sum([torch.mean(F.relu(1-logit)) for logit in logits_fake]) / len(logits_fake)

    # 2. Relative Feature Matching
    loss_fm = 0.0
    for fm_r_scale, fm_f_scale in zip(fmaps_real, fmaps_fake):
        for f_r, f_f in zip(fm_r_scale, fm_f_scale):
            f_r = f_r.detach()
            loss_fm += torch.mean(torch.abs(f_r - f_f)) / (torch.mean(torch.abs(f_r)) + 1e-8)
            
    loss_fm = loss_fm / len(fmaps_real)
    
    return loss_hinge, loss_fm

def compute_discriminator_hinge(
    logits_real: list[torch.Tensor], 
    logits_fake_detached: list[torch.Tensor]
) -> torch.Tensor:
    """
    Computes the Discriminator Hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake)).
    NOTE: logits_fake_detached MUST be computed from hat_x.detach() to prevent 
    gradients from flowing back into the generator during the discriminator's step.
    """
    loss_d = 0.0
    for l_real, l_fake in zip(logits_real, logits_fake_detached):
        loss_d += torch.mean(F.relu(1.0 - l_real)) + torch.mean(F.relu(1.0 + l_fake))
                  
    return loss_d / len(logits_real)

class DynamicSparseBalancer:
    """
    A dictionary-based Gradient Balancer designed for stochastic multi-discriminator routing.
    Tracks EMA norms independently so randomly skipped grids do not suffer from EMA decay.
    """
    def __init__(self, base_weights: dict, ema_decay: float = 0.999):
        """
        base_weights: dict mapping loss names to their target relative weights.
        Example: {'mssl': 10.0, 'huber': 1.0, 'hinge_1x': 1.0, 'fm_1x': 2.0, ...}
        """
        self.base_weights = base_weights
        self.ema_decay = ema_decay
        self.ema_norms = {} 

    def get_balanced_loss(self, active_losses: dict, active_outputs: dict, ganin_factor: float) -> torch.Tensor:
        """
        active_losses: dict of strictly the losses computed THIS step.
        active_outputs: dict mapping loss keys to the specific generator output tensor they stem from.
        ganin_factor: float (0.0 to 1.0) scaling the adversarial target weights during early epochs.
        """
        scaled_losses = []
        
        # 1. Compute shallow gradients and update EMAs exclusively for active losses
        for key, loss in active_losses.items():
            output = active_outputs[key]
            
            # Shallow backward pass to measure the gradient magnitude w.r.t the generator output
            grad = torch.autograd.grad(loss, output, retain_graph=True)[0]
            norm = torch.norm(grad, p=2).item()
            
            # Initialize or update the EMA for this specific key
            if key not in self.ema_norms:
                self.ema_norms[key] = norm
            else:
                self.ema_norms[key] = self.ema_decay * self.ema_norms[key] + (1.0 - self.ema_decay) * norm
                
        # 2. Compute dynamic target weights (Apply Ganin explicitly to adversarial losses)
        current_target_weights = {}
        for key in active_losses.keys():
            base_w = self.base_weights[key]
            
            if 'hinge' in key or 'fm' in key:
                current_target_weights[key] = base_w * ganin_factor
            else:
                current_target_weights[key] = base_w
                
        # 3. Calculate final scaled losses
        total_active_weight = sum(current_target_weights.values()) + 1e-8
        
        for key, loss in active_losses.items():
            weight_ratio = current_target_weights[key] / total_active_weight
            
            # The scaling mechanism: (Target Proportion) / (Current EMA Norm)
            scale = weight_ratio / (self.ema_norms[key] + 1e-8)
            scaled_losses.append(loss * scale)
            
        # Return the summed scalar loss ready for a deep .backward() call
        return sum(scaled_losses)