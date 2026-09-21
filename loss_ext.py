import torch
import torch.nn.functional as F
from encodec import EncodecModel

def generator_hinge_loss(logits_fake: list[torch.Tensor]) -> torch.Tensor:
    """
    Computes the stable Hinge loss for the generator.
    Expects a list of logits if using a Multi-Scale Discriminator.
    """
    loss_g = 0.0
    for l_fake in logits_fake:
        loss_g += -torch.mean(l_fake)
    
    return loss_g / len(logits_fake)


def discriminator_hinge_loss(
    logits_real: list[torch.Tensor], 
    logits_fake_detached: list[torch.Tensor]
) -> torch.Tensor:
    """
    Computes the stable Hinge loss for the discriminator.
    NOTE: logits_fake_detached MUST be computed from pred.detach() 
    to prevent gradients from flowing back into the generator.
    """
    loss_d = 0.0
    for l_real, l_fake in zip(logits_real, logits_fake_detached):
        # max(0, 1 - D(real)) + max(0, 1 + D(fake))
        loss_d += torch.mean(F.relu(1.0 - l_real)) + torch.mean(F.relu(1.0 + l_fake))
                  
    return loss_d / len(logits_real)

class EncodecIntermediatePerceptualLoss(torch.nn.Module):
    """
    Perceptual loss using intermediate feature maps from EnCodec's encoder blocks.
    Uses forward hooks to cleanly extract feature maps without breaking the computational graph.
    """
    def __init__(self, target_sr: int, device: str = 'cuda'):
        super().__init__()
        
        if target_sr == 24000:
            self.model = EncodecModel.encodec_model_24khz()
        elif target_sr == 48000:
            self.model = EncodecModel.encodec_model_48khz()
        else:
            raise ValueError(f"EnCodec pretrained models only support 24000 or 48000 Hz. Received: {target_sr}")
            
        self.model.to(device)
        self.model.eval()
        self.target_sr = target_sr
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.LSTM):
                module.train()
        
        # Freeze the whole model explicitly
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Storage dictionaries for our hooks
        self.real_features = {}
        self.fake_features = {}
        self.hooks = []
        
        # Register hooks on each major encoder block stage
        # self.model.encoder.model contains the underlying sequential blocks
        for idx, layer in enumerate(self.model.encoder.model):
            # We use hook factories to avoid lambda scope retention bugs
            self.hooks.append(layer.register_forward_hook(self._get_hook(idx)))

    def _get_hook(self, idx: int):
        def hook(module, input, output):
            # We store the features. The hook will be triggered twice per forward step:
            # Once during the real audio pass, and once during the fake audio pass.
            if self._processing_real:
                # Detach real features since we don't backprop through the target
                self.real_features[idx] = output.detach()
            else:
                # Keep fake features attached to the graph for gradient calculation
                # We clone to safely isolate it from any following in-place layers
                self.fake_features[idx] = output.clone()
        return hook

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # EnCodec 48kHz model mandates stereo format (C=2)
        if self.target_sr == 48000:
            if x.shape[1] == 1:
                x = x.repeat(1, 2, 1)
            if x_hat.shape[1] == 1:
                x_hat = x_hat.repeat(1, 2, 1)
        
        # Clear out previous iterations
        self.real_features.clear()
        self.fake_features.clear()
        
        # 1. Process target audio (Real)
        self._processing_real = True
        with torch.no_grad():
            # Running the full encoder safely initializes all internal shapes
            _ = self.model.encoder(x)
            
        # 2. Process generated audio (Fake)
        self._processing_real = False
        # Do NOT use torch.no_grad() here; we need gradients for x_hat!
        _ = self.model.encoder(x_hat)
        
        # 3. Calculate feature matching loss across all collected block layers
        loss = 0.0
        num_layers = len(self.fake_features)
        
        for idx in self.fake_features.keys():
            loss += F.l1_loss(self.fake_features[idx], self.real_features[idx])
            
        return loss / num_layers

    def remove_hooks(self):
        """Call this if you ever need to safely clean up the model hooks from memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

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
            safe_norm = max(self.ema_norms[key], 1e-4)
            scale = weight_ratio / safe_norm
            scaled_losses.append(loss * scale)
            
        # Return the summed scalar loss ready for a deep .backward() call
        return sum(scaled_losses)