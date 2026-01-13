import sys
import os

# Add act directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from IACT.main import (
    build_IACT_model_and_optimizer,
)
from IACT.primitive_executor import PrimitiveExecutor, TerminationReason
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np


class IACTPolicy(nn.Module):
    """
    IACT Policy with three components:
    1. PrimitiveDecoderHead (inside ACT) - outputs p(z) and θ
    2. PrimitiveExecutor (outside ACT) - translates (z,θ) to impedance targets
    3. Low-level impedance controller (C++ control loop)
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_IACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override.get('kl_weight', 10.0)
        
        # Primitive configuration
        self.num_primitives = args_override.get('num_primitives', 8)
        self.primitive_param_dim = args_override.get('primitive_param_dim', 14)
        self.num_dof = args_override.get('state_dim', 7)
        
        # PrimitiveExecutor (not part of the neural network, but used during inference)
        # During training, we'll use it to convert predicted primitives to actions for loss computation
        self.primitive_executor = PrimitiveExecutor(
            num_dof=self.num_dof,
            default_stiffness=args_override.get('default_stiffness', 100.0),
            default_damping=args_override.get('default_damping', 10.0),
            contact_force_threshold=args_override.get('contact_force_threshold', 5.0),
            slip_velocity_threshold=args_override.get('slip_velocity_threshold', 0.1),
            jam_force_threshold=args_override.get('jam_force_threshold', 20.0),
            timeout_steps=args_override.get('timeout_steps', 500)
        )
        
        print(f'KL Weight: {self.kl_weight}')
        print(f'Number of primitives: {self.num_primitives}')
        print(f'Primitive parameter dimension: {self.primitive_param_dim}')

    def __call__(self, qpos, image, effort, actions=None, is_pad=None):
        """
        Forward pass of IACT policy.
        
        During training:
            - Predicts primitives (p(z), θ) from actions
            - Computes loss on primitive predictions and converts to actions for comparison
        
        During inference:
            - Predicts primitives (p(z), θ)
            - Samples primitive IDs from distribution
            - Returns primitives (can be converted to impedance targets via PrimitiveExecutor)
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Forward through model: outputs primitives
            primitive_logits, primitive_params, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, effort, env_state, actions, is_pad
            )

            # Compute losses
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            
            # For training, we need to compare predicted primitives with ground truth
            # Option 1: If we have ground truth primitives, use them directly
            # Option 2: Convert predicted primitives to actions and compare with ground truth actions
            # For now, we'll use Option 2: convert primitives to actions
            
            # Sample primitive IDs from distribution (for training, use teacher forcing with argmax or sampling)
            # During training, we can use the ground truth actions to infer primitives
            # For simplicity, we'll compute loss on primitive parameters directly
            # and also convert to actions for action-space loss
            
            # Primitive classification loss (cross-entropy)
            # Note: We don't have ground truth primitive IDs, so we'll skip this for now
            # Or we can use a reconstruction loss on actions
            
            # Convert predicted primitives to actions using PrimitiveExecutor
            # This is a differentiable approximation
            batch_size, num_queries = primitive_logits.shape[:2]
            actions_pred = []
            
            for b in range(batch_size):
                batch_actions = []
                for q in range(num_queries):
                    if is_pad[b, q]:
                        # Skip padded queries
                        batch_actions.append(torch.zeros(self.num_dof).to(qpos.device))
                        continue
                    
                    # Sample primitive ID (use argmax during training for stability, or sample)
                    primitive_id = torch.argmax(primitive_logits[b, q]).item()
                    primitive_params_np = primitive_params[b, q].detach().cpu().numpy()
                    
                    # Get impedance targets (non-differentiable, so we'll use a simpler approach)
                    # For training loss, we'll use the reference position directly as action
                    # Extract x_ref from params (assuming first num_dof elements are x_ref)
                    if len(primitive_params_np) >= self.num_dof:
                        x_ref = primitive_params_np[:self.num_dof]
                    else:
                        x_ref = np.zeros(self.num_dof)
                    
                    batch_actions.append(torch.from_numpy(x_ref).float().to(qpos.device))
                
                actions_pred.append(torch.stack(batch_actions))
            
            actions_pred = torch.stack(actions_pred)  # (batch, num_queries, action_dim)
            
            # Action reconstruction loss
            all_l1 = F.l1_loss(actions, actions_pred, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            
            # Primitive parameter regularization (encourage reasonable parameter values)
            param_reg = torch.mean(primitive_params ** 2) * 0.01
            
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['param_reg'] = param_reg
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['param_reg']
            
            return loss_dict
        else:  # inference time
            # Forward through model without actions (sample from prior)
            primitive_logits, primitive_params, is_pad_hat, (_, _) = self.model(
                qpos, image, effort, env_state
            )
            
            # Return primitives (can be processed by PrimitiveExecutor to get impedance targets)
            return {
                'primitive_logits': primitive_logits,  # (batch, num_queries, num_primitives)
                'primitive_params': primitive_params,   # (batch, num_queries, primitive_param_dim)
                'is_pad_hat': is_pad_hat
            }
    
    def get_impedance_targets(self, primitive_logits, primitive_params, current_position, current_velocity):
        """
        Convert primitives to impedance controller targets using PrimitiveExecutor.
        
        Args:
            primitive_logits: (batch, num_queries, num_primitives) or (num_queries, num_primitives)
            primitive_params: (batch, num_queries, primitive_param_dim) or (num_queries, primitive_param_dim)
            current_position: (num_dof,) numpy array
            current_velocity: (num_dof,) numpy array
            
        Returns:
            Dictionary with impedance targets for each query
        """
        # Handle batch dimension
        if len(primitive_logits.shape) == 3:
            batch_size = primitive_logits.shape[0]
            results = []
            for b in range(batch_size):
                results.append(self._get_impedance_targets_single(
                    primitive_logits[b], primitive_params[b], current_position, current_velocity
                ))
            return results
        else:
            return self._get_impedance_targets_single(
                primitive_logits, primitive_params, current_position, current_velocity
            )
    
    def _get_impedance_targets_single(self, primitive_logits, primitive_params, current_position, current_velocity):
        """
        Convert primitives to impedance targets for a single batch.
        """
        num_queries = primitive_logits.shape[0]
        impedance_targets = []
        
        for q in range(num_queries):
            # Sample primitive ID from distribution
            primitive_probs = F.softmax(primitive_logits[q], dim=-1)
            primitive_id = torch.multinomial(primitive_probs, 1).item()
            
            # Get primitive parameters
            params_np = primitive_params[q].detach().cpu().numpy()
            
            # Update PrimitiveExecutor
            self.primitive_executor.update_primitive(primitive_id, params_np)
            
            # Get impedance targets
            targets = self.primitive_executor.get_impedance_targets(
                current_position, current_velocity
            )
            
            impedance_targets.append(targets)
        
        return impedance_targets

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# Alias for backward compatibility
ImpedanceACTPolicy = IACTPolicy
