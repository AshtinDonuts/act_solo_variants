"""
IACT_B Policy: Dual-head ACT with primitive-aware execution.

Pipeline:
ACT → (joint chunk, primitive tokens) → PrimitiveExecutor → executed joint commands
"""
import sys
import os

# Add act directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from IACT_B.main import build_IACT_B_model_and_optimizer
from IACT_B.primitive_executor import PrimitiveExecutor, EventType
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np


class IACT_B_Policy(nn.Module):
    """
    IACT_B Policy with dual heads:
    1. Joint reference head (q_ref) - vanilla ACT output
    2. Primitive head (z, θ) - primitive logits and parameters
    
    The PrimitiveExecutor modifies how q_ref is executed based on the primitive.
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_IACT_B_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override.get('kl_weight', 10.0)
        
        # Primitive configuration
        self.num_primitives = args_override.get('num_primitives', 6)
        self.primitive_param_dim = args_override.get('primitive_param_dim', 8)
        self.num_dof = args_override.get('state_dim', 7)
        
        # Loss weights
        self.primitive_loss_weight = args_override.get('primitive_loss_weight', 1.0)
        self.param_loss_weight = args_override.get('param_loss_weight', 0.1)
        self.event_loss_weight = args_override.get('event_loss_weight', 0.5)
        
        # PrimitiveExecutor (used during inference)
        self.primitive_executor = PrimitiveExecutor(
            num_dof=self.num_dof,
            default_stiffness=args_override.get('default_stiffness', 100.0),
            default_damping=args_override.get('default_damping', 10.0),
            contact_torque_threshold=args_override.get('contact_torque_threshold', 5.0),
            slip_velocity_threshold=args_override.get('slip_velocity_threshold', 0.1),
            jam_force_threshold=args_override.get('jam_force_threshold', 20.0),
            timeout_steps=args_override.get('timeout_steps', 500)
        )
        
        print(f'KL Weight: {self.kl_weight}')
        print(f'Number of primitives: {self.num_primitives}')
        print(f'Primitive parameter dimension: {self.primitive_param_dim}')
        print(f'Primitive loss weight: {self.primitive_loss_weight}')
        print(f'Param loss weight: {self.param_loss_weight}')

    def __call__(self, qpos, image, effort, actions=None, is_pad=None, 
                 primitive_labels=None, primitive_params=None, event_labels=None):
        """
        Forward pass of IACT_B policy.
        
        During training:
            - Predicts joint references (q_ref) and primitives (z, θ)
            - Computes losses on both heads
            - Uses ground truth primitive labels if provided
        
        During inference:
            - Predicts joint references and primitives
            - Returns both for PrimitiveExecutor to process
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # Forward through model: outputs both q_ref and primitives
            # We rename a_hat to q_ref because this is a joint equilibrium \
            # within the context of Impedance control \
            # # though... we are aiminig towards Cartesian Impedance, so this is just \
            # though... we are aiming towards Cartesian Impedance, so this is just \
            # an intermediate processing step
            q_ref, primitive_logits, primitive_params_pred, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, effort, env_state, actions, is_pad
            )

            # Compute losses
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            
            # L_joint: L1/L2 on q_ref (vanilla ACT loss)
            all_l1 = F.l1_loss(actions, q_ref, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            
            # L_prim: cross-entropy on z (primitive classification)
            # GT labels generated from the data_labeling.py
            if primitive_labels is not None:
                # Ground truth primitive labels provided
                primitive_labels = primitive_labels[:, :self.model.num_queries]
                # Reshape for cross-entropy: (batch * num_queries, num_primitives)
                batch_size, num_queries = primitive_logits.shape[:2]
                primitive_logits_flat = primitive_logits.view(-1, self.num_primitives)
                primitive_labels_flat = primitive_labels.view(-1).long()
                is_pad_flat = is_pad.view(-1)
                
                # Only compute loss on non-padded queries
                valid_mask = ~is_pad_flat
                if valid_mask.sum() > 0:
                    prim_loss = F.cross_entropy(
                        primitive_logits_flat[valid_mask],
                        primitive_labels_flat[valid_mask],
                        reduction='mean'
                    )
                else:
                    prim_loss = torch.tensor(0.0).to(qpos.device)
                loss_dict['primitive'] = prim_loss
            else:
                # No ground truth labels: skip primitive loss or use weak supervision
                loss_dict['primitive'] = torch.tensor(0.0).to(qpos.device)
            
            # L_param: L2/Huber on θ (primitive parameters)
            if primitive_params is not None:
                primitive_params = primitive_params[:, :self.model.num_queries]
                param_loss = F.mse_loss(
                    primitive_params_pred * ~is_pad.unsqueeze(-1),
                    primitive_params * ~is_pad.unsqueeze(-1),
                    reduction='mean'
                )
                loss_dict['param'] = param_loss
            else:
                # Regularization on predicted parameters
                param_loss = torch.mean(primitive_params_pred ** 2) * 0.01
                loss_dict['param'] = param_loss
            
            # L_event: predict contact/jam/slip events as auxiliary task (optional)
            # This helps prevent primitives from collapsing
            if event_labels is not None:
                # Could add an event prediction head, but for now we skip
                loss_dict['event'] = torch.tensor(0.0).to(qpos.device)
            else:
                loss_dict['event'] = torch.tensor(0.0).to(qpos.device)
            
            # Total loss
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = (
                loss_dict['l1'] +
                loss_dict['kl'] * self.kl_weight +
                loss_dict['primitive'] * self.primitive_loss_weight +
                loss_dict['param'] * self.param_loss_weight +
                loss_dict['event'] * self.event_loss_weight
            )
            
            return loss_dict
        else:  # inference time
            # Forward through model without actions (sample from prior)
            q_ref, primitive_logits, primitive_params, is_pad_hat, (_, _) = self.model(
                qpos, image, effort, env_state
            )
            
            # Return both joint references and primitives
            return {
                'q_ref': q_ref,  # (batch, num_queries, state_dim)
                'primitive_logits': primitive_logits,  # (batch, num_queries, num_primitives)
                'primitive_params': primitive_params,  # (batch, num_queries, primitive_param_dim)
                'is_pad_hat': is_pad_hat
            }
    
    def execute_with_primitives(self, q_ref, primitive_logits, primitive_params,
                               q_current, qdot_current, tau_meas=None):
        """
        Execute joint references with primitive-aware modifications.
        
        Args:
            q_ref: Joint references from model (num_queries, num_dof) or (batch, num_queries, num_dof)
            primitive_logits: Primitive logits (num_queries, num_primitives) or (batch, num_queries, num_primitives)
            primitive_params: Primitive parameters (num_queries, param_dim) or (batch, num_queries, param_dim)
            q_current: Current joint positions (num_dof,)
            qdot_current: Current joint velocities (num_dof,)
            tau_meas: Measured torques (num_dof,) or None
            
        Returns:
            Dictionary with executed commands and execution info
        """
        # Handle batch dimension
        if len(q_ref.shape) == 3:
            batch_size = q_ref.shape[0]
            results = []
            for b in range(batch_size):
                results.append(self._execute_with_primitives_single(
                    q_ref[b], primitive_logits[b], primitive_params[b],
                    q_current, qdot_current, tau_meas
                ))
            return results
        else:
            return self._execute_with_primitives_single(
                q_ref, primitive_logits, primitive_params,
                q_current, qdot_current, tau_meas
            )
    
    def _execute_with_primitives_single(self, q_ref, primitive_logits, primitive_params,
                                       q_current, qdot_current, tau_meas):
        """
        Execute for a single batch.
        """
        num_queries = q_ref.shape[0]
        executed_commands = []
        
        # Use the first query's primitive (or take argmax)
        # In practice, you might want to use temporal aggregation
        primitive_probs = F.softmax(primitive_logits[0], dim=-1)
        # primitive_id = torch.argmax(primitive_probs).item()
        primitive_id = torch.argmax(primitive_probs[0]).item()
        
        # Get primitive parameters
        # params_np = primitive_params[0].detach().cpu().numpy()
        params_np = primitive_params[0][0].detach().cpu().numpy()
        
        # Update PrimitiveExecutor
        self.primitive_executor.update_primitive(primitive_id, params_np)
        
        # Execute each query step with primitive-aware modifications
        for q in range(num_queries):
            q_ref_q = q_ref[q].detach().cpu().numpy()
            result = self.primitive_executor.execute_joint_reference(
                q_ref_q, q_current, qdot_current, tau_meas
            )
            executed_commands.append(result)
        
        return executed_commands

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
