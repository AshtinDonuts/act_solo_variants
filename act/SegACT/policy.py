"""
SegACT Policy: ACT with pose-painted segmentation inputs.

Pipeline:
1. Load segmentation masks (or generate from RGB images)
2. Paint pose overlays (object pose, EEF pose, contact normal)
3. Feed pose-painted segmentation to ACT model
"""
import sys
import os

# Add act directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SegACT.main import build_SegACT_model_and_optimizer
from SegACT.pose_painter import paint_pose_on_segmentation
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np


class SegACT_Policy(nn.Module):
    """
    SegACT Policy: ACT with pose-painted segmentation inputs.
    
    The key difference from vanilla ACT:
    - Input images are segmentation masks with pose overlays painted on them
    - Pose information (object pose, EEF pose) is made explicit through visual overlays
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_SegACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override.get('kl_weight', 10.0)
        
        # SegACT specific parameters
        self.ablation_mode = args_override.get('ablation_mode', 'full')
        self.pose_noise_scale = args_override.get('pose_noise_scale', 0.0)
        self.axis_length = args_override.get('axis_length', 0.05)
        self.pose_thickness = args_override.get('pose_thickness', 2)
        
        # Camera intrinsics (can be provided or use defaults)
        self.camera_intrinsics = args_override.get('camera_intrinsics', None)
        self.camera_extrinsics = args_override.get('camera_extrinsics', None)
        
        print(f'KL Weight: {self.kl_weight}')
        print(f'Ablation mode: {self.ablation_mode}')
        print(f'Pose noise scale: {self.pose_noise_scale}')

    def _get_segmentation_mask(self, image):
        """
        Get segmentation mask from RGB image.
        
        For now, we'll use a simple approach:
        - If segmentation is provided, use it
        - Otherwise, generate a simple segmentation from RGB (placeholder)
        
        In practice, this should use a proper segmentation model or load pre-computed masks.
        """
        # TODO: Implement proper segmentation
        # For now, return a simple grayscale version as placeholder
        # In practice, this should use a segmentation model or load from dataset
        if len(image.shape) == 4:  # (batch, num_cam, C, H, W)
            # Convert to grayscale as placeholder segmentation
            seg = torch.mean(image, dim=2, keepdim=True)  # (batch, num_cam, 1, H, W)
            seg = seg.repeat(1, 1, 3, 1, 1)  # (batch, num_cam, 3, H, W)
            return seg
        else:
            # Single image
            seg = torch.mean(image, dim=0, keepdim=True)  # (1, H, W) or (C, H, W)
            if len(seg.shape) == 3:
                seg = seg.unsqueeze(0)  # Add channel dim if needed
            seg = seg.repeat(3, 1, 1)  # Make it 3-channel
            return seg

    def _paint_poses_on_segmentation(self, segmentation, object_pose=None, eef_pose=None, contact_normal=None):
        """
        Paint pose overlays on segmentation masks.
        
        Args:
            segmentation: Segmentation masks (batch, num_cam, C, H, W) or (num_cam, C, H, W)
            object_pose: Object pose [x, y, z, qw, qx, qy, qz] (optional)
            eef_pose: End-effector pose [x, y, z, qw, qx, qy, qz] (optional)
            contact_normal: Dict with 'point' and 'normal' keys (optional)
        
        Returns:
            Painted segmentation images
        """
        # Convert to numpy for painting
        if isinstance(segmentation, torch.Tensor):
            is_torch = True
            device = segmentation.device
            seg_np = segmentation.detach().cpu().numpy()
        else:
            is_torch = False
            seg_np = segmentation
        
        # Handle batch dimension
        if len(seg_np.shape) == 5:  # (batch, num_cam, C, H, W)
            batch_size = seg_np.shape[0]
            num_cam = seg_np.shape[1]
            painted_images = []
            
            for b in range(batch_size):
                cam_images = []
                for c in range(num_cam):
                    seg_cam = seg_np[b, c]  # (C, H, W)
                    seg_cam = np.transpose(seg_cam, (1, 2, 0))  # (H, W, C)
                    seg_cam = (seg_cam * 255).astype(np.uint8)  # Convert to uint8
                    
                    # Paint poses
                    painted = paint_pose_on_segmentation(
                        seg_cam,
                        object_pose=object_pose,
                        eef_pose=eef_pose,
                        contact_normal=contact_normal,
                        camera_intrinsics=self.camera_intrinsics,
                        camera_extrinsics=self.camera_extrinsics,
                        axis_length=self.axis_length,
                        thickness=self.pose_thickness,
                        noise_scale=self.pose_noise_scale if self.training else 0.0,
                        ablation_mode=self.ablation_mode
                    )
                    
                    painted = painted.astype(np.float32) / 255.0  # Back to [0, 1]
                    painted = np.transpose(painted, (2, 0, 1))  # (C, H, W)
                    cam_images.append(painted)
                
                painted_images.append(np.stack(cam_images, axis=0))  # (num_cam, C, H, W)
            
            painted_np = np.stack(painted_images, axis=0)  # (batch, num_cam, C, H, W)
        else:
            # Single image or no batch dimension
            if len(seg_np.shape) == 4:  # (num_cam, C, H, W)
                num_cam = seg_np.shape[0]
                cam_images = []
                for c in range(num_cam):
                    seg_cam = seg_np[c]  # (C, H, W)
                    seg_cam = np.transpose(seg_cam, (1, 2, 0))  # (H, W, C)
                    seg_cam = (seg_cam * 255).astype(np.uint8)
                    
                    painted = paint_pose_on_segmentation(
                        seg_cam,
                        object_pose=object_pose,
                        eef_pose=eef_pose,
                        contact_normal=contact_normal,
                        camera_intrinsics=self.camera_intrinsics,
                        camera_extrinsics=self.camera_extrinsics,
                        axis_length=self.axis_length,
                        thickness=self.pose_thickness,
                        noise_scale=self.pose_noise_scale if self.training else 0.0,
                        ablation_mode=self.ablation_mode
                    )
                    
                    painted = painted.astype(np.float32) / 255.0
                    painted = np.transpose(painted, (2, 0, 1))  # (C, H, W)
                    cam_images.append(painted)
                
                painted_np = np.stack(cam_images, axis=0)  # (num_cam, C, H, W)
            else:
                # Single camera image
                seg_cam = np.transpose(seg_np, (1, 2, 0))  # (H, W, C)
                seg_cam = (seg_cam * 255).astype(np.uint8)
                
                painted = paint_pose_on_segmentation(
                    seg_cam,
                    object_pose=object_pose,
                    eef_pose=eef_pose,
                    contact_normal=contact_normal,
                    camera_intrinsics=self.camera_intrinsics,
                    camera_extrinsics=self.camera_extrinsics,
                    axis_length=self.axis_length,
                    thickness=self.pose_thickness,
                    noise_scale=self.pose_noise_scale if self.training else 0.0,
                    ablation_mode=self.ablation_mode
                )
                
                painted = painted.astype(np.float32) / 255.0
                painted_np = np.transpose(painted, (2, 0, 1))  # (C, H, W)
        
        # Convert back to torch if needed
        if is_torch:
            painted_tensor = torch.from_numpy(painted_np).float().to(device)
            return painted_tensor
        else:
            return painted_np

    def __call__(self, qpos, image, effort, actions=None, is_pad=None,
                 object_pose=None, eef_pose=None, contact_normal=None):
        """
        Forward pass of SegACT policy.
        
        Args:
            qpos: Robot joint positions
            image: RGB images (will be converted to segmentation)
            effort: Joint efforts/torques
            actions: Action sequence for training
            is_pad: Padding mask
            object_pose: Object pose [x, y, z, qw, qx, qy, qz] (optional)
            eef_pose: End-effector pose [x, y, z, qw, qx, qy, qz] (optional)
            contact_normal: Dict with 'point' and 'normal' keys (optional)
        
        Returns:
            During training: loss_dict
            During inference: predicted actions
        """
        env_state = None
        
        # Get segmentation mask
        segmentation = self._get_segmentation_mask(image)
        
        # Paint poses on segmentation
        painted_segmentation = self._paint_poses_on_segmentation(
            segmentation, object_pose, eef_pose, contact_normal
        )
        
        # Normalize (same as ACT)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        painted_segmentation = normalize(painted_segmentation)
        
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, painted_segmentation, effort, env_state, actions, is_pad
            )
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, painted_segmentation, effort, env_state)
            return a_hat

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
