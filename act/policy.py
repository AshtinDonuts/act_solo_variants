from detr.main import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
)
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import modern_robotics as mr


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.use_obs_target = args_override.get('use_obs_target', False)
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # Heuristic to determine if Training
            
            if self.use_obs_target:  # RQ1 : Pass Follower Joints as Leader Joint states
                # When use_obs_target is True, qpos is already loaded as a sequence from dataset
                # qpos has shape (batch, seq, qpos_dim), use it directly as actions
                # But we still need the first timestep for the model input
                qpos_input = qpos[:, 0] if len(qpos.shape) == 3 else qpos  # (batch, qpos_dim) for model input
                qpos_chunk = qpos[:, :self.model.num_queries]  # (batch, num_queries, qpos_dim)
                actions = qpos_chunk
            else:               # default actions as target
                qpos_input = qpos  # (batch, qpos_dim) - already single timestep
                actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos_input, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
            
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACT_IK_Policy(nn.Module):
    """ACT Policy that operates in task space.

    This policy converts joint-space observations and actions to end-effector
    task-space poses using forward kinematics before passing them to ACT.
    """

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.use_obs_target = args_override.get('use_obs_target', False)

        # Robot kinematics configuration for FK
        # args_override can optionally provide:
        #   - 'robot_model': name from interbotix_xs_modules.xs_robot.mr_descriptions
        #                    (e.g., 'aloha_vx300s', 'aloha_wx250s')
        self.robot_model = args_override.get('robot_model', 'aloha_vx300s')
        from interbotix_xs_modules.xs_robot import mr_descriptions as mrd  # type: ignore[import]
        robot_des = getattr(mrd, self.robot_model)
        self._M = robot_des.M
        self._Slist = robot_des.Slist
        self.num_arm_joints = self._Slist.shape[1]

        # Task-space representation: [x, y, z, axis-angle(3)] => 6D
        self.task_dim = 6

        print(f'KL Weight {self.kl_weight}')
        print(f'ACT_IK_Policy using robot model: {self.robot_model} with {self.num_arm_joints} arm joints')

    def _joints_to_task(self, qpos_tensor: torch.Tensor) -> torch.Tensor:
        """Convert joint-space qpos/actions to task-space pose vectors.

        - Expects last dimension to contain at least ``num_arm_joints`` joint angles.
        - Uses Modern Robotics FKinSpace with (M, Slist) from Interbotix descriptions.
        - Returns pose as [x, y, z, axis-angle_x, axis-angle_y, axis-angle_z].
        """
        if qpos_tensor is None:
            return None

        if qpos_tensor.shape[-1] < self.num_arm_joints:
            raise ValueError(
                f"qpos last dimension ({qpos_tensor.shape[-1]}) is smaller than "
                f"num_arm_joints ({self.num_arm_joints}) for IK conversion."
            )

        orig_shape = qpos_tensor.shape
        # Flatten all leading dims, keep feature dim
        flat = qpos_tensor.reshape(-1, orig_shape[-1])

        # Detach from graph and move to CPU for numpy-based FK
        joints_np = flat[:, : self.num_arm_joints].detach().cpu().numpy()

        task_list = []
        for joints in joints_np:
            # Forward kinematics to get 4x4 transform
            T = mr.FKinSpace(self._M, self._Slist, joints.tolist())
            p = T[:3, 3]
            R = T[:3, :3]

            # Orientation as axis-angle (3D) via matrix log
            omega_theta = mr.so3ToVec(mr.MatrixLog3(R))

            task_vec = np.concatenate([p, omega_theta], axis=0)  # (6,)
            task_list.append(task_vec)

        task_np = np.stack(task_list, axis=0)  # (N, 6)
        task_tensor = torch.from_numpy(task_np).to(qpos_tensor.device, dtype=qpos_tensor.dtype)

        return task_tensor.view(*orig_shape[:-1], self.task_dim)

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # training time

            if self.use_obs_target:
                # qpos is a sequence: (batch, seq, qpos_dim)
                qpos_input = qpos[:, 0] if len(qpos.shape) == 3 else qpos  # (batch, qpos_dim)
                qpos_chunk = qpos[:, : self.model.num_queries]  # (batch, num_queries, qpos_dim)
                actions = qpos_chunk
            else:
                # default: provided actions are targets
                qpos_input = qpos  # (batch, qpos_dim)
                actions = actions[:, : self.model.num_queries]

            # Convert joint-space to task-space before feeding into ACT
            qpos_input_task = self._joints_to_task(qpos_input)
            actions_task = self._joints_to_task(actions)

            is_pad = is_pad[:, : self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos_input_task, image, env_state, actions_task, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = dict()
            all_l1 = F.l1_loss(actions_task, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict

        else:  # inference time
            # Convert current joint observation to task-space pose
            qpos_task = self._joints_to_task(qpos)
            a_hat, _, (_, _) = self.model(qpos_task, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
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
