import argparse
from copy import deepcopy
import os
import re
import sys
import pickle
import cv2
import threading
import yaml

# to avoid x11/quartz over ssh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb

def load_yaml_file(config_type: str = "robot",
                   name: str = "aloha_stationary",
                   base_path: str = None) -> dict:
    """
    Standalone YAML loader (adapted from `aloha.robot_utils.load_yaml_file`)
    so that training does not require the `aloha` Python package.

    :param config_type: Type of configuration to load, e.g., 'robot' or 'task'.
    :param name: Name of the robot configuration to load when config_type == "robot".
    :param base_path: Base directory containing the YAML configuration files.
    :return: The loaded configuration as a dictionary.
    :raises FileNotFoundError: If the specified configuration file does not exist.
    :raises RuntimeError: If there is an error loading the YAML file.
    """

    if base_path is None:
        raise ValueError("`base_path` must be provided for `load_yaml_file`.")

    # Set the YAML file path based on the configuration type
    if config_type == "robot":
        yaml_file_path = os.path.join(base_path, "robot", f"{name}.yaml")
    elif config_type == "task":
        yaml_file_path = os.path.join(base_path, "tasks_config.yaml")
    else:
        raise ValueError(
            f"Unsupported config_type '{config_type}'. Use 'robot' or 'task'."
        )

    # Check if file exists and load
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(
            f"Configuration file '{yaml_file_path}' not found."
        )

    try:
        with open(yaml_file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(
            f"Failed to load YAML file '{yaml_file_path}': {e}"
        )


# Initialize USE_OBS_TARGET (will be set from command-line argument in main())
USE_OBS_TARGET = False

from policy import (
    ACTPolicy,
    CNNMLPPolicy,
    ACT_IK_Policy,
)
# Simulation is not supported for Aloha Solo
# TODO: Uncommenting this will result in error. Needs to be fixed
# from sim_env import (
#     BOX_POSE,
# )
from utils import (
    compute_dict_mean,
    detach_dict,
    load_data,
    sample_box_pose,
    sample_insertion_pose,
    save_videos,
    set_seed,
)


def main(args):
    set_seed(1)
    # Set module-level variable for USE_OBS_TARGET (used by policy.py)
    global USE_OBS_TARGET
    USE_OBS_TARGET = args.get('use_obs_target', False)
    
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    robot = args['robot']


    # base_path = os.path.expanduser("~/interbotix_ws/src/aloha/config")
    base_path = os.path.expanduser("~/aloha/config")

    robot_config = load_yaml_file("robot", robot, base_path=base_path).get('robot')

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        task_config = load_yaml_file("task", base_path=base_path)
        task_config = task_config['tasks'].get(task_name)

    dataset_dir = os.path.expanduser(task_config.get('dataset_dir'))
    episode_len = task_config.get('episode_len')

    # Camera filtering: exclude cameras specified in args['exclude_cameras'] if provided
    all_camera_names = [camera['name'] for camera in robot_config.get('cameras').get('camera_instances')]
    exclude_cameras = args.get('exclude_cameras', [])
    if exclude_cameras:
        if isinstance(exclude_cameras, str):
            exclude_cameras = [exclude_cameras]
        # Check if all exclude_cameras exist in all_camera_names
        for cam in exclude_cameras:
            if cam not in all_camera_names:
                raise ValueError(f"Camera to exclude '{cam}' not found in available camera names: {all_camera_names}")
        camera_names = [name for name in all_camera_names if name not in exclude_cameras]
    else:
        camera_names = all_camera_names

    # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class in ['ACT', 'ACT_IK']:
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': camera_names,
            'use_obs_target': USE_OBS_TARGET,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr': args['lr'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'num_queries': 1,
            'camera_names': camera_names,
        }
    else:
        raise NotImplementedError("policy_class must be one of 'ACT', 'ACT_IK', or 'CNNMLP'.")

    ##  Consolidate the all changes to the configurations above
    ##  Iow Config parameters AFTER this line is not supposed to be modified.

    # Construct default overlay image path from task name if not provided
    overlay_image_path = args.get('overlay_image_path')
    if overlay_image_path is None:
        # Default pattern: /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/{task_name}/overlay.jpg
        default_overlay_path = f'/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/{task_name}/overlay.jpg'
        if os.path.exists(default_overlay_path):
            overlay_image_path = default_overlay_path
            print(f'Using default overlay image: {overlay_image_path}')
        else:
            print(f'Default overlay image not found at {default_overlay_path}, overlay will be disabled')
    
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'robot_config': robot_config,
        'num_rollouts': args['num_rollouts'],
        'overlay_image_path': overlay_image_path,
        'overlay_camera': args.get('overlay_camera', 'camera_left_shoulder'),
        'overlay_opacity': args.get('overlay_opacity', 0.5),
    }

    # Initialize wandb if enabled
    use_wandb = args.get('use_wandb', False)
    if use_wandb:
        wandb_project = args.get('wandb_project', 'act_training')
        wandb_entity = args.get('wandb_entity', None)
        wandb_name = args.get('wandb_name', None) or f"{task_name}_{policy_class}_seed{args['seed']}"
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config=config,
            dir=ckpt_dir,
        )
        print(f'Initialized wandb: project={wandb_project}, name={wandb_name}')

    if is_eval:
        print("Eval mode is not supported in this no-ROS version of the script.")
        if use_wandb:
            wandb.finish()
        return

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_obs_target=USE_OBS_TARGET,
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, use_wandb=use_wandb)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
    
    if use_wandb:
        wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'ACT_IK ':
        policy = ACT_IK_Policy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'ACT_IK':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_latest_rollout_index(ckpt_dir):
    """Detect the latest rollout index from existing video and qpos files."""
    max_index = -1
    if not os.path.isdir(ckpt_dir):
        return max_index
    
    # Check for video files: video{number}.mp4
    video_pattern = re.compile(r'video(\d+)\.mp4$')
    # Check for qpos files: qpos_{number}.pkl
    qpos_pattern = re.compile(r'qpos_(\d+)\.pkl$')
    
    for filename in os.listdir(ckpt_dir):
        # Check video files
        match = video_pattern.match(filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
        
        # Check qpos files
        match = qpos_pattern.match(filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    
    return max_index


        


def forward_pass(data, policy):
    image_data, qpos_data, effort_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # effort_data is kept in data tuple for recording purposes but not used by policy
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config, use_wandb=False):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # Log training and validation metrics to wandb together
        if use_wandb:
            log_dict = {'epoch': epoch}
            # Add validation metrics
            val_epoch_summary = validation_history[-1]
            for k, v in val_epoch_summary.items():
                log_dict[f'val/{k}'] = v.item()
            # Add training metrics
            for k, v in epoch_summary.items():
                log_dict[f'train/{k}'] = v.item()
            wandb.log(log_dict, step=epoch)

        if epoch % 5000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            if use_wandb:
                wandb.log({'checkpoint/epoch': epoch})
        if epoch % 100 == 0:
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            # Log plots to wandb
            if use_wandb:
                for key in train_history[0]:
                    plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
                    if os.path.exists(plot_path):
                        wandb.log({f'plots/{key}': wandb.Image(plot_path)}, step=epoch)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')  ## DO NOT CHANGE THIS ONE
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    
    # Log best checkpoint info to wandb
    if use_wandb:
        wandb.log({
            'best/epoch': best_epoch,
            'best/val_loss': min_val_loss,
        })
        wandb.summary['best_epoch'] = best_epoch
        wandb.summary['best_val_loss'] = min_val_loss

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    
    # Log final plots to wandb
    if use_wandb:
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
            if os.path.exists(plot_path):
                wandb.log({f'final_plots/{key}': wandb.Image(plot_path)})

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--robot', action='store', type=str, help='robot', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--num_rollouts', action='store', type=int, default=10, help='num_rollouts', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    ##  cameras := camera_wrist_left, camera_right_shoulder, camera_left_shoulder
    parser.add_argument('--exclude_cameras', action='store', type=str, nargs='+', help='Camera names to exclude (e.g., camera_right_shoulder)', required=False)
    
    # wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging', required=False)
    parser.add_argument('--wandb_project', action='store', type=str, help='wandb project name', required=False, default='act_training')
    parser.add_argument('--wandb_entity', action='store', type=str, help='wandb entity/team name', required=False, default=None)
    parser.add_argument('--wandb_name', action='store', type=str, help='wandb run name', required=False, default=None)
    
    # USE_OBS_TARGET argument
    parser.add_argument('--use_obs_target', action='store_true', help='Use observations as target instead of actions', required=False)
    
    # Overlay arguments
    parser.add_argument('--overlay_image_path', action='store', type=str, help='Path to overlay image file (optional)', required=False, default=None)
    parser.add_argument('--overlay_camera', action='store', type=str, help='Camera name for overlay display (default: camera_left_shoulder)', required=False, default='camera_left_shoulder')
    parser.add_argument('--overlay_opacity', action='store', type=float, help='Initial opacity for overlay (0.0 to 1.0, default: 0.5)', required=False, default=0.5)

    argument = vars(parser.parse_args())
    main(argument)
