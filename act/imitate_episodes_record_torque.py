import argparse
from copy import deepcopy
import os
import re
import sys
import pickle
import cv2
import threading

# to avoid x11/quartz over ssh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROS2 imports for live camera feed
try:
    import rclpy
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("Warning: ROS2 not available. Live camera overlay will use static frame.")

from aloha.robot_utils import (
    load_yaml_file,
    FOLLOWER_GRIPPER_JOINT_OPEN
)
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import wandb

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


    base_path = os.path.expanduser("~/interbotix_ws/src/aloha/config")

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
        ckpt_names = ['policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, use_wandb=use_wandb)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        
        if use_wandb:
            wandb.finish()
        exit()

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


class ImageSubscriber(Node):
    """ROS2 node to subscribe to image_rect_raw topic."""
    
    def __init__(self, camera_name, image_callback):
        super().__init__('overlay_image_subscriber')
        self.camera_name = camera_name
        self.image_callback = image_callback
        self.bridge = CvBridge()
        
        # Subscribe to the image_rect_raw topic (same as recording script uses)
        topic_name = f'{camera_name}/camera/color/image_rect_raw'
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.ros_image_callback,
            10
        )
        self.get_logger().info(f'Subscribed to: {topic_name}')
    
    def ros_image_callback(self, msg):
        """Convert ROS Image message to OpenCV format and call callback."""
        try:
            # Convert ROS Image to OpenCV (BGR format)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_callback(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')


def show_overlay(overlay_image_path, camera_name='camera_left_shoulder', initial_opacity=0.5):
    """
    Display live camera feed with overlay image using OpenCV and ROS2.
    Similar to realsense_overlay.py - subscribes to live ROS2 camera topic.
    
    Args:
        overlay_image_path: Path to overlay image file (optional, if None, no overlay)
        camera_name: Name of camera being displayed (e.g., 'camera_left_shoulder')
        initial_opacity: Initial opacity value (0.0 to 1.0)
    
    Returns:
        None (blocks until window is closed)
    """
    if not ROS2_AVAILABLE:
        print("Error: ROS2 not available. Cannot display live camera overlay.")
        return
    
    opacity = np.clip(initial_opacity, 0.0, 1.0)
    latest_frame = None
    frame_lock = threading.Lock()
    
    # Load overlay image if provided
    overlay_image = None
    if overlay_image_path and os.path.exists(overlay_image_path):
        overlay_image = cv2.imread(overlay_image_path)
        if overlay_image is None:
            print(f"Warning: Could not load overlay image: {overlay_image_path}")
            overlay_image = None
        else:
            print(f"Loaded overlay image: {overlay_image_path}")
            print(f"Overlay image shape: {overlay_image.shape} (H, W, C)")
    
    def on_image_received(cv_image):
        """Callback when new image is received from ROS2 topic."""
        with frame_lock:
            nonlocal latest_frame
            latest_frame = cv_image
    
    # Initialize ROS2
    try:
        rclpy.init()
    except RuntimeError:
        # ROS2 already initialized
        pass
    
    image_subscriber = ImageSubscriber(camera_name, on_image_received)
    
    # Create window
    window_name = f"Overlay - {camera_name} (Press 'q' or ESC to continue)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbar for opacity if overlay exists
    if overlay_image is not None:
        cv2.createTrackbar("Opacity", window_name, int(opacity * 100), 100, lambda val: None)
    
    def prepare_overlay(target_shape, overlay_img):
        """Prepare overlay image for blending."""
        if overlay_img is None:
            return None, None
        
        target_h, target_w = target_shape[:2]
        overlay_h, overlay_w = overlay_img.shape[:2]
        
        # Create canvas and alpha mask
        overlay_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((target_h, target_w), dtype=np.float32)
        
        # If dimensions match exactly, place at (0, 0) for pixel-perfect alignment
        if overlay_h == target_h and overlay_w == target_w:
            overlay_canvas[:, :] = overlay_img
            alpha_mask[:, :] = 1.0
        # Check if overlay is smaller than target
        elif overlay_h <= target_h and overlay_w <= target_w:
            # Center the overlay image
            start_y = (target_h - overlay_h) // 2
            start_x = (target_w - overlay_w) // 2
            end_y = start_y + overlay_h
            end_x = start_x + overlay_w
            
            overlay_canvas[start_y:end_y, start_x:end_x] = overlay_img
            alpha_mask[start_y:end_y, start_x:end_x] = 1.0
        else:
            # Overlay is larger, resize to fit
            scale = min(target_w / overlay_w, target_h / overlay_h)
            new_w = int(overlay_w * scale)
            new_h = int(overlay_h * scale)
            
            overlay_resized = cv2.resize(
                overlay_img,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Center the resized overlay
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            end_y = start_y + new_h
            end_x = start_x + new_w
            
            overlay_canvas[start_y:end_y, start_x:end_x] = overlay_resized
            alpha_mask[start_y:end_y, start_x:end_x] = 1.0
        
        return overlay_canvas, alpha_mask
    
    def blend_images(camera_frame, overlay_canvas, alpha_mask, opacity_val):
        """Blend camera frame with overlay image."""
        if overlay_canvas is None or alpha_mask is None:
            return camera_frame
        
        # Convert to float for blending
        camera_float = camera_frame.astype(np.float32)
        overlay_float = overlay_canvas.astype(np.float32)
        
        # Expand alpha mask to 3 channels
        alpha_3d = np.stack([alpha_mask] * 3, axis=2)
        
        # Blend: result = (1 - opacity * alpha) * camera + (opacity * alpha) * overlay
        blended = (1.0 - opacity_val * alpha_3d) * camera_float + \
                  (opacity_val * alpha_3d) * overlay_float
        
        return blended.astype(np.uint8)
    
    print(f"\n=== Overlay Display ===")
    print(f"Camera: {camera_name}")
    print(f"Topic: {camera_name}/camera/color/image_rect_raw")
    if overlay_image is not None:
        print(f"Overlay image: {overlay_image_path}")
        print(f"Initial opacity: {opacity:.2f}")
        print("Controls:")
        print("  'q' or ESC: Continue to rollout")
        print("  '+' or '=': Increase opacity")
        print("  '-' or '_': Decrease opacity")
        print("  'r': Reset opacity to 0.5")
        print("  Trackbar: Adjust opacity (0-100)")
    else:
        print("No overlay image - showing camera feed only")
        print("Press 'q' or ESC to continue to rollout")
    print("Waiting for camera frames...")
    print("========================\n")
    
    # Start ROS2 spinning in a separate thread
    def spin_ros():
        rclpy.spin(image_subscriber)
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    try:
        while rclpy.ok():
            # Get latest frame from ROS2 subscriber
            with frame_lock:
                camera_frame = latest_frame
            
            if camera_frame is None:
                # No frame received yet, show waiting message
                waiting_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    waiting_img,
                    "Waiting for camera frames...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, waiting_img)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue
            
            # Get current opacity from trackbar if overlay exists
            if overlay_image is not None:
                opacity = cv2.getTrackbarPos("Opacity", window_name) / 100.0
            
            # Prepare overlay if image exists
            if overlay_image is not None:
                overlay_canvas, alpha_mask = prepare_overlay(camera_frame.shape, overlay_image)
                blended = blend_images(camera_frame, overlay_canvas, alpha_mask, opacity)
                
                # Add text overlay showing current opacity
                opacity_text = f"Opacity: {opacity:.2f}"
                cv2.putText(
                    blended,
                    opacity_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                display_frame = blended
            else:
                display_frame = camera_frame
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif overlay_image is not None:
                if key == ord('+') or key == ord('='):
                    opacity = min(1.0, opacity + 0.05)
                    cv2.setTrackbarPos("Opacity", window_name, int(opacity * 100))
                    print(f"Opacity: {opacity:.2f}")
                elif key == ord('-') or key == ord('_'):
                    opacity = max(0.0, opacity - 0.05)
                    cv2.setTrackbarPos("Opacity", window_name, int(opacity * 100))
                    print(f"Opacity: {opacity:.2f}")
                elif key == ord('r'):
                    opacity = 0.5
                    cv2.setTrackbarPos("Opacity", window_name, int(opacity * 100))
                    print(f"Opacity reset to: {opacity:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        image_subscriber.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
        cv2.destroyAllWindows()
        print("Overlay window closed. Starting rollout...")


def eval_bc(config, ckpt_name, save_episode=True, use_wandb=False):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    config_robot = config['robot_config']
    onscreen_cam = 'angle'
    dt = 1/config_robot.get('fps', 50)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha.robot_utils import move_grippers # requires aloha
        from aloha.real_env import make_real_env # requires aloha
        from interbotix_common_modules.common_robot.robot import (
            create_interbotix_global_node,
            get_interbotix_global_node,
            robot_startup,
        )
        from interbotix_common_modules.common_robot.exceptions import InterbotixException
        try:
            node = get_interbotix_global_node()
        except:
            node = create_interbotix_global_node('aloha')
        env = make_real_env(node=node, setup_robots= False, torque_base=False, setup_base=False, config=config_robot, bool_dynamics_torque=True)
        try:
            robot_startup(node)
        except InterbotixException:
            pass
        
        # Manual gripper setup for all follower robots (since setup_robots=False)
        # This is required for gripper commands to work correctly
        for name, bot in env.robots.items():
            if 'follower' in name:
                print(f"Setting up gripper for {name}...")
                bot.core.robot_reboot_motors('single', 'gripper', True)
                bot.core.robot_set_operating_modes('group', 'arm', 'position')
                bot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
                bot.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)
                bot.core.robot_torque_enable('group', 'arm', True)
                bot.core.robot_torque_enable('single', 'gripper', True)
        
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = config['num_rollouts']
    
    # Detect latest index and start from next index for rolling saves
    start_index = get_latest_rollout_index(ckpt_dir) + 1
    if start_index > 0:
        print(f'Found existing files up to index {start_index - 1}. Starting new rollouts from index {start_index}.')
    
    episode_returns = []
    highest_rewards = []
    for rollout_offset in range(num_rollouts):
        rollout_id = start_index + rollout_offset
        ### set task
        # TODO: Simulation is not supported
        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()
        
        # Show overlay before starting rollout (only for real robot)
        if real_robot:
            overlay_camera_name = config.get('overlay_camera', 'camera_left_shoulder')
            overlay_image_path = config.get('overlay_image_path', None)
            overlay_opacity = config.get('overlay_opacity', 0.5)
            
            # Show live camera overlay with ROS2 subscription
            show_overlay(
                overlay_image_path,
                camera_name=overlay_camera_name,
                initial_opacity=overlay_opacity
            )
        
        # Open gripper after reset completes (gripper stayed in position during reset)
        # Only do this after the first rollout (not before the first rollout)
        if real_robot and rollout_id > 0:
            move_grippers(
                env.follower_bots,
                [FOLLOWER_GRIPPER_JOINT_OPEN],
                moving_time=0.5,
                dt=dt,
            )

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        qpos_data_path = os.path.join(ckpt_dir, f'qpos_{rollout_id}.pkl')
        
        # Lists to store dynamics torque components
        gravity_torques_list = []
        kinetic_friction_torques_list = []
        static_friction_torques_list = []
        dither_speeds_list = []
        no_load_currents_list = []
        dynamics_torque_data_path = os.path.join(ckpt_dir, f'dynamics_torque_{rollout_id}.pkl')
        
        def save_qpos_data():
            """Helper function to save qpos data incrementally"""
            if save_episode:
                with open(qpos_data_path, 'wb') as f:
                    pickle.dump({'qpos': qpos_list, 'target_qpos': target_qpos_list, 'rewards': rewards}, f)
        
        def save_dynamics_torque_data():
            """Helper function to save dynamics torque data incrementally"""
            if save_episode and real_robot:
                with open(dynamics_torque_data_path, 'wb') as f:
                    pickle.dump({
                        'gravity_torques': gravity_torques_list,
                        'kinetic_friction_torques': kinetic_friction_torques_list,
                        'static_friction_torques': static_friction_torques_list,
                        'dither_speeds': dither_speeds_list,
                        'no_load_currents': no_load_currents_list,
                    }, f)
        
        try:
            with torch.inference_mode():
                for t in range(max_timesteps):
                    ### update onscreen render and wait for DT
                    if onscreen_render:
                        image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                        plt_img.set_data(image)
                        plt.pause(dt)

                    ### process previous timestep to get qpos and image_list
                    obs = ts.observation
                    if 'images' in obs:
                        image_list.append(obs['images'])
                    else:
                        image_list.append({'main': obs['image']})
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos

                    ## VIZ: Skip left shoulder camera
                    # camera_names = [camera for camera in camera_names if camera != 'left_shoulder_camera']
                    curr_image = get_image(ts, camera_names)

                    ### query policy
                    if config['policy_class'] in ['ACT', 'ACT_IK']:
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                    ### step the environment
                    # ts = env.step(target_qpos.astype(float).tolist())
                    ts = env.step(target_qpos.astype(float).tolist(), debug=True)

                    ### for visualization
                    #  This script itself doesn't have visualization.
                    #  
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(ts.reward)
                    
                    # Record dynamics torque components if available (only for real robot)
                    # Check ts.observation after step, not the old obs
                    if real_robot and 'dynamics_torque' in ts.observation:
                        dynamics = ts.observation['dynamics_torque']
                        gravity_torques_list.append(dynamics.get('gravity_torques', None))
                        kinetic_friction_torques_list.append(dynamics.get('kinetic_friction_torques', None))
                        static_friction_torques_list.append(dynamics.get('static_friction_torques', None))
                        dither_speeds_list.append(dynamics.get('dither_speeds', None))
                        no_load_currents_list.append(dynamics.get('no_load_currents', None))
                    
                    # Save incrementally every timestep
                    save_qpos_data()
                    if real_robot:
                        save_dynamics_torque_data()

            plt.close()
        except KeyboardInterrupt:
            print(f"\nInterrupted at timestep {len(qpos_list)}/{max_timesteps}. Saving collected data...")
            save_qpos_data()
            if real_robot:
                save_dynamics_torque_data()
            raise
        
        # Note: Gripper opening is now done after reset completes (see above)
        # This ensures gripper stays in position during arm reset

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, dt, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            qpos_data_path = os.path.join(ckpt_dir, f'qpos_{rollout_id}.pkl')
            with open(qpos_data_path, 'wb') as f:
                pickle.dump({'qpos': qpos_list, 'target_qpos': target_qpos_list, 'rewards': rewards}, f)
            
            # Save dynamics torque data (only for real robot)
            if real_robot:
                dynamics_torque_data_path = os.path.join(ckpt_dir, f'dynamics_torque_{rollout_id}.pkl')
                with open(dynamics_torque_data_path, 'wb') as f:
                    pickle.dump({
                        'gravity_torques': gravity_torques_list,
                        'kinetic_friction_torques': kinetic_friction_torques_list,
                        'static_friction_torques': static_friction_torques_list,
                        'dither_speeds': dither_speeds_list,
                        'no_load_currents': no_load_currents_list,
                    }, f)

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # Log to wandb if enabled
    if use_wandb:
        log_dict = {
            'eval/success_rate': success_rate,
            'eval/avg_return': avg_return,
            'eval/ckpt_name': ckpt_name,
            'eval/num_rollouts': num_rollouts,
        }
        # Add reward threshold rates
        for r in range(env_max_reward+1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            log_dict[f'eval/reward_>={r}_rate'] = more_or_equal_r_rate
        
        # Add summary statistics for rollouts
        log_dict['eval/std_return'] = np.std(episode_returns)
        log_dict['eval/min_return'] = np.min(episode_returns)
        log_dict['eval/max_return'] = np.max(episode_returns)
        
        wandb.log(log_dict)
        
        # Log per-rollout metrics as a table
        rollout_table = wandb.Table(columns=['rollout_id', 'return', 'highest_reward', 'success'])
        for i, (ep_return, ep_highest) in enumerate(zip(episode_returns, highest_rewards)):
            rollout_table.add_data(i, ep_return, ep_highest, int(ep_highest == env_max_reward))
        wandb.log({'eval/rollout_table': rollout_table})

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


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
