"""
Pose painting utilities for SegACT.

Draws pose overlays on segmentation masks:
- Object pose axes (XYZ RGB)
- End-effector (EEF) frame axes / approach direction
- Optional contact normal arrow
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import random


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    
    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w]
    
    Returns:
        3x3 rotation matrix
    """
    if quat.shape[0] == 4:
        # Assume [w, x, y, z] format
        w, x, y, z = quat
    else:
        raise ValueError(f"Expected quaternion of length 4, got {len(quat)}")
    
    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def project_3d_to_2d(point_3d: np.ndarray, 
                    camera_intrinsics: Optional[np.ndarray] = None,
                    camera_extrinsics: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """
    Project 3D point to 2D image coordinates.
    
    Args:
        point_3d: 3D point (x, y, z) in world/camera frame
        camera_intrinsics: 3x3 camera intrinsic matrix (if None, assumes identity)
        camera_extrinsics: 4x4 camera extrinsic matrix (if None, assumes point is in camera frame)
    
    Returns:
        (u, v) pixel coordinates
    """
    if camera_intrinsics is None:
        # Default intrinsics (approximate, should be provided)
        fx = fy = 500.0
        cx = 320.0
        cy = 240.0
        camera_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    # Transform to camera frame if extrinsics provided
    if camera_extrinsics is not None:
        point_3d_hom = np.append(point_3d, 1.0)
        point_cam = camera_extrinsics @ point_3d_hom
        point_3d = point_cam[:3]
    
    # Project to image plane
    if point_3d[2] <= 0:
        # Point behind camera
        return None, None
    
    x, y, z = point_3d
    u = int(camera_intrinsics[0, 0] * x / z + camera_intrinsics[0, 2])
    v = int(camera_intrinsics[1, 1] * y / z + camera_intrinsics[1, 2])
    
    return u, v


def draw_axes_triad(image: np.ndarray,
                    pose: np.ndarray,
                    axis_length: float = 0.05,
                    camera_intrinsics: Optional[np.ndarray] = None,
                    camera_extrinsics: Optional[np.ndarray] = None,
                    thickness: int = 2,
                    noise_scale: float = 0.0) -> np.ndarray:
    """
    Draw XYZ axes triad at object pose.
    
    Args:
        image: Segmentation canvas (H, W, 3) uint8
        pose: Object pose [x, y, z, qw, qx, qy, qz] or [x, y, z, qx, qy, qz, qw]
        axis_length: Length of axes in meters
        camera_intrinsics: 3x3 camera intrinsic matrix
        camera_extrinsics: 4x4 camera extrinsic matrix
        thickness: Line thickness (with randomization)
        noise_scale: Scale for random noise on pose (for augmentation)
    
    Returns:
        Image with axes drawn
    """
    image = image.copy()
    H, W = image.shape[:2]
    
    # Parse pose
    if len(pose) == 7:
        pos = pose[:3]
        quat = pose[3:]
        # Assume [x, y, z, qw, qx, qy, qz] format (common in robotics)
        # Convert to [w, x, y, z] for our quaternion functions
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])  # [qw, qx, qy, qz] -> [w, x, y, z]
    else:
        raise ValueError(f"Expected pose of length 7, got {len(pose)}")
    
    # Add noise for augmentation
    if noise_scale > 0:
        pos = pos + np.random.normal(0, noise_scale, 3)
        # Add small rotation noise (simplified, without scipy)
        noise_angle = np.random.normal(0, noise_scale * 0.1)
        if abs(noise_angle) > 1e-6:
            noise_axis = np.random.randn(3)
            noise_axis = noise_axis / (np.linalg.norm(noise_axis) + 1e-8)
            # Small angle approximation: quaternion rotation
            cos_half = np.cos(noise_angle / 2)
            sin_half = np.sin(noise_angle / 2)
            noise_quat = np.array([cos_half, sin_half * noise_axis[0], 
                                  sin_half * noise_axis[1], sin_half * noise_axis[2]])
            # Quaternion multiplication (simplified)
            w1, x1, y1, z1 = quat
            w2, x2, y2, z2 = noise_quat
            quat = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            # Normalize
            quat = quat / (np.linalg.norm(quat) + 1e-8)
    
    # Get rotation matrix
    R = quaternion_to_rotation_matrix(quat)
    
    # Define axis directions in local frame
    axes_local = np.array([
        [axis_length, 0, 0],  # X axis (red)
        [0, axis_length, 0],  # Y axis (green)
        [0, 0, axis_length],  # Z axis (blue)
    ])
    
    # Transform to world frame
    axes_world = pos[None, :] + (R @ axes_local.T).T
    
    # Project to image
    origin_2d = project_3d_to_2d(pos, camera_intrinsics, camera_extrinsics)
    if origin_2d[0] is None:
        return image  # Point behind camera
    
    # Randomize thickness
    if thickness > 0:
        actual_thickness = max(1, int(thickness + random.uniform(-1, 1)))
    else:
        actual_thickness = 1
    
    # Draw axes with RGB colors
    colors = [
        (0, 0, 255),    # Red for X
        (0, 255, 0),    # Green for Y
        (255, 0, 0),    # Blue for Z
    ]
    
    for i, (axis_world, color) in enumerate(zip(axes_world, colors)):
        axis_2d = project_3d_to_2d(axis_world, camera_intrinsics, camera_extrinsics)
        if axis_2d[0] is None:
            continue
        
        # Add color noise for augmentation
        if noise_scale > 0:
            color_noise = np.random.randint(-20, 20, 3)
            color = tuple(np.clip(np.array(color) + color_noise, 0, 255).astype(int))
        
        # Draw line
        cv2.line(image, 
                (int(origin_2d[0]), int(origin_2d[1])),
                (int(axis_2d[0]), int(axis_2d[1])),
                color, actual_thickness)
    
    return image


def draw_eef_axes(image: np.ndarray,
                  eef_pose: np.ndarray,
                  axis_length: float = 0.05,
                  approach_direction: Optional[np.ndarray] = None,
                  camera_intrinsics: Optional[np.ndarray] = None,
                  camera_extrinsics: Optional[np.ndarray] = None,
                  thickness: int = 2,
                  noise_scale: float = 0.0) -> np.ndarray:
    """
    Draw end-effector frame axes and approach direction.
    
    Args:
        image: Segmentation canvas (H, W, 3) uint8
        eef_pose: EEF pose [x, y, z, qw, qx, qy, qz]
        axis_length: Length of axes in meters
        approach_direction: Optional approach direction vector (3D) - if None, uses Z-axis
        camera_intrinsics: 3x3 camera intrinsic matrix
        camera_extrinsics: 4x4 camera extrinsic matrix
        thickness: Line thickness
        noise_scale: Scale for random noise
    
    Returns:
        Image with EEF axes drawn
    """
    image = image.copy()
    
    # Parse pose
    if len(eef_pose) == 7:
        pos = eef_pose[:3]
        quat = eef_pose[3:]
        # Assume [x, y, z, qw, qx, qy, qz] format
        quat = np.array([quat[0], quat[1], quat[2], quat[3]])  # [qw, qx, qy, qz] -> [w, x, y, z]
    else:
        raise ValueError(f"Expected eef_pose of length 7, got {len(eef_pose)}")
    
    # Add noise for augmentation
    if noise_scale > 0:
        pos = pos + np.random.normal(0, noise_scale, 3)
    
    # Get rotation matrix
    R = quaternion_to_rotation_matrix(quat)
    
    # Use Z-axis as approach direction if not provided
    if approach_direction is None:
        approach_direction = R @ np.array([0, 0, 1])  # Z-axis in local frame
    approach_direction = approach_direction / np.linalg.norm(approach_direction)
    
    # Draw approach direction arrow (longer, cyan color)
    approach_end = pos + approach_direction * axis_length * 1.5
    origin_2d = project_3d_to_2d(pos, camera_intrinsics, camera_extrinsics)
    approach_end_2d = project_3d_to_2d(approach_end, camera_intrinsics, camera_extrinsics)
    
    if origin_2d[0] is not None and approach_end_2d[0] is not None:
        # Randomize thickness
        if thickness > 0:
            actual_thickness = max(1, int(thickness + random.uniform(-1, 1)))
        else:
            actual_thickness = 1
        
        # Cyan color for approach direction
        color = (255, 255, 0)  # Cyan in BGR
        if noise_scale > 0:
            color_noise = np.random.randint(-20, 20, 3)
            color = tuple(np.clip(np.array(color) + color_noise, 0, 255).astype(int))
        
        cv2.arrowedLine(image,
                       (int(origin_2d[0]), int(origin_2d[1])),
                       (int(approach_end_2d[0]), int(approach_end_2d[1])),
                       color, actual_thickness, tipLength=0.2)
    
    # Also draw frame axes (smaller, yellow)
    axes_local = np.array([
        [axis_length * 0.5, 0, 0],  # X
        [0, axis_length * 0.5, 0],   # Y
        [0, 0, axis_length * 0.5],   # Z
    ])
    axes_world = pos[None, :] + (R @ axes_local.T).T
    
    origin_2d = project_3d_to_2d(pos, camera_intrinsics, camera_extrinsics)
    if origin_2d[0] is not None:
        color = (0, 255, 255)  # Yellow in BGR
        if noise_scale > 0:
            color_noise = np.random.randint(-20, 20, 3)
            color = tuple(np.clip(np.array(color) + color_noise, 0, 255).astype(int))
        
        for axis_world in axes_world:
            axis_2d = project_3d_to_2d(axis_world, camera_intrinsics, camera_extrinsics)
            if axis_2d[0] is not None:
                cv2.line(image,
                        (int(origin_2d[0]), int(origin_2d[1])),
                        (int(axis_2d[0]), int(axis_2d[1])),
                        color, max(1, actual_thickness - 1))
    
    return image


def draw_contact_normal(image: np.ndarray,
                       contact_point: np.ndarray,
                       normal: np.ndarray,
                       normal_length: float = 0.03,
                       camera_intrinsics: Optional[np.ndarray] = None,
                       camera_extrinsics: Optional[np.ndarray] = None,
                       thickness: int = 2,
                       noise_scale: float = 0.0) -> np.ndarray:
    """
    Draw desired contact normal arrow.
    
    Args:
        image: Segmentation canvas (H, W, 3) uint8
        contact_point: Contact point [x, y, z]
        normal: Normal direction [nx, ny, nz] (normalized)
        normal_length: Length of normal arrow in meters
        camera_intrinsics: 3x3 camera intrinsic matrix
        camera_extrinsics: 4x4 camera extrinsic matrix
        thickness: Line thickness
        noise_scale: Scale for random noise
    
    Returns:
        Image with contact normal drawn
    """
    image = image.copy()
    
    # Normalize normal
    normal = normal / np.linalg.norm(normal)
    
    # Add noise
    if noise_scale > 0:
        contact_point = contact_point + np.random.normal(0, noise_scale, 3)
        normal = normal + np.random.normal(0, noise_scale * 0.1, 3)
        normal = normal / np.linalg.norm(normal)
    
    # Compute arrow end
    arrow_end = contact_point + normal * normal_length
    
    # Project to image
    point_2d = project_3d_to_2d(contact_point, camera_intrinsics, camera_extrinsics)
    end_2d = project_3d_to_2d(arrow_end, camera_intrinsics, camera_extrinsics)
    
    if point_2d[0] is not None and end_2d[0] is not None:
        # Randomize thickness
        if thickness > 0:
            actual_thickness = max(1, int(thickness + random.uniform(-1, 1)))
        else:
            actual_thickness = 1
        
        # Magenta color for contact normal
        color = (255, 0, 255)  # Magenta in BGR
        if noise_scale > 0:
            color_noise = np.random.randint(-20, 20, 3)
            color = tuple(np.clip(np.array(color) + color_noise, 0, 255).astype(int))
        
        cv2.arrowedLine(image,
                       (int(point_2d[0]), int(point_2d[1])),
                       (int(end_2d[0]), int(end_2d[1])),
                       color, actual_thickness, tipLength=0.3)
    
    return image


def paint_pose_on_segmentation(segmentation: np.ndarray,
                               object_pose: Optional[np.ndarray] = None,
                               eef_pose: Optional[np.ndarray] = None,
                               contact_normal: Optional[Dict] = None,
                               camera_intrinsics: Optional[np.ndarray] = None,
                               camera_extrinsics: Optional[np.ndarray] = None,
                               axis_length: float = 0.05,
                               thickness: int = 2,
                               noise_scale: float = 0.0,
                               ablation_mode: str = 'full') -> np.ndarray:
    """
    Main function to paint pose overlays on segmentation canvas.
    
    Args:
        segmentation: Segmentation mask/canvas (H, W, 3) uint8
        object_pose: Object pose [x, y, z, qw, qx, qy, qz] (optional)
        eef_pose: End-effector pose [x, y, z, qw, qx, qy, qz] (optional)
        contact_normal: Dict with 'point' and 'normal' keys (optional)
        camera_intrinsics: 3x3 camera intrinsic matrix
        camera_extrinsics: 4x4 camera extrinsic matrix
        axis_length: Length of axes in meters
        thickness: Base line thickness (will be randomized)
        noise_scale: Scale for random noise (for augmentation)
        ablation_mode: 'seg_only', 'seg_obj', 'seg_obj_eef', 'full'
    
    Returns:
        Painted segmentation image (H, W, 3) uint8
    """
    image = segmentation.copy()
    
    # Ablation modes
    if ablation_mode == 'seg_only':
        # Only segmentation, no pose overlays
        return image
    elif ablation_mode == 'seg_obj':
        # Segmentation + object pose only
        if object_pose is not None:
            image = draw_axes_triad(image, object_pose, axis_length,
                                   camera_intrinsics, camera_extrinsics,
                                   thickness, noise_scale)
        return image
    elif ablation_mode == 'seg_obj_eef' or ablation_mode == 'full':
        # Segmentation + object pose + EEF pose
        if object_pose is not None:
            image = draw_axes_triad(image, object_pose, axis_length,
                                   camera_intrinsics, camera_extrinsics,
                                   thickness, noise_scale)
        if eef_pose is not None:
            image = draw_eef_axes(image, eef_pose, axis_length,
                                 None, camera_intrinsics, camera_extrinsics,
                                 thickness, noise_scale)
        if ablation_mode == 'full' and contact_normal is not None:
            image = draw_contact_normal(image,
                                       contact_normal['point'],
                                       contact_normal['normal'],
                                       0.03, camera_intrinsics, camera_extrinsics,
                                       thickness, noise_scale)
        return image
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")
    
    return image
