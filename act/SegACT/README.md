# SegACT: Pose-Painted Segmentation for ACT

SegACT implements "Plan A: Pose-painted segmentation" - making pose explicit in ACT without changing the internal architecture.

## Overview

SegACT takes segmentation masks and "paints" pose information on them as visual overlays:
- **Object pose**: XYZ axes triad (RGB colored)
- **End-effector (EEF) pose**: Frame axes and approach direction arrow
- **Contact normal**: Optional arrow indicating desired contact direction

The pose-painted segmentation images are then fed to the same ACT architecture, allowing the network to learn from visually aligned representations where segmentation provides shape/location and arrows provide pose cues.

## Architecture

- **Model**: Same as ACT (DETR-VAE with ResNet backbone)
- **Input**: Pose-painted segmentation images (instead of raw RGB)
- **Training**: Identical to baseline ACT (L1 + KL divergence losses)

## Key Features

### Ablation Modes

SegACT supports multiple ablation modes for experimentation:

1. **`seg_only`**: Only segmentation masks, no pose overlays
2. **`seg_obj`**: Segmentation + object pose axes
3. **`seg_obj_eef`**: Segmentation + object pose + EEF pose
4. **`full`**: All overlays including contact normal (default)

### Data Augmentation

- **Pose noise**: Random noise on pose positions and orientations (configurable via `pose_noise_scale`)
- **Visual noise**: Random variations in arrow thickness and color to prevent overfitting to rendering style

## Usage

### Training

```bash
python act/imitate_episodes.py \
    --ckpt_dir <checkpoint_dir> \
    --robot <robot_name> \
    --policy_class SegACT \
    --task_name <task_name> \
    --batch_size <batch_size> \
    --seed <seed> \
    --num_epochs <num_epochs> \
    --lr <learning_rate> \
    --kl_weight <kl_weight> \
    --chunk_size <chunk_size> \
    --hidden_dim <hidden_dim> \
    --dim_feedforward <dim_feedforward> \
    --ablation_mode full \
    --pose_noise_scale 0.01 \
    --axis_length 0.05 \
    --pose_thickness 2
```

### Evaluation

```bash
python act/imitate_episodes.py \
    --eval \
    --ckpt_dir <checkpoint_dir> \
    --robot <robot_name> \
    --policy_class SegACT \
    --task_name <task_name> \
    --batch_size <batch_size> \
    --seed <seed> \
    --lr <learning_rate> \
    --kl_weight <kl_weight> \
    --chunk_size <chunk_size> \
    --hidden_dim <hidden_dim> \
    --dim_feedforward <dim_feedforward> \
    --ablation_mode full
```

## Parameters

### SegACT-Specific Parameters

- `--ablation_mode`: Ablation mode (`seg_only`, `seg_obj`, `seg_obj_eef`, `full`)
- `--pose_noise_scale`: Scale for random noise on pose overlays (default: 0.0)
- `--axis_length`: Length of pose axes in meters (default: 0.05)
- `--pose_thickness`: Base thickness of pose overlay lines (default: 2)

### Standard ACT Parameters

All standard ACT parameters are supported (see `imitate_episodes.py` for full list).

## Dataset Requirements

### Pose Annotations (Optional)

If pose annotations are available in your dataset, they should be stored in the HDF5 files:

- **Object pose**: `/observations/object_pose` - shape `(episode_len, 7)` - `[x, y, z, qw, qx, qy, qz]`
- **EEF pose**: `/observations/eef_pose` - shape `(episode_len, 7)` - `[x, y, z, qw, qx, qy, qz]`
- **Contact normal**: `/observations/contact_normal` - shape `(episode_len, 6)` - `[point_x, point_y, point_z, normal_x, normal_y, normal_z]`

If pose annotations are not available, SegACT will:
1. Use placeholder segmentation (grayscale conversion of RGB images)
2. Skip pose overlays if `ablation_mode='seg_only'`
3. For other modes, you'll need to provide pose estimation or use forward kinematics

### Segmentation Masks (Optional)

If segmentation masks are available:
- Store in `/observations/segmentation/{camera_name}` - shape `(episode_len, H, W, 3)` uint8

If not available, SegACT will generate placeholder segmentation from RGB images.

## Implementation Details

### Pose Painting

The `pose_painter.py` module handles:
- 3D to 2D projection (requires camera intrinsics/extrinsics)
- Drawing axes triads with RGB colors (X=red, Y=green, Z=blue)
- Drawing EEF frame and approach direction
- Drawing contact normal arrows
- Randomization for data augmentation

### Model Architecture

The `SegACT_VAE` model is identical to ACT's `DETRVAE`, but expects pose-painted segmentation images as input. The preprocessing happens in the policy layer before feeding to the model.

## Ablations for Paper

Suggested ablation studies:

1. **Segmentation only vs segmentation+object pose vs segmentation+object+EEF pose**
   - Compare `seg_only`, `seg_obj`, `seg_obj_eef`, `full` modes

2. **Wrong pose overlay sensitivity test**
   - Inject noise into pose overlays during training/eval
   - Vary `pose_noise_scale` to test robustness

3. **Rendering style robustness**
   - Vary `pose_thickness` and arrow colors
   - Test generalization across different rendering styles

## Gotchas

1. **Pose estimation dependency**: If arrows depend on external pose estimation, system performance is limited by estimator quality
2. **Rendering style overfitting**: Randomize thickness/color/noise to prevent overfitting to specific rendering style
3. **Camera calibration**: Requires camera intrinsics/extrinsics for proper 3D-to-2D projection (defaults provided if not available)

## Files

- `pose_painter.py`: Utilities for painting pose overlays on segmentation
- `models/SegACT.py`: Model architecture (identical to ACT)
- `policy.py`: Policy wrapper that handles pose painting
- `main.py`: Model builder and optimizer setup

## Future Improvements

- [ ] Integrate proper segmentation model (e.g., SAM, DINO)
- [ ] Add forward kinematics for EEF pose from qpos
- [ ] Support multiple object poses
- [ ] Learnable pose rendering (instead of fixed arrows)
