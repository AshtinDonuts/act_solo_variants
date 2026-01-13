# IACT_A: Impedance ACT with Primitive-Based Control

## Overview

IACT_A (Impedance ACT) extends vanilla ACT by replacing direct action prediction with **primitive-based control**. Instead of outputting joint positions directly, the model predicts **primitives** (high-level behaviors) and their parameters, which are then converted to **impedance controller targets** (stiffness K, damping D, reference positions x_ref, reference velocities v_ref).

## Architecture

### Pipeline

```
IACT_A Pipeline:
ACT → primitive tokens (z, θ) → PrimitiveExecutor → impedance targets (K, D, x_ref, v_ref) → impedance controller
```

The key difference from vanilla ACT:
- **Vanilla ACT**: `ACT → q_ref (joint positions) → position controller`
- **IACT_A**: `ACT → primitives (z, θ) → impedance targets → impedance controller`

### Model Architecture

The IACT_A model uses a **PrimitiveDecoderHead** that outputs:

1. **Primitive Logits** `z[t:t+H]`: Categorical distribution over primitive types
2. **Primitive Parameters** `θ[t:t+H]`: Regression parameters for each primitive

The model architecture is based on DETRVAE (same as vanilla ACT) but replaces the action head with a primitive decoder head.

### Primitive Types

IACT_A supports 8 primitive types:

- **MOVE**: Free-space motion
- **GRASP**: Grasping with higher stiffness
- **RELEASE**: Releasing with lower stiffness
- **INSERT**: Insertion with moderate stiffness
- **PULL**: Pulling with moderate stiffness
- **PUSH**: Pushing with moderate stiffness
- **ROTATE**: Rotation with reduced stiffness
- **HOLD**: Holding with high stiffness

## Key Components

### 1. Model (`models/IACT.py`)

The `IACTVAE` model:

- **Shared Backbone**: Same ResNet backbone and transformer encoder/decoder as vanilla ACT
- **PrimitiveDecoderHead**: 
  - `primitive_id_head`: Outputs primitive logits (categorical)
  - `primitive_param_head`: Outputs primitive parameters (regression)
- **VAE Latent**: Uses the same VAE latent space for action chunking

**Model Outputs:**
- `primitive_logits`: (batch, num_queries, num_primitives)
- `primitive_params`: (batch, num_queries, primitive_param_dim)
- `is_pad_hat`: Padding predictions
- `(mu, logvar)`: VAE latent statistics

### 2. PrimitiveExecutor (`primitive_executor.py`)

The core component that translates primitives to impedance controller targets.

#### Functionality

**Inputs:**
- `primitive_id`: Primitive type (from categorical distribution)
- `primitive_params`: Parameters θ (e.g., stiffness, damping, reference positions)

**Outputs:**
- `K`: Stiffness matrix (num_dof, num_dof) - diagonal matrix
- `D`: Damping matrix (num_dof, num_dof) - diagonal matrix
- `x_ref`: Reference position (num_dof,)
- `v_ref`: Reference velocity (num_dof,)

#### Parameter Structure

The primitive parameters are expected to have the following structure:
```
θ = [K_diag (num_dof), D_diag (num_dof), x_ref (num_dof), v_ref (num_dof), ...]
```

If the parameter dimension is insufficient, the executor falls back to:
- Primitive-specific default stiffness/damping values
- Extracted reference positions from the first `num_dof` elements

#### Primitive-Specific Impedance

Each primitive has default stiffness and damping multipliers:

| Primitive | Stiffness Multiplier | Damping Multiplier |
|-----------|---------------------|-------------------|
| MOVE | 1.0 | 1.0 |
| GRASP | 2.0 | 1.5 |
| RELEASE | 0.5 | 0.5 |
| INSERT | 1.5 | 1.2 |
| PULL | 1.2 | 1.0 |
| PUSH | 1.2 | 1.0 |
| ROTATE | 0.8 | 0.8 |
| HOLD | 2.0 | 1.5 |

#### Termination Evaluation

The executor can evaluate termination conditions:
- **CONTACT**: Force exceeds threshold
- **SLIP**: High velocity during grasp
- **JAM**: High force with low velocity
- **TIMEOUT**: Maximum steps reached
- **SUCCESS**: Reached target position (⚠️ **Not fully implemented**)

### 3. Policy Wrapper (`IACT_policy.py`)

The `IACTPolicy` class wraps the model and handles training/inference.

#### Training

During training, the policy:
1. Predicts primitives from ground truth actions
2. Converts primitives to actions (simplified, non-differentiable)
3. Computes reconstruction loss on actions

**Current Training Approach:**
- Extracts `x_ref` from primitive parameters (first `num_dof` elements)
- Compares with ground truth actions using L1 loss
- Uses parameter regularization to encourage reasonable values

**Loss Function:**
```
L = L_reconstruction + λ_kl * L_kl + λ_param * L_param_reg
```

Where:
- `L_reconstruction`: L1 loss between predicted and ground truth actions
- `L_kl`: KL divergence (VAE regularization)
- `L_param_reg`: Parameter regularization (L2 on primitive parameters)

#### Inference

During inference:
1. Model predicts primitive logits and parameters
2. Primitive ID is sampled from the distribution
3. PrimitiveExecutor converts to impedance targets
4. Returns impedance targets for low-level controller

## Implementation Status

### ✅ Fully Implemented

1. **Model Architecture**
   - PrimitiveDecoderHead with primitive logits and parameters
   - VAE encoder/decoder structure
   - Transformer backbone integration

2. **PrimitiveExecutor**
   - Primitive-to-impedance conversion
   - Primitive-specific stiffness/damping mapping
   - Basic termination evaluation (contact, slip, jam, timeout)

3. **Policy Wrapper**
   - Training loop with action reconstruction loss
   - Inference with primitive sampling
   - Impedance target conversion

### ⚠️ In Progress / Partially Implemented

1. **Training Loss**
   - ❌ **No ground truth primitive labels**: The training doesn't use supervised primitive classification
   - ❌ **Simplified action conversion**: Currently just extracts `x_ref` from params (non-differentiable)
   - ❌ **No primitive classification loss**: Cross-entropy loss on primitives is commented out
   - ⚠️ **Workaround**: Uses action reconstruction loss as proxy

2. **Data Labeling**
   - ❌ **No auto-segmentation utilities**: Unlike IACT_B, there's no automatic primitive labeling from events
   - ❌ **No event detection**: No utilities to detect contact/slip/jam events from logged data
   - ❌ **No primitive parameter extraction**: No automatic extraction of parameters from demonstrations

3. **Termination Evaluation**
   - ⚠️ **Target position handling**: `evaluate_termination()` raises `NotImplementedError` when `target_position` is provided
   - ⚠️ **Success detection**: Cannot properly detect when a primitive completes successfully

4. **Configuration**
   - ⚠️ **Hardcoded state_dim**: Model has `state_dim = 7` hardcoded (TODO comment)
   - ⚠️ **Hardcoded parameters**: Some thresholds are hardcoded (e.g., `JAM_VELOCITY_TOLERANCE = 0.01`)

5. **Differentiable Training**
   - ❌ **Non-differentiable conversion**: Primitive-to-action conversion uses `.detach()` and numpy, breaking gradients
   - ⚠️ **Impact**: Cannot backpropagate through primitive predictions to improve primitive learning

## Usage

### Training

```python
from IACT_policy import IACTPolicy

args_override = {
    'kl_weight': 10.0,
    'num_primitives': 8,
    'primitive_param_dim': 14,
    'state_dim': 7,
    # ... other ACT args
}

policy = IACTPolicy(args_override)

# Training loop
for batch in dataloader:
    qpos, image, effort, actions, is_pad = batch
    loss_dict = policy(qpos, image, effort, actions, is_pad)
    loss = loss_dict['loss']
    # ... backprop and optimize
```

**Note**: Currently, training uses action reconstruction as a proxy. For proper training, you would need:
- Ground truth primitive labels
- Proper differentiable primitive-to-action conversion
- Primitive classification loss

### Inference

```python
# Inference
outputs = policy(qpos, image, effort)
primitive_logits = outputs['primitive_logits']  # (batch, num_queries, num_primitives)
primitive_params = outputs['primitive_params']  # (batch, num_queries, param_dim)

# Convert to impedance targets
impedance_targets = policy.get_impedance_targets(
    primitive_logits, primitive_params,
    current_position, current_velocity
)

# Each target contains: {'K', 'D', 'x_ref', 'v_ref'}
# Send to impedance controller
for target in impedance_targets:
    controller.set_impedance(target['K'], target['D'])
    controller.set_reference(target['x_ref'], target['v_ref'])
```

## Differences from Vanilla ACT and IACT_B

### vs. Vanilla ACT

- **Vanilla ACT**: Outputs joint positions → position controller
- **IACT_A**: Outputs primitives → impedance targets → impedance controller
- **Requirement**: IACT_A requires an impedance controller (not just position control)

### vs. IACT_B

- **IACT_A**: Single head → primitives → impedance targets → impedance control
- **IACT_B**: Dual heads → (joint refs + primitives) → modifies joint ref execution → position control
- **Key Difference**: IACT_A replaces actions with primitives; IACT_B augments actions with primitives

## File Structure

```
IACT_A/
├── __init__.py              # Package exports
├── README.md                # This file
├── main.py                  # Model builder and optimizer
├── primitive_executor.py    # Primitive-to-impedance conversion
└── models/
    ├── __init__.py
    └── IACT.py              # IACTVAE model with PrimitiveDecoderHead
```

## Configuration Parameters

### Model
- `num_primitives`: Number of primitive types (default: 8)
- `primitive_param_dim`: Dimension of primitive parameters (default: 14)
- `state_dim`: Robot state dimension (default: 7, **hardcoded**)

### PrimitiveExecutor
- `default_stiffness`: Default stiffness (default: 100.0)
- `default_damping`: Default damping (default: 10.0)
- `contact_force_threshold`: Force threshold for contact (default: 5.0)
- `slip_velocity_threshold`: Velocity threshold for slip (default: 0.1)
- `jam_force_threshold`: Force threshold for jam (default: 20.0)
- `timeout_steps`: Maximum steps before timeout (default: 500)

### Training
- `kl_weight`: KL divergence weight (default: 10.0)

## Future Work / TODO

### High Priority

1. **Proper Training Loss**
   - [ ] Add ground truth primitive labeling from demonstrations
   - [ ] Implement primitive classification loss (cross-entropy)
   - [ ] Make primitive-to-action conversion differentiable
   - [ ] Add proper parameter regression loss with ground truth parameters

2. **Data Labeling Utilities**
   - [ ] Implement auto-segmentation from events (similar to IACT_B)
   - [ ] Add event detection (contact, slip, jam)
   - [ ] Extract primitive parameters from logged demonstrations

3. **Termination Evaluation**
   - [ ] Implement target position handling
   - [ ] Add proper success detection
   - [ ] Improve termination condition logic

### Medium Priority

4. **Configuration**
   - [ ] Make `state_dim` configurable (remove hardcoding)
   - [ ] Make all thresholds configurable
   - [ ] Add configuration validation

5. **Differentiable Training**
   - [ ] Implement differentiable primitive-to-impedance conversion
   - [ ] Add gradient flow through PrimitiveExecutor
   - [ ] Consider using soft primitive assignments

6. **Primitive Parameter Structure**
   - [ ] Document expected parameter structure clearly
   - [ ] Add parameter validation
   - [ ] Support flexible parameter dimensions

### Low Priority

7. **Testing**
   - [ ] Add unit tests for PrimitiveExecutor
   - [ ] Add integration tests for training/inference
   - [ ] Add validation on real robot data

8. **Documentation**
   - [ ] Add detailed parameter structure documentation
   - [ ] Add examples for each primitive type
   - [ ] Add troubleshooting guide

## Known Issues

1. **Training doesn't use primitive labels**: The model learns primitives indirectly through action reconstruction, which may not be optimal.

2. **Non-differentiable conversion**: The primitive-to-action conversion breaks gradients, limiting end-to-end learning.

3. **Target position not implemented**: Success detection requires target positions, which is not currently supported.

4. **Hardcoded dimensions**: State dimension and some parameters are hardcoded, limiting flexibility.

## References

- ACT: Action Chunking with Transformers (Zhou et al.)
- Impedance control theory
- Primitive-based manipulation
