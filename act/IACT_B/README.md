# IACT_B: Contact-Aware ACT with Primitive-Aware Execution

## Overview

IACT_B extends vanilla ACT (Action Chunking with Transformers) with a **dual-head architecture** that enables **contact-aware execution** of joint references. The key innovation is that the same joint reference (`q_ref`) can be executed differently depending on the detected contact mode, allowing the robot to adapt its behavior to different phases of manipulation tasks.

## Architecture

### Pipeline

IACT_B maintains the vanilla ACT pipeline while adding a parallel channel:

```
Vanilla ACT Pipeline:
ACT → joint chunk (q_ref) → position controller

IACT_B Pipeline:
ACT → (joint chunk, primitive tokens) → PrimitiveExecutor → executed joint commands
     ↓                                    ↓
  q_ref[t:t+H]                      z[t:t+H], θ[t:t+H]
  (joint positions)                  (primitive logits, params)
```

The **PrimitiveExecutor** sits between the policy and the robot command publisher, modifying how `q_ref` is executed based on the active primitive and detected events.

### Dual-Head Model

The IACT_B model outputs two heads per chunk step `t`:

1. **Joint Reference Head** (existing ACT behavior)
   - `q_ref[t:t+H]`: Joint position targets for the next `H` timesteps
   - Trained with L1/L2 loss against ground truth actions

2. **Primitive Head** (new)
   - `z[t:t+H]`: Primitive logits (categorical distribution over primitive types)
   - `θ[t:t+H]`: Primitive parameters (regression, e.g., force limits, compliance levels)
   - Trained with cross-entropy loss on `z` and MSE/Huber loss on `θ`

### Primitive Types

IACT_B supports 6 primitive types that correspond to different contact modes:

- **APPROACH**: Free-space motion with high stiffness and normal tracking
- **GUARDED_MOVE**: Reduced stiffness, smaller increments, stops on contact
- **SLIDE_SEARCH**: Very low stiffness, small increments, allows deviation from reference
- **PRESS**: Moderate stiffness, strong force limits, requires progress
- **INSERT**: Moderate stiffness, very small increments, strong force limits
- **RELEASE**: Low stiffness, faster tracking, very low force limits

## Key Components

### 1. Model (`models/IACT_B.py`)

The `IACT_B_VAE` model extends the vanilla ACT transformer architecture:

- **Shared Backbone**: Same ResNet backbone and transformer encoder/decoder as vanilla ACT
- **Dual Heads**: 
  - `action_head`: Outputs `q_ref` (joint positions)
  - `primitive_head`: Outputs primitive logits and parameters
- **VAE Latent**: Uses the same VAE latent space as vanilla ACT for action chunking

### 2. PrimitiveExecutor (`primitive_executor.py`)

The core component that modifies joint reference execution:

#### Execution Logic

**Step A - Choose Active Primitive:**
- Takes `argmax(z[t])` at the start of each chunk
- Holds the mode until termination (event/timeout triggers transition)

**Step B - Interpret q_ref Under That Mode:**

The same `q_ref` is executed differently:

| Primitive | Stiffness | Velocity Scale | Max Δq | Force Limit | Behavior |
|-----------|-----------|----------------|--------|-------------|----------|
| APPROACH | High (0.0) | 1.0 | 0.1 | ∞ | Normal tracking, large increments |
| GUARDED_MOVE | Reduced (0.5) | 0.5 | 0.05 | 15.0 Nm | Slower, smaller increments, stops on contact |
| SLIDE_SEARCH | Very Low (0.8) | 0.3 | 0.02 | 10.0 Nm | Very slow, allows deviation |
| PRESS | Moderate (0.3) | 0.4 | 0.03 | 20.0 Nm | Slow, strong force limit |
| INSERT | Moderate (0.4) | 0.3 | 0.02 | 25.0 Nm | Very slow, very strong force limit |
| RELEASE | Low (0.6) | 0.8 | 0.08 | 5.0 Nm | Faster, very low force limit |

#### Execution Modifications

1. **Velocity Scaling**: `q_cmd = q_current + (q_ref - q_current) * velocity_scale`
2. **Position Increment Limits**: Clamp `Δq` to `max_delta_q` per timestep
3. **Gain Scheduling**: If controller supports online gain changes:
   - `kp_scale = 1.0 - 0.9 * compliance_level`
   - `kd_scale = 1.0 - 0.5 * compliance_level`
4. **Safety Gating**: 
   - Stop (hold position) on JAM events
   - Transition on CONTACT_ON events (handled by policy)

#### Event Detection

The PrimitiveExecutor detects events from sensor measurements:

- **CONTACT_ON**: Torque spikes above threshold (`contact_torque_threshold`)
- **CONTACT_OFF**: Torque drops below threshold
- **SLIP**: High velocity when contact is expected (tangential motion + contact proxy)
- **JAM**: High force with low velocity (commanded motion stalls + force rises)

### 3. Data Labeling (`data_labeling.py`)

Weakly supervised labeling for training:

#### Auto-Segmentation

Each episode is automatically segmented into primitives using events:

1. **Event Detection**:
   ```python
   events = detect_events(qpos, qdot, q_ref, tau_meas)
   ```
   - CONTACT_ON when `||τ|| > threshold`
   - SLIP when `||qdot|| > threshold` and contact detected
   - JAM when `||τ|| > jam_threshold` and `||qdot|| < tolerance`

2. **Primitive Assignment**:
   ```python
   primitives = segment_primitives(events, qpos, qdot, tau_meas)
   ```
   - State machine transitions based on events
   - APPROACH → GUARDED_MOVE/PRESS on CONTACT_ON
   - Contact primitives → RELEASE on CONTACT_OFF
   - SLIP → SLIDE_SEARCH

3. **Parameter Extraction**:
   ```python
   primitive_params = extract_primitive_parameters(...)
   ```
   - `force_limit`: Based on measured torque (1.5x safety margin)
   - `compliance_level`: Primitive-dependent (0.0=stiff, 1.0=compliant)
   - `velocity_scale`: Based on current velocity magnitude
   - `max_delta_q`: Based on desired position change

#### Required Logged Signals

Per timestep:
- `qpos`: Joint positions
- `qdot`: Joint velocities  
- `q_ref`: Commanded joint positions (if available)
- `tau_meas`: Measured torques or motor currents
- `gripper_state`: Gripper state
- (optional) `ee_twist`: Estimated end-effector twist

### 4. Policy Wrapper (`policy.py`)

The `IACT_B_Policy` class wraps the model and handles training/inference:

#### Training

```python
loss_dict = policy(qpos, image, effort, actions, is_pad, 
                   primitive_labels, primitive_params, event_labels)
```

Losses:
- **L_joint**: `L1(q_ref, actions)` - vanilla ACT loss
- **L_prim**: `CrossEntropy(z, primitive_labels)` - primitive classification
- **L_param**: `MSE(θ, primitive_params)` - parameter regression
- **L_event**: (optional) Event prediction auxiliary task
- **L_kl**: KL divergence (VAE regularization)

Total: `L = L_joint + λ_kl * L_kl + λ_prim * L_prim + λ_param * L_param + λ_event * L_event`

#### Inference

```python
outputs = policy(qpos, image, effort)
# Returns: {'q_ref', 'primitive_logits', 'primitive_params', 'is_pad_hat'}

# Execute with primitive-aware modifications
executed = policy.execute_with_primitives(
    q_ref, primitive_logits, primitive_params,
    q_current, qdot_current, tau_meas
)
```

## Training

### Data Preparation

1. **Collect demonstrations** with logged signals:
   ```python
   # Per timestep:
   - observations (images, proprio, etc.)
   - qpos, qdot
   - q_ref (commanded positions)
   - tau_meas (torques/currents)
   - gripper_state
   ```

2. **Auto-label episodes**:
   ```python
   from IACT_B.data_labeling import label_episode
   
   labels = label_episode(
       qpos, qdot, q_ref, tau_meas,
       contact_torque_threshold=5.0,
       slip_velocity_threshold=0.1,
       jam_force_threshold=20.0,
       primitive_param_dim=8
   )
   # Returns: {'events', 'primitives', 'primitive_params'}
   ```

3. **Create dataset** with primitive labels and parameters

### Training Configuration

```python
args_override = {
    'kl_weight': 10.0,
    'num_primitives': 6,
    'primitive_param_dim': 8,
    'primitive_loss_weight': 1.0,  # Weight for L_prim
    'param_loss_weight': 0.1,      # Weight for L_param
    'event_loss_weight': 0.5,      # Weight for L_event (optional)
    'state_dim': 7,
    # ... other ACT args
}

policy = IACT_B_Policy(args_override)
```

**Important**: Weight `L_prim` enough that primitives don't collapse (default: 1.0).

## Runtime Execution

### Step-by-Step

1. **Policy Inference**:
   ```python
   outputs = policy(qpos, image, effort)
   q_ref = outputs['q_ref']  # (num_queries, num_dof)
   z = outputs['primitive_logits']  # (num_queries, num_primitives)
   θ = outputs['primitive_params']  # (num_queries, param_dim)
   ```

2. **Choose Active Primitive**:
   ```python
   primitive_id = torch.argmax(z[0])  # Use first query
   ```

3. **Execute with PrimitiveExecutor**:
   ```python
   executor.update_primitive(primitive_id, θ[0])
   result = executor.execute_joint_reference(
       q_ref[0], q_current, qdot_current, tau_meas
   )
   q_cmd = result['q_cmd']  # Modified joint command
   ```

4. **Publish to Robot**:
   - If controller supports gain scheduling: use `kp_scale`, `kd_scale`
   - Otherwise: use `q_cmd` with velocity scaling and position limits

### Integration Example

```python
from IACT_B import IACT_B_Policy, PrimitiveExecutor

# Initialize
policy = IACT_B_Policy(args_override)
executor = PrimitiveExecutor(num_dof=7)

# Control loop
for step in range(episode_length):
    # Get observations
    qpos, image, effort = get_observations()
    
    # Policy inference
    outputs = policy(qpos, image, effort)
    q_ref = outputs['q_ref'][0]  # First query
    z = outputs['primitive_logits'][0]
    θ = outputs['primitive_params'][0]
    
    # Choose primitive
    primitive_id = torch.argmax(z).item()
    executor.update_primitive(primitive_id, θ.detach().cpu().numpy())
    
    # Execute with primitive-aware modifications
    result = executor.execute_joint_reference(
        q_ref.detach().cpu().numpy(),
        qpos,
        qdot,
        tau_meas
    )
    
    # Publish command
    publish_joint_command(result['q_cmd'])
    
    # Handle events
    if result['event'] == EventType.JAM:
        # Safety stop or fallback behavior
        handle_jam()
```

## Differences from Vanilla ACT and IACT_A

### vs. Vanilla ACT

- **Vanilla ACT**: Single head → `q_ref` → direct position control
- **IACT_B**: Dual heads → `q_ref` + primitives → PrimitiveExecutor → modified execution

### vs. IACT_A

- **IACT_A**: Outputs only primitives → converts to impedance targets → impedance control
- **IACT_B**: Outputs both `q_ref` and primitives → modifies `q_ref` execution → position control with gain scheduling

IACT_B maintains compatibility with position-controlled robots while adding contact awareness.

## File Structure

```
IACT_B/
├── __init__.py              # Package exports
├── README.md                # This file
├── main.py                  # Model builder and optimizer
├── policy.py                # IACT_B_Policy wrapper
├── primitive_executor.py    # Primitive-aware execution
├── data_labeling.py         # Auto-segmentation utilities
└── models/
    ├── __init__.py
    └── IACT_B.py           # Dual-head model architecture
```

## Configuration Parameters

### Model
- `num_primitives`: Number of primitive types (default: 6)
- `primitive_param_dim`: Dimension of primitive parameters (default: 8)
- `state_dim`: Robot state dimension (default: 7)

### PrimitiveExecutor
- `default_stiffness`: Default stiffness (default: 100.0)
- `default_damping`: Default damping (default: 10.0)
- `contact_torque_threshold`: Torque threshold for contact (default: 5.0 Nm)
- `slip_velocity_threshold`: Velocity threshold for slip (default: 0.1 rad/s)
- `jam_force_threshold`: Force threshold for jam (default: 20.0 Nm)
- `timeout_steps`: Maximum steps before timeout (default: 500)

### Training
- `kl_weight`: KL divergence weight (default: 10.0)
- `primitive_loss_weight`: Primitive classification loss weight (default: 1.0)
- `param_loss_weight`: Parameter regression loss weight (default: 0.1)
- `event_loss_weight`: Event prediction loss weight (default: 0.5)

## Future Extensions

- **Per-axis force limits**: Extend parameters to support per-DOF force limits
- **Surface normal estimation**: Add contact axis/surface normal to parameters
- **Duration prediction**: Add termination logits for primitive duration
- **Event prediction head**: Explicit event prediction as auxiliary task
- **Multi-primitive execution**: Support for parallel primitives on different DOFs

## References

- ACT: Action Chunking with Transformers (Zhou et al.)
- Contact-aware manipulation primitives
- Impedance control and gain scheduling
