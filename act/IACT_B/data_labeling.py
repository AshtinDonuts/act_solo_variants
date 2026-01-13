"""
Data labeling utilities for auto-segmentation of primitives using events.

Auto-segment each episode into primitives using events:
- CONTACT_ON when torque residual / current spikes above threshold
- SLIP when tangential motion + shear proxy
- JAM when commanded motion persists but actual progress stalls + force rises
- CONTACT_OFF when residual drops

Then assign primitive IDs over time:
- APPROACH
- GUARDED_MOVE
- SLIDE_SEARCH
- PRESS
- INSERT
- RELEASE/RETREAT
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class PrimitiveLabel(IntEnum):
    """Primitive labels for weakly supervised learning"""
    APPROACH = 0
    GUARDED_MOVE = 1
    SLIDE_SEARCH = 2
    PRESS = 3
    INSERT = 4
    RELEASE = 5


class EventLabel(IntEnum):
    """Event labels for auxiliary task"""
    NONE = 0
    CONTACT_ON = 1
    CONTACT_OFF = 2
    SLIP = 3
    JAM = 4


def detect_events(qpos: np.ndarray,
                  qdot: np.ndarray,
                  q_ref: np.ndarray,
                  tau_meas: np.ndarray,
                  contact_torque_threshold: float = 5.0,
                  slip_velocity_threshold: float = 0.1,
                  jam_force_threshold: float = 20.0,
                  jam_velocity_tolerance: float = 0.01) -> np.ndarray:
    """
    Detect events from logged signals.
    
    Args:
        qpos: Joint positions (T, num_dof)
        qdot: Joint velocities (T, num_dof)
        q_ref: Commanded joint positions (T, num_dof)
        tau_meas: Measured torques (T, num_dof)
        contact_torque_threshold: Torque threshold for contact (Nm)
        slip_velocity_threshold: Velocity threshold for slip (rad/s)
        jam_force_threshold: Force threshold for jam (Nm)
        jam_velocity_tolerance: Velocity tolerance for jam detection (rad/s)
        
    Returns:
        Event labels (T,) as EventLabel enum values
    """
    T = qpos.shape[0]
    events = np.zeros(T, dtype=np.int32)
    
    for t in range(T):
        tau_mag = np.linalg.norm(tau_meas[t])
        qdot_mag = np.linalg.norm(qdot[t])
        
        # CONTACT_ON: torque spikes above threshold
        if tau_mag > contact_torque_threshold:
            # Check if this is a new contact (previous was below threshold)
            if t > 0 and np.linalg.norm(tau_meas[t-1]) < contact_torque_threshold * 0.5:
                events[t] = EventLabel.CONTACT_ON
            # Check for JAM: high force + low velocity
            elif tau_mag > jam_force_threshold and qdot_mag < jam_velocity_tolerance:
                events[t] = EventLabel.JAM
            # Otherwise, maintain contact state
            elif t > 0 and events[t-1] == EventLabel.CONTACT_ON:
                events[t] = EventLabel.CONTACT_ON
            else:
                events[t] = EventLabel.CONTACT_ON
        # CONTACT_OFF: torque drops below threshold
        elif t > 0 and np.linalg.norm(tau_meas[t-1]) > contact_torque_threshold * 0.5:
            events[t] = EventLabel.CONTACT_OFF
        # SLIP: tangential motion + contact proxy
        elif tau_mag > contact_torque_threshold * 0.5 and qdot_mag > slip_velocity_threshold:
            events[t] = EventLabel.SLIP
        else:
            events[t] = EventLabel.NONE
    
    return events


def segment_primitives(events: np.ndarray,
                      qpos: np.ndarray,
                      qdot: np.ndarray,
                      tau_meas: np.ndarray) -> np.ndarray:
    """
    Auto-segment episode into primitives using events.
    
    Args:
        events: Event labels (T,)
        qpos: Joint positions (T, num_dof)
        qdot: Joint velocities (T, num_dof)
        tau_meas: Measured torques (T, num_dof)
        
    Returns:
        Primitive labels (T,) as PrimitiveLabel enum values
    """
    T = events.shape[0]
    primitives = np.zeros(T, dtype=np.int32)
    
    # State machine for primitive assignment
    in_contact = False
    current_primitive = PrimitiveLabel.APPROACH
    
    for t in range(T):
        event = events[t]
        tau_mag = np.linalg.norm(tau_meas[t])
        qdot_mag = np.linalg.norm(qdot[t])
        
        # State transitions based on events
        if event == EventLabel.CONTACT_ON and not in_contact:
            # Transition from APPROACH to GUARDED_MOVE or PRESS
            in_contact = True
            # If high force, likely PRESS/INSERT
            if tau_mag > 15.0:
                current_primitive = PrimitiveLabel.PRESS
            else:
                current_primitive = PrimitiveLabel.GUARDED_MOVE
        elif event == EventLabel.CONTACT_OFF and in_contact:
            # Transition from contact to RELEASE
            in_contact = False
            current_primitive = PrimitiveLabel.RELEASE
        elif event == EventLabel.SLIP:
            # Transition to SLIDE_SEARCH
            current_primitive = PrimitiveLabel.SLIDE_SEARCH
        elif event == EventLabel.JAM:
            # Could be INSERT getting stuck, or need to transition
            if current_primitive == PrimitiveLabel.PRESS:
                current_primitive = PrimitiveLabel.INSERT
            # Otherwise maintain current primitive
        elif not in_contact and current_primitive != PrimitiveLabel.APPROACH:
            # Reset to APPROACH if not in contact
            current_primitive = PrimitiveLabel.APPROACH
        
        primitives[t] = current_primitive.value
    
    return primitives


def extract_primitive_parameters(qpos: np.ndarray,
                                qdot: np.ndarray,
                                q_ref: np.ndarray,
                                tau_meas: np.ndarray,
                                primitives: np.ndarray,
                                primitive_param_dim: int = 8) -> np.ndarray:
    """
    Extract primitive parameters from logged data.
    
    At minimum:
    - force_limit (scalar or per-axis)
    - compliance_level (scalar mapped to gains)
    - contact_axis or surface_normal (optional)
    - duration or termination logits
    
    Args:
        qpos: Joint positions (T, num_dof)
        qdot: Joint velocities (T, num_dof)
        q_ref: Commanded joint positions (T, num_dof)
        tau_meas: Measured torques (T, num_dof)
        primitives: Primitive labels (T,)
        primitive_param_dim: Dimension of primitive parameters
        
    Returns:
        Primitive parameters (T, primitive_param_dim)
    """
    T, num_dof = qpos.shape
    params = np.zeros((T, primitive_param_dim))
    
    for t in range(T):
        primitive = primitives[t]
        tau_mag = np.linalg.norm(tau_meas[t])
        
        # Parameter structure: [force_limit, compliance_level, velocity_scale, max_delta_q, ...]
        # force_limit: based on measured torque
        params[t, 0] = np.clip(tau_mag * 1.5, 0.1, 50.0)  # 1.5x safety margin
        
        # compliance_level: primitive-dependent
        if primitive == PrimitiveLabel.APPROACH:
            params[t, 1] = 0.0  # High stiffness
        elif primitive == PrimitiveLabel.GUARDED_MOVE:
            params[t, 1] = 0.5  # Reduced stiffness
        elif primitive == PrimitiveLabel.SLIDE_SEARCH:
            params[t, 1] = 0.8  # Very low stiffness
        elif primitive == PrimitiveLabel.PRESS:
            params[t, 1] = 0.3  # Moderate stiffness
        elif primitive == PrimitiveLabel.INSERT:
            params[t, 1] = 0.4  # Moderate stiffness
        elif primitive == PrimitiveLabel.RELEASE:
            params[t, 1] = 0.6  # Low stiffness
        
        # velocity_scale: based on current velocity magnitude
        qdot_mag = np.linalg.norm(qdot[t])
        params[t, 2] = np.clip(1.0 - qdot_mag * 0.5, 0.1, 1.0)
        
        # max_delta_q: based on desired change
        if t > 0:
            delta_q = np.linalg.norm(q_ref[t] - qpos[t-1])
            params[t, 3] = np.clip(delta_q, 0.01, 0.2)
        else:
            params[t, 3] = 0.05  # Default
        
        # Remaining parameters can be zeros or additional features
        # (e.g., contact axis, surface normal, duration estimate)
        for i in range(4, primitive_param_dim):
            params[t, i] = 0.0
    
    return params


def label_episode(qpos: np.ndarray,
                  qdot: np.ndarray,
                  q_ref: np.ndarray,
                  tau_meas: np.ndarray,
                  contact_torque_threshold: float = 5.0,
                  slip_velocity_threshold: float = 0.1,
                  jam_force_threshold: float = 20.0,
                  primitive_param_dim: int = 8) -> Dict[str, np.ndarray]:
    """
    Label an entire episode with primitives and parameters.
    
    Args:
        qpos: Joint positions (T, num_dof)
        qdot: Joint velocities (T, num_dof)
        q_ref: Commanded joint positions (T, num_dof)
        tau_meas: Measured torques (T, num_dof)
        contact_torque_threshold: Torque threshold for contact
        slip_velocity_threshold: Velocity threshold for slip
        jam_force_threshold: Force threshold for jam
        primitive_param_dim: Dimension of primitive parameters
        
    Returns:
        Dictionary with keys:
            - 'events': Event labels (T,)
            - 'primitives': Primitive labels (T,)
            - 'primitive_params': Primitive parameters (T, primitive_param_dim)
    """
    # Detect events
    events = detect_events(
        qpos, qdot, q_ref, tau_meas,
        contact_torque_threshold=contact_torque_threshold,
        slip_velocity_threshold=slip_velocity_threshold,
        jam_force_threshold=jam_force_threshold
    )
    
    # Segment into primitives
    primitives = segment_primitives(events, qpos, qdot, tau_meas)
    
    # Extract primitive parameters
    primitive_params = extract_primitive_parameters(
        qpos, qdot, q_ref, tau_meas, primitives,
        primitive_param_dim=primitive_param_dim
    )
    
    return {
        'events': events,
        'primitives': primitives,
        'primitive_params': primitive_params
    }
