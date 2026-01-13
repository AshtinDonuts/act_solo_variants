"""
PrimitiveExecutor: Middle-level component that translates primitives to impedance controller targets
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class PrimitiveType(Enum):
    """Enumeration of primitive types"""
    MOVE = 0
    GRASP = 1
    RELEASE = 2
    INSERT = 3
    PULL = 4
    PUSH = 5
    ROTATE = 6
    HOLD = 7


class TerminationReason(Enum):
    """
    Termination reasons for primitives:
    - CONTACT: Contact detected
    - SLIP: Slip detected
    - JAM: Jam detected
    - TIMEOUT: Timeout reached
    - SUCCESS: Primitive completed successfully
    - NONE: Not terminated
    """
    CONTACT = "contact"
    SLIP = "slip"
    JAM = "jam"
    TIMEOUT = "timeout"
    SUCCESS = "success"
    NONE = "none"


class PrimitiveExecutor:
    """
    Middle-level component (outside ACT) that:
    1. Holds current primitive state
    2. Translates (z, θ) → controller targets: impedance (K, D, x_ref / v_ref)
    3. Evaluates termination: contact detected / slip / jam / timeout
    """
    
    def __init__(self, 
                 num_dof: int = 7,
                 default_stiffness: float = 100.0,
                 default_damping: float = 10.0,
                 contact_force_threshold: float = 5.0,
                 slip_velocity_threshold: float = 0.1,
                 jam_force_threshold: float = 20.0,
                 timeout_steps: int = 500):
        """
        Args:
            num_dof: Number of degrees of freedom
            default_stiffness: Default stiffness K (N/m or Nm/rad)
            default_damping: Default damping D (N·s/m or Nm·s/rad)
            contact_force_threshold: Force threshold for contact detection (N)
            slip_velocity_threshold: Velocity threshold for slip detection (m/s)
            jam_force_threshold: Force threshold for jam detection (N)
            timeout_steps: Maximum steps before timeout
        """
        self.num_dof = num_dof
        self.default_stiffness = default_stiffness
        self.default_damping = default_damping
        self.contact_force_threshold = contact_force_threshold
        self.slip_velocity_threshold = slip_velocity_threshold
        self.jam_force_threshold = jam_force_threshold
        self.timeout_steps = timeout_steps
        
        # Current primitive state
        self.current_primitive_id: Optional[int] = None
        self.current_primitive_params: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.start_position: Optional[np.ndarray] = None
        
        # Primitive-specific impedance parameters
        self.primitive_stiffness_map = {
            PrimitiveType.MOVE.value: default_stiffness,
            PrimitiveType.GRASP.value: default_stiffness * 2.0,  # Higher stiffness for grasping
            PrimitiveType.RELEASE.value: default_stiffness * 0.5,  # Lower stiffness for release
            PrimitiveType.INSERT.value: default_stiffness * 1.5,
            PrimitiveType.PULL.value: default_stiffness * 1.2,
            PrimitiveType.PUSH.value: default_stiffness * 1.2,
            PrimitiveType.ROTATE.value: default_stiffness * 0.8,
            PrimitiveType.HOLD.value: default_stiffness * 2.0,
        }
        
        self.primitive_damping_map = {
            PrimitiveType.MOVE.value: default_damping,
            PrimitiveType.GRASP.value: default_damping * 1.5,
            PrimitiveType.RELEASE.value: default_damping * 0.5,
            PrimitiveType.INSERT.value: default_damping * 1.2,
            PrimitiveType.PULL.value: default_damping * 1.0,
            PrimitiveType.PUSH.value: default_damping * 1.0,
            PrimitiveType.ROTATE.value: default_damping * 0.8,
            PrimitiveType.HOLD.value: default_damping * 1.5,
        }
    
    def update_primitive(self, primitive_id: int, primitive_params: np.ndarray):
        """
        Update the current primitive being executed.
        
        Args:
            primitive_id: ID of the primitive (from p(z) distribution)
            primitive_params: Parameters θ for the primitive
        """
        self.current_primitive_id = primitive_id
        self.current_primitive_params = primitive_params.copy()
        self.step_count = 0
    
    def get_impedance_targets(self, 
                             current_position: np.ndarray,
                             current_velocity: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Translate (z, θ) → controller targets: impedance (K, D, x_ref / v_ref)
        
        Args:
            current_position: Current joint/end-effector position (num_dof,)
            current_velocity: Current joint/end-effector velocity (num_dof,)
            
        Returns:
            Dictionary with keys:
                - 'K': Stiffness matrix (num_dof, num_dof) or diagonal (num_dof,)
                - 'D': Damping matrix (num_dof, num_dof) or diagonal (num_dof,)
                - 'x_ref': Reference position (num_dof,)
                - 'v_ref': Reference velocity (num_dof,)
        """
        if self.current_primitive_id is None or self.current_primitive_params is None:
            # Default: hold current position
            K = np.eye(self.num_dof) * self.default_stiffness
            D = np.eye(self.num_dof) * self.default_damping
            x_ref = current_position.copy()
            v_ref = np.zeros(self.num_dof)
            return {'K': K, 'D': D, 'x_ref': x_ref, 'v_ref': v_ref}
        
        # Get primitive-specific stiffness and damping
        K_base = self.primitive_stiffness_map.get(
            self.current_primitive_id, 
            self.default_stiffness
        )
        D_base = self.primitive_damping_map.get(
            self.current_primitive_id,
            self.default_damping
        )
        
        # Create diagonal impedance matrices
        # Allow per-DOF stiffness/damping from params if available
        if len(self.current_primitive_params) >= self.num_dof * 2:
            # First num_dof params are stiffness, next num_dof are damping
            K_diag = self.current_primitive_params[:self.num_dof] * K_base
            D_diag = self.current_primitive_params[self.num_dof:2*self.num_dof] * D_base
        else:
            K_diag = np.ones(self.num_dof) * K_base
            D_diag = np.ones(self.num_dof) * D_base
        
        K = np.diag(K_diag)
        D = np.diag(D_diag)
        
        # Extract reference position and velocity from params
        # Assuming params structure: [K_diag (num_dof), D_diag (num_dof), x_ref (num_dof), v_ref (num_dof), ...]
        param_offset = min(2 * self.num_dof, len(self.current_primitive_params))
        
        if len(self.current_primitive_params) >= param_offset + self.num_dof:
            x_ref = self.current_primitive_params[param_offset:param_offset + self.num_dof]
        else:
            # Default: move towards target position encoded in params
            if len(self.current_primitive_params) >= self.num_dof:
                x_ref = self.current_primitive_params[:self.num_dof]
            else:
                x_ref = current_position.copy()
        
        if len(self.current_primitive_params) >= param_offset + 2 * self.num_dof:
            v_ref = self.current_primitive_params[param_offset + self.num_dof:param_offset + 2 * self.num_dof]
        else:
            v_ref = np.zeros(self.num_dof)
        
        # Clamp reference values to reasonable ranges
        x_ref = np.clip(x_ref, -np.inf, np.inf)  # Can be adjusted based on joint limits
        v_ref = np.clip(v_ref, -1.0, 1.0)  # Reasonable velocity limits
        
        return {'K': K, 'D': D, 'x_ref': x_ref, 'v_ref': v_ref}
    
    def evaluate_termination(self,
                            current_position: np.ndarray,
                            current_velocity: np.ndarray,
                            current_force: np.ndarray,
                            target_position: Optional[np.ndarray] = None) -> Tuple[bool, TerminationReason]:
        """
        Evaluate termination conditions: contact / slip / jam / timeout
        
        Args:
            current_position: Current joint/end-effector position
            current_velocity: Current joint/end-effector velocity
            current_force: Current force/torque measurements
            target_position: Target position (optional, for success detection)
            
        Returns:
            Tuple of (should_terminate, termination_reason)
        """
        #  Assume target pos not known in this setup.
        if target_position is not None: raise NotImplementedError  # DEBUG

        if self.current_primitive_id is None:
            return False, TerminationReason.NONE
        
        self.step_count += 1
        
        # Check timeout
        if self.step_count >= self.timeout_steps:
            return True, TerminationReason.TIMEOUT
        
        # Check contact detection (force exceeds threshold)
        force_magnitude = np.linalg.norm(current_force)
        if force_magnitude > self.contact_force_threshold:
            # For some primitives, contact is expected (e.g., GRASP, INSERT)
            if self.current_primitive_id in [PrimitiveType.GRASP.value, 
                                            PrimitiveType.INSERT.value,
                                            PrimitiveType.PULL.value,
                                            PrimitiveType.PUSH.value]:
                # Check if we've reached target (success)
                if target_position is not None:
                    position_error = np.linalg.norm(current_position - target_position)
                    if position_error < 0.01:  # 1cm tolerance
                        return True, TerminationReason.SUCCESS
                # Otherwise, contact detected but not at target yet
                return False, TerminationReason.NONE
            else:
                # For other primitives, unexpected contact might indicate jam
                if force_magnitude > self.jam_force_threshold:
                    return True, TerminationReason.JAM
                return True, TerminationReason.CONTACT
        
        # Check slip detection (high velocity when should be stationary)
        velocity_magnitude = np.linalg.norm(current_velocity)
        if self.current_primitive_id == PrimitiveType.GRASP.value:
            # During grasp, high velocity might indicate slip
            if velocity_magnitude > self.slip_velocity_threshold:
                return True, TerminationReason.SLIP
        
        # Check jam detection (high force with low velocity)
        JAM_VELOCITY_TOLERANCE = 0.01  # TODO: softcode
        if force_magnitude > self.jam_force_threshold and velocity_magnitude < JAM_VELOCITY_TOLERANCE:
            return True, TerminationReason.JAM
        
        # Check success (reached target position)
        if target_position is not None:
            position_error = np.linalg.norm(current_position - target_position)
            if position_error < 0.01:  # 1cm tolerance
                return True, TerminationReason.SUCCESS
        
        return False, TerminationReason.NONE
    
    def reset(self):
        """Reset the primitive executor state"""
        self.current_primitive_id = None
        self.current_primitive_params = None
        self.step_count = 0
        self.start_position = None
