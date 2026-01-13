"""
PrimitiveExecutor: Modifies how joint references are executed based on contact mode.

The same joint reference (q_ref) can be executed differently depending on the primitive:
- APPROACH: high stiffness, normal tracking
- GUARDED_MOVE: reduced stiffness, smaller Δq, stop on contact
- SLIDE_SEARCH: very low stiffness, small Δq, allow deviation
- PRESS/INSERT: moderate stiffness, force safety clamp, require progress
"""
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum


class PrimitiveType(Enum):
    """Primitive types for contact-aware execution"""
    APPROACH = 0
    GUARDED_MOVE = 1
    SLIDE_SEARCH = 2
    PRESS = 3
    INSERT = 4
    RELEASE = 5


class EventType(Enum):
    """Event types for primitive transitions"""
    CONTACT_ON = "contact_on"
    CONTACT_OFF = "contact_off"
    SLIP = "slip"
    JAM = "jam"
    NONE = "none"


class PrimitiveExecutor:
    """
    Executes joint references with primitive-aware gain scheduling and safety gating.
    
    The same q_ref can be executed differently based on the active primitive:
    - Velocity scaling
    - Position increment limits (Δq clamping)
    - Gain scheduling (if controller supports it)
    - Safety reflexes (stop/retreat on events)
    """
    
    def __init__(self,
                 num_dof: int = 7,
                 default_stiffness: float = 100.0,
                 default_damping: float = 10.0,
                 contact_torque_threshold: float = 5.0,
                 slip_velocity_threshold: float = 0.1,
                 jam_force_threshold: float = 20.0,
                 timeout_steps: int = 500):
        """
        Args:
            num_dof: Number of degrees of freedom
            default_stiffness: Default stiffness (for gain scheduling if supported)
            default_damping: Default damping (for gain scheduling if supported)
            contact_torque_threshold: Torque threshold for CONTACT_ON event (Nm)
            slip_velocity_threshold: Velocity threshold for SLIP event (rad/s)
            jam_force_threshold: Force threshold for JAM event (N or Nm)
            timeout_steps: Maximum steps before timeout
        """
        self.num_dof = num_dof
        self.default_stiffness = default_stiffness
        self.default_damping = default_damping
        self.contact_torque_threshold = contact_torque_threshold
        self.slip_velocity_threshold = slip_velocity_threshold
        self.jam_force_threshold = jam_force_threshold
        self.timeout_steps = timeout_steps
        
        # Current primitive state
        self.current_primitive_id: Optional[int] = None
        self.current_primitive_params: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.last_q_ref: Optional[np.ndarray] = None
        
        # Primitive-specific execution parameters
        self._setup_primitive_configs()
    
    def _setup_primitive_configs(self):
        """Setup primitive-specific execution parameters"""
        # Compliance level: 0.0 = stiff, 1.0 = compliant
        # Force limit: maximum allowed force/torque
        # Velocity scale: scaling factor for commanded velocity
        # Max delta_q: maximum position increment per step
        
        self.primitive_configs = {
            PrimitiveType.APPROACH.value: {
                'compliance_level': 0.0,  # High stiffness
                'velocity_scale': 1.0,    # Normal tracking
                'max_delta_q': 0.1,       # Large increments allowed
                'force_limit': np.inf,     # No force limit
            },
            PrimitiveType.GUARDED_MOVE.value: {
                'compliance_level': 0.5,  # Reduced stiffness
                'velocity_scale': 0.5,    # Slower tracking
                'max_delta_q': 0.05,      # Smaller increments
                'force_limit': 15.0,      # Moderate force limit
            },
            PrimitiveType.SLIDE_SEARCH.value: {
                'compliance_level': 0.8,  # Very low stiffness
                'velocity_scale': 0.3,    # Slow tracking
                'max_delta_q': 0.02,     # Very small increments
                'force_limit': 10.0,     # Low force limit
            },
            PrimitiveType.PRESS.value: {
                'compliance_level': 0.3,  # Moderate stiffness
                'velocity_scale': 0.4,    # Slow tracking
                'max_delta_q': 0.03,      # Small increments
                'force_limit': 20.0,      # Strong force limit
            },
            PrimitiveType.INSERT.value: {
                'compliance_level': 0.4,  # Moderate stiffness
                'velocity_scale': 0.3,    # Slow tracking
                'max_delta_q': 0.02,      # Very small increments
                'force_limit': 25.0,      # Strong force limit
            },
            PrimitiveType.RELEASE.value: {
                'compliance_level': 0.6,  # Low stiffness
                'velocity_scale': 0.8,    # Faster tracking
                'max_delta_q': 0.08,      # Moderate increments
                'force_limit': 5.0,       # Very low force limit
            },
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
    
    def execute_joint_reference(self,
                               q_ref: np.ndarray,
                               q_current: np.ndarray,
                               qdot_current: np.ndarray,
                               tau_meas: Optional[np.ndarray] = None) -> Dict:
        """
        Execute joint reference with primitive-aware modifications.
        
        Args:
            q_ref: Desired joint positions from ACT (num_dof,)
            q_current: Current joint positions (num_dof,)
            qdot_current: Current joint velocities (num_dof,)
            tau_meas: Measured torques/currents (num_dof,) for event detection
            
        Returns:
            Dictionary with keys:
                - 'q_cmd': Executed joint command (num_dof,)
                - 'velocity_scale': Velocity scaling factor
                - 'compliance_level': Compliance level (0.0=stiff, 1.0=compliant)
                - 'kp_scale': Stiffness scaling factor (if controller supports gain scheduling)
                - 'kd_scale': Damping scaling factor (if controller supports gain scheduling)
                - 'event': Detected event type
        """
        # Default: use q_ref as-is
        if self.current_primitive_id is None:
            return {
                'q_cmd': q_ref.copy(),
                'velocity_scale': 1.0,
                'compliance_level': 0.0,
                'kp_scale': 1.0,
                'kd_scale': 1.0,
                'event': EventType.NONE
            }
        
        # Get primitive configuration
        config = self.primitive_configs.get(
            self.current_primitive_id,
            self.primitive_configs[PrimitiveType.APPROACH.value]
        )
        
        # Extract parameters from primitive_params if available
        # Expected structure: [force_limit, compliance_level, velocity_scale, max_delta_q, ...]
        if self.current_primitive_params is not None and len(self.current_primitive_params) >= 4:
            force_limit = float(self.current_primitive_params[0])
            compliance_override = float(self.current_primitive_params[1])
            velocity_scale_override = float(self.current_primitive_params[2])
            max_delta_q_override = float(self.current_primitive_params[3])
            
            # Clamp to reasonable ranges
            force_limit = np.clip(force_limit, 0.1, 50.0)
            compliance_override = np.clip(compliance_override, 0.0, 1.0)
            velocity_scale_override = np.clip(velocity_scale_override, 0.1, 1.0)
            max_delta_q_override = np.clip(max_delta_q_override, 0.01, 0.2)
            
            # Override defaults with learned parameters
            config = config.copy()
            config['force_limit'] = force_limit
            config['compliance_level'] = compliance_override
            config['velocity_scale'] = velocity_scale_override
            config['max_delta_q'] = max_delta_q_override
        
        # Compute desired change
        delta_q_desired = q_ref - q_current
        
        # Apply max_delta_q limit
        delta_q_magnitude = np.linalg.norm(delta_q_desired)
        if delta_q_magnitude > config['max_delta_q']:
            delta_q_desired = delta_q_desired * (config['max_delta_q'] / delta_q_magnitude)
        
        # Apply velocity scaling
        delta_q_desired = delta_q_desired * config['velocity_scale']
        
        # Compute executed command
        q_cmd = q_current + delta_q_desired
        
        # Detect events
        event = self._detect_event(q_current, qdot_current, tau_meas, config)
        
        # Safety gating: stop or retreat on critical events
        if event == EventType.JAM:
            # Stop: hold current position
            q_cmd = q_current.copy()
        elif event == EventType.CONTACT_ON and self.current_primitive_id == PrimitiveType.APPROACH.value:
            # Transition from APPROACH to GUARDED_MOVE (handled by policy)
            pass
        
        # Compute gain scaling factors (for controllers that support online gain scheduling)
        # compliance_level: 0.0 = stiff (kp=1.0), 1.0 = compliant (kp=0.1)
        kp_scale = 1.0 - 0.9 * config['compliance_level']
        kd_scale = 1.0 - 0.5 * config['compliance_level']
        
        self.step_count += 1
        self.last_q_ref = q_ref.copy()
        
        return {
            'q_cmd': q_cmd,
            'velocity_scale': config['velocity_scale'],
            'compliance_level': config['compliance_level'],
            'kp_scale': kp_scale,
            'kd_scale': kd_scale,
            'event': event
        }
    
    def _detect_event(self,
                     q_current: np.ndarray,
                     qdot_current: np.ndarray,
                     tau_meas: Optional[np.ndarray],
                     config: Dict) -> EventType:
        """
        Detect events based on current state and measurements.
        
        Args:
            q_current: Current joint positions
            qdot_current: Current joint velocities
            tau_meas: Measured torques
            config: Primitive configuration
            
        Returns:
            Detected event type
        """
        if tau_meas is None:
            return EventType.NONE
        
        # Check timeout
        if self.step_count >= self.timeout_steps:
            return EventType.JAM  # Treat timeout as jam
        
        # Compute torque magnitude
        tau_magnitude = np.linalg.norm(tau_meas)
        
        # CONTACT_ON: torque spikes above threshold
        if tau_magnitude > self.contact_torque_threshold:
            # Check if this is expected contact (for PRESS, INSERT)
            if self.current_primitive_id in [PrimitiveType.PRESS.value, PrimitiveType.INSERT.value]:
                # Expected contact, but check for jam
                if tau_magnitude > self.jam_force_threshold:
                    return EventType.JAM
                return EventType.CONTACT_ON
            else:
                # Unexpected contact
                return EventType.CONTACT_ON
        
        # CONTACT_OFF: torque drops below threshold (if we were in contact)
        if tau_magnitude < self.contact_torque_threshold * 0.5:
            # Could be contact off, but we don't track previous state here
            # This would be handled by the policy/state machine
            pass
        
        # SLIP: tangential motion + shear proxy
        # Simplified: high velocity when we expect low velocity (for contact primitives)
        velocity_magnitude = np.linalg.norm(qdot_current)
        if self.current_primitive_id in [PrimitiveType.PRESS.value, PrimitiveType.INSERT.value]:
            if velocity_magnitude > self.slip_velocity_threshold and tau_magnitude > self.contact_torque_threshold * 0.5:
                return EventType.SLIP
        
        # JAM: commanded motion persists but actual progress stalls + force rises
        if self.last_q_ref is not None:
            desired_progress = np.linalg.norm(q_current - self.last_q_ref)
            if desired_progress < 0.01 and tau_magnitude > self.jam_force_threshold:
                return EventType.JAM
        
        return EventType.NONE
    
    def reset(self):
        """Reset the primitive executor state"""
        self.current_primitive_id = None
        self.current_primitive_params = None
        self.step_count = 0
        self.last_q_ref = None
