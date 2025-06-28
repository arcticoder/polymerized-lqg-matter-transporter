"""
Multi-Variable PID Controller for Enhanced Stargate Transporter

This module implements multi-variable PID control with cross-coupling compensation
for coordinated Einstein tensor field regulation.

Mathematical Framework:
    H_PID(t) = ∫_V [K_P·e(t) + K_I·∫e(τ)dτ + K_D·ė(t)] dV
    
Where e(t) = G_μν(x,t) - G_μν^target is the Einstein tensor error field.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
import warnings

# Import our enhanced transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter

class MultiVarPIDController:
    """
    Multi-variable PID controller with cross-coupling compensation.
    
    Provides coordinated control of multiple Einstein tensor components
    with anti-windup, derivative filtering, and gain scheduling.
    """
    
    def __init__(self, transporter: EnhancedStargateTransporter,
                 pid_gains: Optional[Dict] = None,
                 coupling_matrix: Optional[jnp.ndarray] = None,
                 control_config: Optional[Dict] = None):
        """
        Initialize multi-variable PID controller.
        
        Args:
            transporter: Enhanced stargate transporter instance
            pid_gains: PID gain matrices {Kp, Ki, Kd}
            coupling_matrix: Cross-coupling compensation matrix
            control_config: Additional control configuration
        """
        self.transporter = transporter
        
        # System dimensions
        self.n_states = 16  # 4x4 Einstein tensor components
        self.n_controls = 4  # Control actuator dimensions
        
        # Default PID gains (tuned for Einstein tensor control)
        if pid_gains is None:
            pid_gains = self._design_default_gains()
            
        self.Kp = jnp.array(pid_gains['Kp'])  # Proportional gains
        self.Ki = jnp.array(pid_gains['Ki'])  # Integral gains  
        self.Kd = jnp.array(pid_gains['Kd'])  # Derivative gains
        
        # Cross-coupling compensation matrix
        if coupling_matrix is None:
            coupling_matrix = self._identify_coupling_matrix()
        self.coupling_matrix = jnp.array(coupling_matrix)
        
        # Control configuration
        if control_config is None:
            control_config = {
                'dt': 0.01,                    # Sampling time
                'integral_limit': 1e6,        # Anti-windup limit
                'derivative_filter_tau': 0.1,  # Derivative filter time constant
                'gain_schedule_enable': True,   # Enable adaptive gains
                'deadband': 1e-9              # Control deadband
            }
        self.config = control_config
        
        # Controller state variables
        self.integral_state = jnp.zeros((self.n_states,))
        self.previous_error = jnp.zeros((self.n_states,))
        self.filtered_derivative = jnp.zeros((self.n_states,))
        self.time_history = []
        
        # Performance tracking
        self.error_history = []
        self.control_history = []
        self.gain_history = []
        
        # Gain scheduling parameters
        self.nominal_gains = {
            'Kp': self.Kp.copy(),
            'Ki': self.Ki.copy(), 
            'Kd': self.Kd.copy()
        }
        
        print(f"MultiVarPIDController initialized:")
        print(f"  System dimensions: {self.n_states} states, {self.n_controls} controls")
        print(f"  Proportional gains: {jnp.linalg.norm(self.Kp):.2e}")
        print(f"  Integral gains: {jnp.linalg.norm(self.Ki):.2e}")
        print(f"  Derivative gains: {jnp.linalg.norm(self.Kd):.2e}")
        print(f"  Cross-coupling compensation: {'✅ Enabled' if jnp.linalg.norm(self.coupling_matrix) > 1e-12 else '❌ Disabled'}")
        
    def _design_default_gains(self) -> Dict:
        """Design default PID gains using Ziegler-Nichols-like tuning."""
        
        # Critical gain estimation (simplified)
        # In practice, would use system identification
        
        # Proportional gains (scaled by Einstein tensor magnitudes)
        Kp_diag = np.array([1e-3, 5e-4, 5e-4, 1e-3,  # g_00, g_01, g_02, g_03
                           5e-4, 1e-3, 5e-4, 5e-4,   # g_10, g_11, g_12, g_13
                           5e-4, 5e-4, 1e-3, 5e-4,   # g_20, g_21, g_22, g_23
                           1e-3, 5e-4, 5e-4, 1e-3])  # g_30, g_31, g_32, g_33
        
        Kp = np.diag(Kp_diag[:self.n_controls])  # Control-to-state mapping
        
        # Integral gains (Ti = 2.2 * Tp for Ziegler-Nichols)
        Ki = Kp / 2.2
        
        # Derivative gains (Td = 0.6 * Tp for Ziegler-Nichols)
        Kd = Kp * 0.6
        
        return {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
    
    def _identify_coupling_matrix(self) -> jnp.ndarray:
        """Identify cross-coupling between Einstein tensor components."""
        
        # Simplified coupling identification
        # In practice, would use system identification experiments
        
        n = self.n_controls
        coupling = np.zeros((n, n))
        
        # Main diagonal (no self-coupling)
        for i in range(n):
            coupling[i, i] = 1.0
            
        # Off-diagonal coupling (geometric coupling in Einstein tensor)
        coupling_strength = 0.15  # 15% cross-coupling
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Geometric coupling based on tensor structure
                    if (i // 2) == (j // 2):  # Same spatial dimension
                        coupling[i, j] = coupling_strength
                    else:  # Different dimensions
                        coupling[i, j] = coupling_strength * 0.5
                        
        return coupling
    
    @jit 
    def _apply_gain_scheduling(self, error: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply adaptive gain scheduling based on operating conditions."""
        
        # Error magnitude-based scheduling
        error_norm = jnp.linalg.norm(error)
        
        # Scheduling function: higher gains for larger errors
        schedule_factor = jnp.where(
            error_norm > 1e-6,
            1.0 + 2.0 * jnp.tanh(error_norm * 1e6),  # Increase gains for large errors
            1.0  # Nominal gains for small errors
        )
        
        # Apply scheduling
        Kp_scheduled = self.Kp * schedule_factor
        Ki_scheduled = self.Ki * schedule_factor
        Kd_scheduled = self.Kd * schedule_factor
        
        return Kp_scheduled, Ki_scheduled, Kd_scheduled
    
    @jit
    def _apply_anti_windup(self, integral_candidate: jnp.ndarray) -> jnp.ndarray:
        """Apply anti-windup protection to integral terms."""
        
        limit = self.config['integral_limit']
        
        # Clamp integral states to prevent windup
        integral_limited = jnp.clip(integral_candidate, -limit, limit)
        
        return integral_limited
    
    @jit
    def _filter_derivative(self, derivative_raw: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Apply low-pass filtering to derivative term."""
        
        tau = self.config['derivative_filter_tau']
        alpha = dt / (tau + dt)  # Filter coefficient
        
        # First-order low-pass filter
        derivative_filtered = alpha * derivative_raw + (1 - alpha) * self.filtered_derivative
        
        return derivative_filtered
    
    @jit
    def compute_pid_control(self, error: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute multi-variable PID control action.
        
        Args:
            error: Current Einstein tensor error (n_states,)
            dt: Time step
            
        Returns:
            Control action and detailed computation data
        """
        # Apply gain scheduling if enabled
        if self.config['gain_schedule_enable']:
            Kp, Ki, Kd = self._apply_gain_scheduling(error, dt)
        else:
            Kp, Ki, Kd = self.Kp, self.Ki, self.Kd
        
        # Proportional term
        proportional = Kp @ error
        
        # Integral term with anti-windup
        integral_candidate = self.integral_state + error * dt
        integral_limited = self._apply_anti_windup(integral_candidate)
        integral_term = Ki @ integral_limited
        
        # Derivative term with filtering
        derivative_raw = (error - self.previous_error) / dt
        derivative_filtered = self._filter_derivative(derivative_raw, dt)
        derivative_term = Kd @ derivative_filtered
        
        # Cross-coupling compensation
        control_raw = proportional + integral_term + derivative_term
        control_compensated = self.coupling_matrix @ control_raw
        
        # Apply deadband
        deadband = self.config['deadband']
        control_final = jnp.where(
            jnp.abs(control_compensated) > deadband,
            control_compensated,
            0.0
        )
        
        # Update controller states
        new_integral_state = integral_limited
        new_previous_error = error
        new_filtered_derivative = derivative_filtered
        
        # Prepare computation details
        computation_data = {
            'proportional': proportional,
            'integral': integral_term,
            'derivative': derivative_term,
            'cross_coupling_correction': control_compensated - control_raw,
            'deadband_active': jnp.any(jnp.abs(control_compensated) <= deadband),
            'gain_schedule_factor': jnp.linalg.norm(Kp) / jnp.linalg.norm(self.Kp),
            'integral_saturation': jnp.any(jnp.abs(integral_candidate) >= self.config['integral_limit'])
        }
        
        return control_final, computation_data, new_integral_state, new_previous_error, new_filtered_derivative
    
    def apply_control(self, t: float, target_field: Optional[jnp.ndarray] = None) -> Dict:
        """
        Apply multi-variable PID control to transporter system.
        
        Args:
            t: Current time
            target_field: Target Einstein tensor field (optional)
            
        Returns:
            Control analysis results
        """
        # Compute current Einstein tensor field
        field_config = self.transporter.compute_complete_field_configuration(t)
        
        # Mock Einstein tensor computation for demonstration
        # In practice, would compute from metric field
        nx, ny, nz = 16, 16, 16  # Reduced grid for performance
        
        G_current = jnp.zeros((nx, ny, nz, 4, 4))
        if target_field is None:
            G_target = jnp.zeros((nx, ny, nz, 4, 4))  # Target: flat spacetime
        else:
            G_target = target_field
            
        # Fill current Einstein tensor with field-based values
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    rho = i * self.transporter.config.R_payload / nx
                    z = (k - nz//2) * self.transporter.config.L_corridor / nz
                    
                    stress_energy = self.transporter.stress_energy_density(rho, z, t)
                    
                    # Simplified Einstein tensor
                    G_current = G_current.at[i, j, k, 0, 0].set(8 * jnp.pi * stress_energy)
        
        # Compute average error for control (spatial averaging)
        error_field = G_current - G_target
        error_avg = jnp.mean(error_field.reshape(-1, 16), axis=0)  # Average over spatial points
        
        # Time step
        if len(self.time_history) > 0:
            dt = t - self.time_history[-1]
        else:
            dt = self.config['dt']
        dt = max(dt, 1e-6)  # Prevent division by zero
        
        # Compute PID control
        control_action, computation_data, new_integral, new_previous, new_derivative = self.compute_pid_control(error_avg, dt)
        
        # Update controller state
        self.integral_state = new_integral
        self.previous_error = new_previous
        self.filtered_derivative = new_derivative
        
        # Store history
        self.time_history.append(t)
        self.error_history.append(error_avg)
        self.control_history.append(control_action)
        self.gain_history.append(computation_data['gain_schedule_factor'])
        
        # Analyze control performance
        error_norm = jnp.linalg.norm(error_avg)
        control_norm = jnp.linalg.norm(control_action)
        
        performance_data = {
            'time': t,
            'error_norm': float(error_norm),
            'control_norm': float(control_norm),
            'proportional_contribution': float(jnp.linalg.norm(computation_data['proportional'])),
            'integral_contribution': float(jnp.linalg.norm(computation_data['integral'])),
            'derivative_contribution': float(jnp.linalg.norm(computation_data['derivative'])),
            'cross_coupling_active': float(jnp.linalg.norm(computation_data['cross_coupling_correction'])) > 1e-12,
            'gain_schedule_factor': float(computation_data['gain_schedule_factor']),
            'integral_saturation': bool(computation_data['integral_saturation'])
        }
        
        return {
            'control_action': control_action,
            'performance': performance_data,
            'einstein_error_field': error_field,
            'computation_details': computation_data,
            'control_breakdown': {
                'P': computation_data['proportional'],
                'I': computation_data['integral'], 
                'D': computation_data['derivative']
            }
        }
    
    def tune_gains(self, reference_trajectory: List[Dict], 
                   optimization_method: str = 'least_squares') -> Dict:
        """
        Auto-tune PID gains using system response data.
        
        Args:
            reference_trajectory: Reference system responses
            optimization_method: Tuning method ('least_squares', 'ise', 'iae')
            
        Returns:
            Tuning results and optimized gains
        """
        print(f"\n🎛️ AUTO-TUNING PID GAINS")
        print("-" * 50)
        
        def objective_function(gains_flat):
            """Objective function for gain optimization."""
            
            # Reshape gains
            n_gains = len(gains_flat) // 3
            Kp_flat = gains_flat[:n_gains]
            Ki_flat = gains_flat[n_gains:2*n_gains]
            Kd_flat = gains_flat[2*n_gains:3*n_gains]
            
            # Update controller gains temporarily
            old_gains = (self.Kp.copy(), self.Ki.copy(), self.Kd.copy())
            self.Kp = jnp.diag(Kp_flat)
            self.Ki = jnp.diag(Ki_flat)  
            self.Kd = jnp.diag(Kd_flat)
            
            # Reset controller state
            self.integral_state = jnp.zeros((self.n_states,))
            self.previous_error = jnp.zeros((self.n_states,))
            
            # Simulate response
            total_error = 0.0
            for trajectory_point in reference_trajectory:
                t = trajectory_point['time']
                target = trajectory_point['target']
                
                result = self.apply_control(t, target)
                error_norm = result['performance']['error_norm']
                
                if optimization_method == 'ise':  # Integral Square Error
                    total_error += error_norm**2
                elif optimization_method == 'iae':  # Integral Absolute Error
                    total_error += abs(error_norm)
                else:  # least_squares
                    total_error += error_norm**2
                    
            # Restore original gains
            self.Kp, self.Ki, self.Kd = old_gains
            
            return total_error
        
        # Initial guess (current gains)
        gains_initial = jnp.concatenate([
            jnp.diag(self.Kp),
            jnp.diag(self.Ki),
            jnp.diag(self.Kd)
        ])
        
        # Optimization bounds (prevent negative gains)
        bounds = [(1e-6, 1e2) for _ in range(len(gains_initial))]
        
        # Optimize gains
        try:
            result = minimize(
                objective_function,
                gains_initial,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.success:
                # Update gains with optimized values
                n_gains = len(result.x) // 3
                self.Kp = jnp.diag(result.x[:n_gains])
                self.Ki = jnp.diag(result.x[n_gains:2*n_gains])
                self.Kd = jnp.diag(result.x[2*n_gains:3*n_gains])
                
                print(f"✅ Tuning successful:")
                print(f"  Iterations: {result.nit}")
                print(f"  Final objective: {result.fun:.2e}")
                print(f"  Kp norm: {jnp.linalg.norm(self.Kp):.2e}")
                print(f"  Ki norm: {jnp.linalg.norm(self.Ki):.2e}")
                print(f"  Kd norm: {jnp.linalg.norm(self.Kd):.2e}")
                
                return {
                    'success': True,
                    'optimized_gains': {
                        'Kp': self.Kp,
                        'Ki': self.Ki,
                        'Kd': self.Kd
                    },
                    'objective_value': result.fun,
                    'iterations': result.nit
                }
            else:
                print(f"❌ Tuning failed: {result.message}")
                return {'success': False, 'message': result.message}
                
        except Exception as e:
            print(f"❌ Tuning error: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_performance(self) -> Dict:
        """Analyze multi-variable PID controller performance."""
        
        if not self.error_history:
            return {'status': 'No performance data available'}
            
        errors = np.array([jnp.linalg.norm(e) for e in self.error_history])
        controls = np.array([jnp.linalg.norm(c) for c in self.control_history])
        times = np.array(self.time_history)
        gains = np.array(self.gain_history)
        
        # Performance metrics
        steady_state_error = np.mean(errors[-10:]) if len(errors) > 10 else np.mean(errors)
        overshoot = (np.max(errors) - steady_state_error) / steady_state_error * 100 if steady_state_error > 0 else 0
        settling_time = self._compute_settling_time(times, errors, steady_state_error)
        
        # Control effort analysis
        average_control = np.mean(controls)
        peak_control = np.max(controls)
        
        # Gain scheduling activity
        gain_variation = np.std(gains) / np.mean(gains) * 100 if np.mean(gains) > 0 else 0
        
        return {
            'steady_state_error': steady_state_error,
            'overshoot_percent': overshoot,
            'settling_time': settling_time,
            'average_control_effort': average_control,
            'peak_control_effort': peak_control,
            'gain_schedule_variation_percent': gain_variation,
            'cross_coupling_compensation': jnp.linalg.norm(self.coupling_matrix) > 1e-12
        }
    
    def _compute_settling_time(self, times: np.ndarray, errors: np.ndarray, 
                              steady_state_error: float, tolerance: float = 0.02) -> float:
        """Compute settling time (within 2% of steady-state)."""
        
        if len(errors) < 2:
            return 0.0
            
        threshold = steady_state_error * (1 + tolerance)
        
        # Find last time error exceeded threshold
        for i in range(len(errors) - 1, -1, -1):
            if errors[i] > threshold:
                return times[i] if i < len(times) else times[-1]
                
        return times[0]  # Settled from beginning

def main():
    """Demonstration of multi-variable PID controller."""
    from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
    
    print("="*70)
    print("MULTI-VARIABLE PID CONTROLLER DEMONSTRATION")
    print("="*70)
    
    # Create transporter
    config = EnhancedTransporterConfig(
        R_payload=2.0,
        R_neck=0.08,
        L_corridor=50.0,
        corridor_mode="sinusoidal",
        v_conveyor_max=1e6
    )
    transporter = EnhancedStargateTransporter(config)
    
    # Initialize multi-variable PID controller
    pid_controller = MultiVarPIDController(transporter)
    
    # Test control over time
    times = jnp.linspace(0, 15, 30)
    
    print(f"\n🎛️ CONTROL SIMULATION")
    print("-" * 50)
    
    for i, t in enumerate(times):
        result = pid_controller.apply_control(float(t))
        
        if i % 6 == 0:  # Print every 6th step
            perf = result['performance']
            breakdown = result['control_breakdown']
            
            print(f"t = {t:5.2f}s: error = {perf['error_norm']:.2e}, "
                  f"P = {jnp.linalg.norm(breakdown['P']):.2e}, "
                  f"I = {jnp.linalg.norm(breakdown['I']):.2e}, "
                  f"D = {jnp.linalg.norm(breakdown['D']):.2e}")
    
    # Analyze performance
    analysis = pid_controller.analyze_performance()
    
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("-" * 50)
    print(f"Steady-state error: {analysis['steady_state_error']:.2e}")
    print(f"Overshoot: {analysis['overshoot_percent']:.1f}%")
    print(f"Settling time: {analysis['settling_time']:.2f}s")
    print(f"Average control effort: {analysis['average_control_effort']:.2e}")
    print(f"Gain scheduling variation: {analysis['gain_schedule_variation_percent']:.1f}%")
    print(f"Cross-coupling compensation: {'✅ Active' if analysis['cross_coupling_compensation'] else '❌ Inactive'}")
    
    target_performance = {
        'error': 1e-8,
        'overshoot': 10.0,  # %
        'settling_time': 5.0  # s
    }
    
    print(f"\n🎯 TARGET ACHIEVEMENT")
    print("-" * 50)
    for metric, target in target_performance.items():
        if metric in analysis:
            achieved = analysis[metric] if 'percent' not in metric else analysis[metric.replace('_', '_') + '_percent']
            status = "✅ TARGET MET" if achieved <= target else "⚠️ ABOVE TARGET"
            print(f"{metric.replace('_', ' ').title()}: {achieved:.2e} (target: {target:.2e}) {status}")
    
    return pid_controller

if __name__ == "__main__":
    pid_controller = main()
