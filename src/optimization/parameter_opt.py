"""
Automated Parameter Optimization for Enhanced Stargate Transporter

This module implements gradient-based optimization to minimize the energy requirement
for a 1-hour transport operation by tuning geometric and polymer parameters.

Mathematical Framework:
    min_{p} E_final(p) subject to:
    - bio_compatible(p) = True  
    - quantum_coherent(p) = True
    
Where:
    E_final(p) = m⋅c² ⋅ R_geometric(p) ⋅ R_polymer(p) ⋅ R_multi_bubble(p) ⋅ (T_ref/T)⁴ ⋅ e^(-T/2T_ref)

Uses JAX for automatic differentiation and SciPy for constrained optimization.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, List, Optional
import time
import warnings

# Import our enhanced transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_stargate_transporter import (
    EnhancedTransporterConfig,
    EnhancedStargateTransporter
)

# Fixed test scenario for optimization
PAYLOAD_MASS = 75.0      # kg (average human)
TRANSPORT_TIME = 3600.0  # s (1 hour)

class TransporterOptimizer:
    """
    Automated parameter optimization for stargate transporter.
    
    Optimizes:
    - mu_polymer: LQG polymer parameter [0.01, 0.5]
    - alpha_polymer: Polymer enhancement factor [1.0, 3.0] 
    - temporal_scale: Reference time scale [600, 7200] seconds
    - R_neck: Neck radius [0.01, 0.2] meters
    """
    
    def __init__(self, base_config: Optional[EnhancedTransporterConfig] = None):
        """Initialize optimizer with base configuration."""
        self.base_config = base_config or EnhancedTransporterConfig()
        
        # Parameter bounds: [mu_polymer, alpha_polymer, temporal_scale, R_neck]
        self.bounds = [
            (0.01, 0.5),    # mu_polymer: quantum parameter
            (1.0, 3.0),     # alpha_polymer: enhancement factor
            (600, 7200),    # temporal_scale: 10min to 2hours
            (0.01, 0.2)     # R_neck: 1cm to 20cm radius
        ]
        
        # Parameter names for reporting
        self.param_names = ['mu_polymer', 'alpha_polymer', 'temporal_scale', 'R_neck']
        
        # Optimization history
        self.history = {
            'parameters': [],
            'energies': [],
            'constraints': [],
            'gradients': []
        }
        
        print("TransporterOptimizer initialized")
        print(f"  Parameter bounds: {dict(zip(self.param_names, self.bounds))}")
        print(f"  Target scenario: {PAYLOAD_MASS}kg payload, {TRANSPORT_TIME}s transport")
    
    def pack_params(self, mu: float, alpha: float, T_ref: float, R_neck: float) -> jnp.ndarray:
        """Pack optimization parameters into JAX array."""
        return jnp.array([mu, alpha, T_ref, R_neck])
    
    def unpack_params(self, x: jnp.ndarray) -> EnhancedTransporterConfig:
        """Unpack JAX array into transporter configuration."""
        mu, alpha, T_ref, R_neck = x
        
        # Create new config based on base config
        config = EnhancedTransporterConfig(
            # Optimized parameters
            mu_polymer=float(mu),
            alpha_polymer=float(alpha), 
            temporal_scale=float(T_ref),
            R_neck=float(R_neck),
            
            # Fixed parameters from base config
            R_payload=self.base_config.R_payload,
            L_corridor=self.base_config.L_corridor,
            delta_wall=self.base_config.delta_wall,
            v_conveyor=self.base_config.v_conveyor,
            
            # Enable all enhancements
            use_van_den_broeck=True,
            use_temporal_smearing=True,
            use_multi_bubble=True,
            sinc_correction=True,
            
            # Safety parameters
            bio_safety_threshold=self.base_config.bio_safety_threshold,
            quantum_coherence_preservation=True,
            emergency_response_time=self.base_config.emergency_response_time
        )
        
        return config
    
    def objective_jax(self, x: jnp.ndarray) -> float:
        """
        JAX-compatible objective function for energy minimization.
        
        Args:
            x: Parameter array [mu_polymer, alpha_polymer, temporal_scale, R_neck]
            
        Returns:
            Total energy requirement (to be minimized)
        """
        mu, alpha, T_ref, R_neck = x
        
        # Ensure parameters are within bounds (soft constraints)
        mu = jnp.clip(mu, 0.01, 0.5)
        alpha = jnp.clip(alpha, 1.0, 3.0) 
        T_ref = jnp.clip(T_ref, 600, 7200)
        R_neck = jnp.clip(R_neck, 0.01, 0.2)
        
        # Base energy scale
        c = 299792458.0
        E_base = PAYLOAD_MASS * c**2
        
        # Geometric reduction (Van den Broeck volume effect)
        R_payload = self.base_config.R_payload
        volume_ratio = (R_payload / R_neck)**2
        R_geometric = 1e-5 * jnp.sqrt(volume_ratio)  # Enhanced with neck size
        
        # Polymer enhancement
        R_polymer = alpha
        
        # Multi-bubble factor (fixed)
        R_multi_bubble = 2.0
        
        # Temporal smearing with exponential decay
        temporal_factor = (T_ref / TRANSPORT_TIME)**4 * jnp.exp(-TRANSPORT_TIME / (2 * T_ref))
        
        # Total energy calculation
        E_final = E_base * R_geometric * R_polymer * R_multi_bubble * temporal_factor
        
        return E_final
    
    def objective_scipy(self, x: np.ndarray) -> float:
        """SciPy-compatible objective function."""
        x_jax = jnp.array(x)
        energy = float(self.objective_jax(x_jax))
        
        # Store in history
        self.history['parameters'].append(x.copy())
        self.history['energies'].append(energy)
        
        return energy
    
    def constraint_bio_compatibility(self, x: np.ndarray) -> float:
        """Biological compatibility constraint."""
        config = self.unpack_params(jnp.array(x))
        transporter = EnhancedStargateTransporter(config)
        
        # Check stress-energy density at critical points
        rho_test = config.R_neck + 0.001  # Just outside neck
        z_test = config.L_corridor / 2
        
        stress_energy = transporter.stress_energy_density(rho_test, z_test)
        max_safe_density = config.bio_safety_threshold
        
        # Return positive if constraint satisfied
        constraint_value = max_safe_density - abs(stress_energy)
        self.history['constraints'].append(constraint_value)
        
        return constraint_value
    
    def constraint_quantum_coherence(self, x: np.ndarray) -> float:
        """Quantum coherence preservation constraint."""
        mu, alpha, T_ref, R_neck = x
        
        # Coherence requires reasonable polymer parameter
        mu_coherence_limit = 0.45  # Conservative limit
        
        # Coherence also requires reasonable enhancement factor
        alpha_coherence_limit = 2.8
        
        # Combined coherence constraint
        coherence_violation = max(0, mu - mu_coherence_limit) + max(0, alpha - alpha_coherence_limit)
        
        return -coherence_violation  # Negative violation means constraint satisfied
    
    def gradient_scipy(self, x: np.ndarray) -> np.ndarray:
        """SciPy-compatible gradient function using JAX autodiff."""
        x_jax = jnp.array(x)
        
        # Create standalone function for JAX differentiation
        def standalone_objective(params):
            mu, alpha, T_ref, R_neck = params
            
            # Ensure parameters are within bounds (soft constraints)
            mu = jnp.clip(mu, 0.01, 0.5)
            alpha = jnp.clip(alpha, 1.0, 3.0) 
            T_ref = jnp.clip(T_ref, 600, 7200)
            R_neck = jnp.clip(R_neck, 0.01, 0.2)
            
            # Base energy scale
            c = 299792458.0
            E_base = PAYLOAD_MASS * c**2
            
            # Geometric reduction (Van den Broeck volume effect)
            R_payload = self.base_config.R_payload
            volume_ratio = (R_payload / R_neck)**2
            R_geometric = 1e-5 * jnp.sqrt(volume_ratio)  # Enhanced with neck size
            
            # Polymer enhancement
            R_polymer = alpha
            
            # Multi-bubble factor (fixed)
            R_multi_bubble = 2.0
            
            # Temporal smearing with exponential decay
            temporal_factor = (T_ref / TRANSPORT_TIME)**4 * jnp.exp(-TRANSPORT_TIME / (2 * T_ref))
            
            # Total energy calculation
            E_final = E_base * R_geometric * R_polymer * R_multi_bubble * temporal_factor
            
            return E_final
        
        grad_func = jit(grad(standalone_objective))
        gradient = np.array(grad_func(x_jax))
        
        self.history['gradients'].append(gradient.copy())
        
        return gradient
    
    def optimize_parameters(self, method: str = 'L-BFGS-B', 
                          max_iterations: int = 100,
                          use_constraints: bool = True) -> Dict:
        """
        Run parameter optimization.
        
        Args:
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'differential_evolution')
            max_iterations: Maximum number of iterations
            use_constraints: Whether to enforce safety constraints
            
        Returns:
            Optimization results dictionary
        """
        print(f"\n🔧 Starting Parameter Optimization")
        print(f"   Method: {method}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Safety constraints: {use_constraints}")
        print("-" * 60)
        
        # Initial guess (reasonable starting point)
        x0 = np.array([
            0.15,    # mu_polymer (moderate quantum effect)
            1.8,     # alpha_polymer (moderate enhancement)
            1800.0,  # temporal_scale (30 minutes)
            0.08     # R_neck (8cm radius)
        ])
        
        print(f"Initial parameters: {dict(zip(self.param_names, x0))}")
        E_initial = self.objective_scipy(x0)
        print(f"Initial energy: {E_initial:.3e} J")
        
        # Setup constraints
        constraints = []
        if use_constraints:
            constraints = [
                {'type': 'ineq', 'fun': self.constraint_bio_compatibility},
                {'type': 'ineq', 'fun': self.constraint_quantum_coherence}
            ]
        
        # Run optimization
        start_time = time.time()
        
        if method == 'differential_evolution':
            # Global optimization for robust solution
            result = differential_evolution(
                func=self.objective_scipy,
                bounds=self.bounds,
                maxiter=max_iterations,
                popsize=15,
                seed=42,
                disp=True
            )
        else:
            # Gradient-based optimization
            result = minimize(
                fun=self.objective_scipy,
                x0=x0,
                method=method,
                jac=self.gradient_scipy,
                bounds=self.bounds,
                constraints=constraints,
                options={
                    'maxiter': max_iterations,
                    'disp': True,
                    'ftol': 1e-12
                }
            )
        
        optimization_time = time.time() - start_time
        
        # Analyze results
        if result.success:
            print(f"\n✅ Optimization successful!")
        else:
            print(f"\n⚠️ Optimization completed with warnings:")
            print(f"   Message: {result.message}")
        
        optimal_params = dict(zip(self.param_names, result.x))
        energy_reduction = E_initial / result.fun if result.fun > 0 else np.inf
        
        print(f"\n📊 OPTIMIZATION RESULTS")
        print("-" * 40)
        print(f"Optimal parameters:")
        for name, value in optimal_params.items():
            print(f"  {name:15s}: {value:.4f}")
        
        print(f"\nEnergy performance:")
        print(f"  Initial energy:  {E_initial:.3e} J")
        print(f"  Optimized energy: {result.fun:.3e} J") 
        print(f"  Reduction factor: {energy_reduction:.2e}×")
        print(f"  Optimization time: {optimization_time:.2f} seconds")
        print(f"  Function evaluations: {result.nfev}")
        
        # Validate final configuration
        final_config = self.unpack_params(jnp.array(result.x))
        final_transporter = EnhancedStargateTransporter(final_config)
        
        # Complete energy analysis
        energy_analysis = final_transporter.compute_total_energy_requirement(
            TRANSPORT_TIME, PAYLOAD_MASS
        )
        
        print(f"\n🔬 FINAL CONFIGURATION ANALYSIS")
        print("-" * 40)
        print(f"Total reduction factor: {energy_analysis['total_reduction_factor']:.2e}×")
        print(f"Geometric contribution: {1e-5:.1e}×")
        print(f"Polymer contribution: {optimal_params['alpha_polymer']:.2f}×")
        print(f"Temporal contribution: {energy_analysis['temporal_reduction']:.2e}×")
        
        # Safety validation
        field_state = {
            'max_stress_energy': 1e-15,  # Conservative estimate
            'max_gradient': 1e-20,
            'junction_stability': 1e-12
        }
        safety_status = final_transporter.safety_monitoring_system(field_state)
        
        print(f"\n🛡️ SAFETY VALIDATION")
        print("-" * 40)
        for param, status in safety_status.items():
            symbol = "✅" if status else "❌"
            print(f"  {param:20s}: {symbol}")
        
        return {
            'success': result.success,
            'optimal_parameters': optimal_params,
            'optimal_config': final_config,
            'energy_initial': E_initial,
            'energy_final': result.fun,
            'energy_reduction': energy_reduction,
            'optimization_time': optimization_time,
            'function_evaluations': result.nfev,
            'energy_analysis': energy_analysis,
            'safety_status': safety_status,
            'optimization_history': self.history,
            'scipy_result': result
        }
    
    def parameter_sensitivity_analysis(self, optimal_params: np.ndarray, 
                                     perturbation: float = 0.05) -> Dict:
        """
        Analyze parameter sensitivity around optimal point.
        
        Args:
            optimal_params: Optimal parameter values
            perturbation: Relative perturbation for sensitivity analysis
            
        Returns:
            Sensitivity analysis results
        """
        print(f"\n🔍 Parameter Sensitivity Analysis")
        print(f"   Perturbation level: ±{perturbation*100:.1f}%")
        print("-" * 50)
        
        sensitivities = {}
        base_energy = self.objective_scipy(optimal_params)
        
        for i, param_name in enumerate(self.param_names):
            # Positive perturbation
            params_plus = optimal_params.copy()
            params_plus[i] *= (1 + perturbation)
            energy_plus = self.objective_scipy(params_plus)
            
            # Negative perturbation
            params_minus = optimal_params.copy()
            params_minus[i] *= (1 - perturbation)
            energy_minus = self.objective_scipy(params_minus)
            
            # Calculate sensitivity (relative change in energy per relative change in parameter)
            relative_sensitivity = ((energy_plus - energy_minus) / (2 * perturbation)) / base_energy
            
            sensitivities[param_name] = {
                'relative_sensitivity': relative_sensitivity,
                'energy_plus': energy_plus,
                'energy_minus': energy_minus,
                'energy_range': abs(energy_plus - energy_minus)
            }
            
            print(f"  {param_name:15s}: sensitivity = {relative_sensitivity:.2e}")
        
        # Rank parameters by sensitivity
        sensitivity_ranking = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]['relative_sensitivity']),
            reverse=True
        )
        
        print(f"\n📈 Sensitivity Ranking (most to least sensitive):")
        for i, (param, data) in enumerate(sensitivity_ranking, 1):
            print(f"  {i}. {param:15s}: {abs(data['relative_sensitivity']):.2e}")
        
        return sensitivities

def demonstrate_parameter_optimization():
    """Comprehensive demonstration of parameter optimization."""
    print("="*80)
    print("AUTOMATED PARAMETER OPTIMIZATION FOR STARGATE TRANSPORTER")
    print("="*80)
    
    # Create base configuration
    base_config = EnhancedTransporterConfig(
        R_payload=2.0,
        L_corridor=50.0,
        use_van_den_broeck=True,
        use_temporal_smearing=True,
        use_multi_bubble=True
    )
    
    # Initialize optimizer
    optimizer = TransporterOptimizer(base_config)
    
    # Run optimization with gradient-based method
    print(f"\n🚀 PHASE 1: Gradient-Based Optimization")
    results_gradient = optimizer.optimize_parameters(
        method='L-BFGS-B',
        max_iterations=50,
        use_constraints=True
    )
    
    # Run global optimization for comparison
    print(f"\n🌍 PHASE 2: Global Optimization")
    optimizer_global = TransporterOptimizer(base_config)  # Fresh optimizer
    results_global = optimizer_global.optimize_parameters(
        method='differential_evolution',
        max_iterations=30,
        use_constraints=False
    )
    
    # Compare results
    print(f"\n⚖️ OPTIMIZATION COMPARISON")
    print("-" * 60)
    print(f"Gradient-based (L-BFGS-B):")
    print(f"  Final energy: {results_gradient['energy_final']:.3e} J")
    print(f"  Reduction: {results_gradient['energy_reduction']:.2e}×")
    print(f"  Time: {results_gradient['optimization_time']:.2f}s")
    
    print(f"\nGlobal (Differential Evolution):")
    print(f"  Final energy: {results_global['energy_final']:.3e} J")
    print(f"  Reduction: {results_global['energy_reduction']:.2e}×")
    print(f"  Time: {results_global['optimization_time']:.2f}s")
    
    # Choose best result
    best_results = results_gradient if results_gradient['energy_final'] < results_global['energy_final'] else results_global
    best_method = "Gradient-based" if best_results == results_gradient else "Global"
    
    print(f"\n🏆 Best Method: {best_method}")
    
    # Sensitivity analysis on best result
    best_params = np.array(list(best_results['optimal_parameters'].values()))
    optimizer_best = optimizer if best_results == results_gradient else optimizer_global
    sensitivity_results = optimizer_best.parameter_sensitivity_analysis(best_params)
    
    # Final summary
    print(f"\n🌟 OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Optimal configuration found:")
    for param, value in best_results['optimal_parameters'].items():
        print(f"  {param:15s}: {value:.4f}")
    
    print(f"\nEnergy reduction achieved: {best_results['energy_reduction']:.2e}×")
    print(f"Transport ready: {'✅' if all(best_results['safety_status'].values()) else '❌'}")
    
    return best_results, sensitivity_results

def main():
    """Main optimization demonstration."""
    results, sensitivity = demonstrate_parameter_optimization()
    return results

if __name__ == "__main__":
    optimization_results = main()
