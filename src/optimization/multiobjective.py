#!/usr/bin/env python3
"""
Multi-Objective Optimization Module
==================================

Implements adaptive multi-objective optimization for matter transporter:
- Energy minimization with safety constraints
- Parameter sensitivity analysis
- Pareto frontier exploration
- JAX-accelerated gradient computation

This module provides advanced optimization beyond simple parameter tuning,
enabling simultaneous optimization of multiple conflicting objectives
while maintaining bio-compatibility and quantum coherence constraints.

Mathematical Foundation:
Multi-objective optimization seeks Pareto-optimal solutions where
no objective can be improved without degrading others. The approach
uses scalarization techniques and evolutionary algorithms to explore
the trade-off space between energy efficiency and safety constraints.

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

# Import utility modules
try:
    from .polymer_correction import PolymerCorrection
    from .casimir_effect import CasimirGenerator, CasimirConfig
    from .temporal_smearing import TemporalSmearing, TemporalConfig
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from polymer_correction import PolymerCorrection
    from casimir_effect import CasimirGenerator, CasimirConfig
    from temporal_smearing import TemporalSmearing, TemporalConfig

@dataclass
class OptimizationConfig:
    """Configuration for multi-objective optimization."""
    # Parameter bounds
    mu_bounds: Tuple[float, float] = (1e-6, 1e-2)          # Polymer scale bounds
    alpha_bounds: Tuple[float, float] = (1e-10, 1e-6)      # Dispersion alpha bounds
    R_neck_bounds: Tuple[float, float] = (1e-6, 1e-3)      # Neck radius bounds
    T_bounds: Tuple[float, float] = (1.0, 300.0)           # Temperature bounds
    plate_sep_bounds: Tuple[float, float] = (1e-9, 1e-5)   # Casimir plate bounds
    
    # Objective weights
    energy_weight: float = 1.0                              # Energy minimization weight
    safety_weight: float = 1000.0                          # Safety constraint weight
    stability_weight: float = 100.0                        # Stability weight
    
    # Optimization parameters
    max_iterations: int = 1000                              # Maximum iterations
    population_size: int = 50                               # Population for evolutionary
    tolerance: float = 1e-8                                # Convergence tolerance
    constraint_tolerance: float = 1e-6                     # Constraint violation tolerance
    
    # Safety thresholds
    bio_field_limit: float = 1e-12                         # Bio-compatibility field limit
    quantum_coherence_min: float = 0.95                    # Minimum coherence
    structural_stress_max: float = 1e6                     # Maximum stress (Pa)

class MultiObjectiveOptimizer:
    """
    Advanced multi-objective optimization for matter transporter.
    
    Optimizes energy efficiency while maintaining safety constraints
    through Pareto frontier exploration and adaptive parameter tuning.
    
    Objectives:
    1. Minimize total energy requirement
    2. Maximize safety margins
    3. Optimize parameter stability
    4. Maintain quantum coherence
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize multi-objective optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        
        # Initialize physics modules
        self.polymer = None  # Will be initialized with parameters
        self.casimir = None
        self.temporal = None
        
        # Optimization history
        self.history = {
            'parameters': [],
            'objectives': [],
            'constraints': [],
            'pareto_front': []
        }
        
        # JAX-compiled functions
        self._compile_objective_functions()
    
    def _compile_objective_functions(self):
        """Compile JAX objective functions for speed."""
        
        @jit
        def energy_objective_jax(params, constants):
            """JAX-compiled energy objective."""
            mu, alpha, R_neck, T, plate_sep = params
            payload_mass, c_light = constants
            
            # Simplified energy calculation for JAX
            # Full calculation done in numpy functions
            rest_energy = payload_mass * c_light**2
            
            # Approximate reduction factors
            R_polymer_approx = jnp.sinc(mu * 1e-20 / 1.055e-34)  # Approximate
            R_temporal_approx = (300.0 / T) ** 4
            R_casimir_approx = (1e-15 / plate_sep**4) * 1e-6 / rest_energy
            
            total_reduction = R_polymer_approx * R_temporal_approx * R_casimir_approx
            energy_final = rest_energy / jnp.maximum(total_reduction, 1e-20)
            
            return energy_final
        
        @jit
        def constraint_violation_jax(params):
            """JAX-compiled constraint violation."""
            mu, alpha, R_neck, T, plate_sep = params
            
            violations = jnp.array([
                jnp.maximum(0, mu - self.config.mu_bounds[1]),
                jnp.maximum(0, self.config.mu_bounds[0] - mu),
                jnp.maximum(0, alpha - self.config.alpha_bounds[1]),
                jnp.maximum(0, self.config.alpha_bounds[0] - alpha),
                jnp.maximum(0, R_neck - self.config.R_neck_bounds[1]),
                jnp.maximum(0, self.config.R_neck_bounds[0] - R_neck),
                jnp.maximum(0, T - self.config.T_bounds[1]),
                jnp.maximum(0, self.config.T_bounds[0] - T),
                jnp.maximum(0, plate_sep - self.config.plate_sep_bounds[1]),
                jnp.maximum(0, self.config.plate_sep_bounds[0] - plate_sep)
            ])
            
            return jnp.sum(violations**2)
        
        self.energy_objective_jax = energy_objective_jax
        self.constraint_violation_jax = constraint_violation_jax
        
        # Compile gradients
        self.energy_grad = grad(energy_objective_jax, argnums=0)
        self.constraint_grad = grad(constraint_violation_jax)
    
    def objective_function(self, params: np.ndarray, transporter=None, 
                          payload_mass: float = 1.0) -> Dict[str, float]:
        """
        Compute multi-objective function values.
        
        Args:
            params: Parameter vector [mu, alpha, R_neck, T, plate_sep]
            transporter: Transporter instance (if available)
            payload_mass: Payload mass for energy calculation
            
        Returns:
            Dictionary with objective values
        """
        mu, alpha_disp, R_neck, T_operating, plate_sep = params
        
        try:
            # Initialize physics modules with current parameters
            polymer = PolymerCorrection(mu)
            
            casimir_config = CasimirConfig(
                plate_separation=plate_sep,
                num_plates=100,
                V_neck=(4/3) * np.pi * R_neck**3
            )
            casimir = CasimirGenerator(casimir_config)
            
            temporal_config = TemporalConfig(
                T_ref=300.0,
                T_operating=T_operating
            )
            temporal = TemporalSmearing(temporal_config)
            
            # Compute reduction factors
            p_test = 1e-20  # Test momentum
            R_polymer = float(polymer.R_polymer(jnp.array([p_test])))
            R_casimir = casimir.R_casimir(payload_mass)
            R_temporal = temporal.R_temporal(T_operating)
            R_dispersion = 1.0 + alpha_disp * 1e15  # Simplified dispersion
            
            # Total energy reduction
            total_reduction = R_polymer * R_casimir * R_temporal * R_dispersion
            energy_final = payload_mass * (299792458.0**2) / max(total_reduction, 1e-20)
            
            # Safety metrics
            bio_field_estimate = abs(R_casimir) * 1e-10  # Simplified field estimate
            quantum_coherence = R_polymer * 0.99  # Polymer preserves coherence
            structural_stress = abs(casimir.force_per_area(plate_sep))
            
            # Constraint violations
            bio_violation = max(0, bio_field_estimate - self.config.bio_field_limit)
            coherence_violation = max(0, self.config.quantum_coherence_min - quantum_coherence)
            stress_violation = max(0, structural_stress - self.config.structural_stress_max)
            
            # Parameter stability (sensitivity to small changes)
            stability_metric = self._compute_parameter_stability(params)
            
            return {
                'energy': energy_final,
                'bio_safety': bio_violation,
                'quantum_coherence': coherence_violation,
                'structural_integrity': stress_violation,
                'parameter_stability': stability_metric,
                'total_reduction': total_reduction,
                'individual_reductions': {
                    'polymer': R_polymer,
                    'casimir': R_casimir,
                    'temporal': R_temporal,
                    'dispersion': R_dispersion
                }
            }
            
        except Exception as e:
            warnings.warn(f"Objective function evaluation failed: {e}")
            return {
                'energy': 1e20,  # Penalty for failed evaluation
                'bio_safety': 1e10,
                'quantum_coherence': 1e10,
                'structural_integrity': 1e10,
                'parameter_stability': 1e10,
                'total_reduction': 1e-20,
                'individual_reductions': {}
            }
    
    def _compute_parameter_stability(self, params: np.ndarray, 
                                   epsilon: float = 1e-6) -> float:
        """
        Compute parameter stability metric.
        
        Measures sensitivity of objective to parameter variations.
        """
        base_obj = self.objective_function(params)
        base_energy = base_obj['energy']
        
        # Finite difference sensitivity
        sensitivities = []
        for i, param in enumerate(params):
            params_plus = params.copy()
            params_plus[i] += epsilon * abs(param)
            
            params_minus = params.copy()
            params_minus[i] -= epsilon * abs(param)
            
            try:
                obj_plus = self.objective_function(params_plus)
                obj_minus = self.objective_function(params_minus)
                
                # Relative sensitivity
                sensitivity = abs(obj_plus['energy'] - obj_minus['energy']) / (2 * epsilon * base_energy)
                sensitivities.append(sensitivity)
            except:
                sensitivities.append(1e10)  # High penalty for unstable regions
        
        return np.mean(sensitivities)
    
    def scalarized_objective(self, params: np.ndarray, weights: Dict[str, float] = None) -> float:
        """
        Compute scalarized objective for single-objective optimization.
        
        Args:
            params: Parameter vector
            weights: Objective weights (uses config defaults if None)
            
        Returns:
            Weighted sum of normalized objectives
        """
        if weights is None:
            weights = {
                'energy': self.config.energy_weight,
                'bio_safety': self.config.safety_weight,
                'quantum_coherence': self.config.safety_weight,
                'structural_integrity': self.config.safety_weight,
                'parameter_stability': self.config.stability_weight
            }
        
        objectives = self.objective_function(params)
        
        # Normalize objectives (approximate scales)
        normalized = {
            'energy': objectives['energy'] / 1e15,  # ~PJ scale
            'bio_safety': objectives['bio_safety'] / self.config.bio_field_limit,
            'quantum_coherence': objectives['quantum_coherence'] / (1 - self.config.quantum_coherence_min),
            'structural_integrity': objectives['structural_integrity'] / self.config.structural_stress_max,
            'parameter_stability': objectives['parameter_stability'] / 1000
        }
        
        # Weighted sum
        total = sum(weights[key] * normalized[key] for key in weights.keys() if key in normalized)
        
        return total
    
    def optimize_single_objective(self, method: str = 'L-BFGS-B', 
                                 weights: Dict[str, float] = None) -> Dict:
        """
        Single-objective optimization using scalarization.
        
        Args:
            method: Optimization method
            weights: Objective weights
            
        Returns:
            Optimization results
        """
        # Parameter bounds
        bounds = [
            self.config.mu_bounds,
            self.config.alpha_bounds,
            self.config.R_neck_bounds,
            self.config.T_bounds,
            self.config.plate_sep_bounds
        ]
        
        # Initial guess (middle of bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds]
        
        # Optimization
        result = minimize(
            fun=self.scalarized_objective,
            x0=x0,
            args=(weights,),
            method=method,
            bounds=bounds,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        # Evaluate final objectives
        final_objectives = self.objective_function(result.x)
        
        return {
            'optimal_parameters': {
                'mu': result.x[0],
                'alpha_dispersion': result.x[1],
                'R_neck': result.x[2],
                'T_operating': result.x[3],
                'plate_separation': result.x[4]
            },
            'optimal_objectives': final_objectives,
            'optimization_success': result.success,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'final_energy': final_objectives['energy'],
            'total_reduction': final_objectives['total_reduction']
        }
    
    def pareto_frontier_optimization(self, n_points: int = 20) -> Dict:
        """
        Multi-objective optimization to find Pareto frontier.
        
        Uses weighted sum method with varying weights to explore trade-offs.
        
        Args:
            n_points: Number of Pareto points to find
            
        Returns:
            Pareto frontier results
        """
        pareto_points = []
        weight_combinations = []
        
        # Generate weight combinations
        for i in range(n_points):
            # Vary energy vs safety trade-off
            energy_weight = 1.0
            safety_weight = 10 ** (i * 4.0 / (n_points - 1))  # 1 to 10^4
            
            weights = {
                'energy': energy_weight,
                'bio_safety': safety_weight,
                'quantum_coherence': safety_weight,
                'structural_integrity': safety_weight,
                'parameter_stability': 100.0
            }
            
            try:
                result = self.optimize_single_objective(weights=weights)
                if result['optimization_success']:
                    pareto_points.append(result)
                    weight_combinations.append(weights)
            except Exception as e:
                warnings.warn(f"Pareto point optimization failed: {e}")
        
        # Filter for actual Pareto optimality
        pareto_optimal = self._filter_pareto_optimal(pareto_points)
        
        return {
            'pareto_points': pareto_optimal,
            'weight_combinations': weight_combinations,
            'n_optimal_points': len(pareto_optimal),
            'energy_range': (
                min(p['final_energy'] for p in pareto_optimal),
                max(p['final_energy'] for p in pareto_optimal)
            ),
            'reduction_range': (
                min(p['total_reduction'] for p in pareto_optimal),
                max(p['total_reduction'] for p in pareto_optimal)
            )
        }
    
    def _filter_pareto_optimal(self, points: List[Dict]) -> List[Dict]:
        """
        Filter points to keep only Pareto-optimal solutions.
        
        Args:
            points: List of optimization results
            
        Returns:
            Pareto-optimal subset
        """
        if not points:
            return []
        
        # Extract objective values (energy, safety violations)
        objectives = []
        for point in points:
            obj = point['optimal_objectives']
            objectives.append([
                obj['energy'],
                obj['bio_safety'] + obj['quantum_coherence'] + obj['structural_integrity']
            ])
        
        objectives = np.array(objectives)
        pareto_optimal = []
        
        for i, point_i in enumerate(points):
            is_dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j:
                    # Point i is dominated if another point is better in all objectives
                    if all(obj_j <= objectives[i]) and any(obj_j < objectives[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(point_i)
        
        return pareto_optimal
    
    def adaptive_optimization(self, stages: int = 3) -> Dict:
        """
        Multi-stage adaptive optimization with refinement.
        
        Args:
            stages: Number of optimization stages
            
        Returns:
            Adaptive optimization results
        """
        results = {}
        current_bounds = [
            self.config.mu_bounds,
            self.config.alpha_bounds,
            self.config.R_neck_bounds,
            self.config.T_bounds,
            self.config.plate_sep_bounds
        ]
        
        for stage in range(stages):
            print(f"Stage {stage + 1}/{stages}: Optimizing with bounds {current_bounds}")
            
            # Optimize current stage
            stage_result = self.optimize_single_objective()
            results[f'stage_{stage + 1}'] = stage_result
            
            if stage < stages - 1 and stage_result['optimization_success']:
                # Refine bounds around current optimum
                optimal_params = list(stage_result['optimal_parameters'].values())
                new_bounds = []
                
                for i, (param, bound) in enumerate(zip(optimal_params, current_bounds)):
                    width = (bound[1] - bound[0]) * 0.1  # 10% of current range
                    new_bound = (
                        max(bound[0], param - width),
                        min(bound[1], param + width)
                    )
                    new_bounds.append(new_bound)
                
                current_bounds = new_bounds
        
        # Final comprehensive analysis
        final_params = list(results[f'stage_{stages}']['optimal_parameters'].values())
        final_analysis = self.comprehensive_analysis(final_params)
        
        return {
            'stage_results': results,
            'final_analysis': final_analysis,
            'convergence_history': [results[f'stage_{i+1}']['final_energy'] for i in range(stages)]
        }
    
    def comprehensive_analysis(self, params: np.ndarray) -> Dict:
        """
        Comprehensive analysis of optimized parameters.
        
        Args:
            params: Optimized parameter vector
            
        Returns:
            Complete analysis results
        """
        # Base objectives
        objectives = self.objective_function(params)
        
        # Parameter interpretation
        mu, alpha_disp, R_neck, T_operating, plate_sep = params
        
        # Physical analysis
        polymer = PolymerCorrection(mu)
        casimir_config = CasimirConfig(plate_separation=plate_sep, V_neck=(4/3)*np.pi*R_neck**3)
        casimir = CasimirGenerator(casimir_config)
        temporal_config = TemporalConfig(T_operating=T_operating)
        temporal = TemporalSmearing(temporal_config)
        
        # Detailed metrics
        coherence_length = temporal.thermal_coherence_length(T_operating)
        casimir_force = casimir.force_per_area()
        
        return {
            'optimized_parameters': {
                'polymer_scale_mu': mu,
                'dispersion_alpha': alpha_disp,
                'neck_radius': R_neck,
                'operating_temperature': T_operating,
                'casimir_plate_separation': plate_sep
            },
            'performance_metrics': objectives,
            'physical_properties': {
                'thermal_coherence_length': coherence_length,
                'casimir_force_per_area': casimir_force,
                'polymer_momentum_scale': 1.055e-34 / mu,
                'neck_volume': (4/3) * np.pi * R_neck**3,
                'cooling_requirements': temporal._estimate_cooling_power(T_operating)
            },
            'optimization_quality': {
                'constraint_satisfaction': all([
                    objectives['bio_safety'] < self.config.constraint_tolerance,
                    objectives['quantum_coherence'] < self.config.constraint_tolerance,
                    objectives['structural_integrity'] < self.config.constraint_tolerance
                ]),
                'parameter_stability': objectives['parameter_stability'],
                'total_energy_reduction': objectives['total_reduction']
            }
        }

# Utility functions
def plot_pareto_frontier(pareto_results: Dict, save_path: Optional[str] = None):
    """
    Plot Pareto frontier results.
    
    Args:
        pareto_results: Results from pareto_frontier_optimization
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        points = pareto_results['pareto_points']
        energies = [p['final_energy'] for p in points]
        reductions = [p['total_reduction'] for p in points]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(energies, reductions, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Final Energy (J)')
        plt.ylabel('Total Reduction Factor')
        plt.title('Pareto Frontier: Energy vs Reduction Trade-off')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    # Demonstration of multi-objective optimization
    print("Multi-Objective Optimization Demonstration")
    print("=" * 50)
    
    # Configuration
    config = OptimizationConfig(
        max_iterations=100,  # Reduced for demo
        population_size=20
    )
    
    optimizer = MultiObjectiveOptimizer(config)
    
    # Single-objective optimization
    print("Running single-objective optimization...")
    result = optimizer.optimize_single_objective()
    
    if result['optimization_success']:
        print(f"Optimization successful!")
        print(f"Final energy: {result['final_energy']:.3e} J")
        print(f"Total reduction: {result['total_reduction']:.3e}")
        print(f"Optimal parameters:")
        for key, value in result['optimal_parameters'].items():
            print(f"  {key}: {value:.6e}")
    else:
        print("Optimization failed")
    
    # Pareto frontier (reduced points for demo)
    print("\nRunning Pareto frontier optimization...")
    pareto_results = optimizer.pareto_frontier_optimization(n_points=5)
    print(f"Found {pareto_results['n_optimal_points']} Pareto-optimal points")
    print(f"Energy range: {pareto_results['energy_range'][0]:.2e} to {pareto_results['energy_range'][1]:.2e} J")
    print(f"Reduction range: {pareto_results['reduction_range'][0]:.2e} to {pareto_results['reduction_range'][1]:.2e}")
