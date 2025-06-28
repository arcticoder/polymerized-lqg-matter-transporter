"""
Automated Parameter Optimization for Enhanced Stargate Transporter
================================================================

Minimizes total energy E_final(p) = mc²·R_geometric(p)·R_polymer(p)·R_multi(p)·(T_ref/T)⁴
subject to safety constraints T_μν^bio ≤ T_safe and quantum coherence preservation.

Mathematical Framework:
    min_p E_final(p) s.t. T_μν^bio(p) ≤ T_safe, C_quantum(p) ≥ 0

Author: Enhanced Implementation Team
Date: June 28, 2025
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, List, Optional
import time
from dataclasses import dataclass, replace

# Import core transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    optimal_parameters: Dict[str, float]
    optimal_energy: float
    energy_reduction_factor: float
    safety_satisfied: bool
    coherence_preserved: bool
    optimization_time: float
    iterations: int
    convergence_achieved: bool

class TransporterOptimizer:
    """
    Advanced parameter optimizer for enhanced stargate transporter.
    
    Optimizes:
    - μ_polymer: Polymer field strength parameter
    - α_polymer: Polymer field coupling strength  
    - temporal_scale: Transport duration optimization
    - R_neck: Neck radius for geometric optimization
    - corridor_mode: Field configuration mode
    """
    
    def __init__(self, base_config: EnhancedTransporterConfig):
        """
        Initialize optimizer with base configuration.
        
        Args:
            base_config: Base transporter configuration to optimize from
        """
        self.base_config = base_config
        self.evaluation_count = 0
        self.best_energy = float('inf')
        self.optimization_history = []
        
        # Safety thresholds
        self.bio_safety_threshold = 1e-12  # Medical-grade safety limit
        self.coherence_threshold = 0.99     # Quantum coherence preservation
        
        print(f"TransporterOptimizer initialized:")
        print(f"  Base payload mass: {base_config.payload_mass:.1f} kg")
        print(f"  Base neck radius: {base_config.R_neck:.3f} m")
        print(f"  Bio-safety threshold: {self.bio_safety_threshold:.0e}")
        
    def _objective_function(self, x: np.ndarray) -> float:
        """
        Objective function: minimize E_final(p) with safety constraints.
        
        Args:
            x: Parameter vector [mu_polymer, alpha_polymer, temporal_scale, R_neck]
            
        Returns:
            Total energy requirement (J) with penalty terms
        """
        self.evaluation_count += 1
        
        try:
            # Unpack optimization parameters
            mu_polymer, alpha_polymer, temporal_scale, R_neck = x
            
            # Create optimized configuration
            optimized_config = replace(
                self.base_config,
                mu_polymer=float(mu_polymer),
                alpha_polymer=float(alpha_polymer), 
                temporal_scale=float(temporal_scale),
                R_neck=float(R_neck)
            )
            
            # Initialize transporter with optimized parameters
            transporter = EnhancedStargateTransporter(optimized_config)
            
            # Compute total energy requirement
            transport_time = temporal_scale
            payload_mass = optimized_config.payload_mass
            
            energy_analysis = transporter.compute_total_energy_requirement(
                transport_time, payload_mass
            )
            
            # Base energy: E = mc²
            E_base = payload_mass * transporter.c**2
            
            # Final energy with all reduction factors
            total_reduction = energy_analysis['total_reduction_factor']
            E_final = E_base / total_reduction
            
            # Safety constraint evaluation
            safety_analysis = transporter.safety_monitoring_system({
                'max_stress_energy': self.bio_safety_threshold,
                'coherence_threshold': self.coherence_threshold
            })
            
            # Penalty terms for constraint violations
            penalty = 0.0
            
            # Bio-safety penalty
            if not safety_analysis['bio_compatible']:
                penalty += 1e12  # Large penalty for safety violation
                
            # Coherence preservation penalty  
            if not safety_analysis.get('coherence_preserved', True):
                penalty += 1e10  # Penalty for coherence loss
                
            # Geometric constraint penalties
            if R_neck < 0.01 or R_neck > 0.5:  # Physical limits
                penalty += 1e8
                
            if temporal_scale < 100 or temporal_scale > 7200:  # Time limits
                penalty += 1e6
                
            total_objective = E_final + penalty
            
            # Track best solution
            if total_objective < self.best_energy:
                self.best_energy = total_objective
                print(f"  New best energy: {E_final:.2e} J (reduction: {total_reduction:.2e})")
                
            # Store optimization history
            self.optimization_history.append({
                'evaluation': self.evaluation_count,
                'parameters': x.copy(),
                'energy': E_final,
                'penalty': penalty,
                'total': total_objective,
                'reduction_factor': total_reduction,
                'safety_ok': safety_analysis['bio_compatible']
            })
            
            return total_objective
            
        except Exception as e:
            print(f"  Evaluation error at x={x}: {e}")
            return 1e15  # Return very large value for failed evaluations
    
    def optimize_lbfgs(self) -> OptimizationResult:
        """
        Optimize parameters using L-BFGS-B algorithm.
        
        Returns:
            Optimization results with optimal parameters and performance metrics
        """
        print("\n🚀 Starting L-BFGS-B optimization...")
        start_time = time.time()
        
        # Initial parameters [mu_polymer, alpha_polymer, temporal_scale, R_neck]
        x0 = np.array([
            self.base_config.mu_polymer,      # 0.12
            self.base_config.alpha_polymer,   # 1.8  
            1800.0,                           # temporal_scale (30 min)
            self.base_config.R_neck           # 0.08 m
        ])
        
        # Parameter bounds
        bounds = [
            (0.01, 0.3),    # mu_polymer: polymer field strength
            (1.0, 3.0),     # alpha_polymer: coupling strength
            (300, 7200),    # temporal_scale: 5 min to 2 hours
            (0.01, 0.2)     # R_neck: neck radius (1 cm to 20 cm)
        ]
        
        print(f"  Initial parameters: {x0}")
        print(f"  Parameter bounds: {bounds}")
        
        # Run optimization
        result = minimize(
            self._objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 100,
                'ftol': 1e-9,
                'gtol': 1e-6
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Extract optimal parameters
        mu_opt, alpha_opt, time_opt, R_opt = result.x
        
        # Evaluate final solution
        final_config = replace(
            self.base_config,
            mu_polymer=float(mu_opt),
            alpha_polymer=float(alpha_opt),
            temporal_scale=float(time_opt),
            R_neck=float(R_opt)
        )
        
        final_transporter = EnhancedStargateTransporter(final_config)
        final_energy_analysis = final_transporter.compute_total_energy_requirement(
            time_opt, self.base_config.payload_mass
        )
        
        final_safety = final_transporter.safety_monitoring_system({
            'max_stress_energy': self.bio_safety_threshold
        })
        
        # Compute energy reduction
        E_base = self.base_config.payload_mass * final_transporter.c**2
        E_optimal = E_base / final_energy_analysis['total_reduction_factor']
        energy_reduction_factor = final_energy_analysis['total_reduction_factor']
        
        print(f"\n✅ L-BFGS-B optimization completed:")
        print(f"  Optimization time: {optimization_time:.1f} seconds")
        print(f"  Function evaluations: {self.evaluation_count}")
        print(f"  Convergence: {result.success}")
        print(f"  Final energy: {E_optimal:.2e} J")
        print(f"  Energy reduction factor: {energy_reduction_factor:.2e}")
        print(f"  Optimal μ_polymer: {mu_opt:.4f}")
        print(f"  Optimal α_polymer: {alpha_opt:.4f}")
        print(f"  Optimal transport time: {time_opt:.1f} s")
        print(f"  Optimal R_neck: {R_opt:.4f} m")
        
        return OptimizationResult(
            optimal_parameters={
                'mu_polymer': mu_opt,
                'alpha_polymer': alpha_opt,
                'temporal_scale': time_opt,
                'R_neck': R_opt
            },
            optimal_energy=E_optimal,
            energy_reduction_factor=energy_reduction_factor,
            safety_satisfied=final_safety['bio_compatible'],
            coherence_preserved=final_safety.get('coherence_preserved', True),
            optimization_time=optimization_time,
            iterations=result.nit,
            convergence_achieved=result.success
        )

def run_optimization_demo():
    """Demonstration of automated parameter optimization."""
    
    print("🔧 Enhanced Stargate Transporter Parameter Optimization Demo")
    print("=" * 70)
    
    # Create base configuration
    base_config = EnhancedTransporterConfig(
        payload_mass=75.0,          # 75 kg payload
        R_neck=0.08,               # 8 cm neck radius
        mu_polymer=0.12,           # Base polymer strength
        alpha_polymer=1.8,         # Base coupling
        bio_safety_threshold=1e-12  # Medical-grade safety
    )
    
    # Initialize optimizer
    optimizer = TransporterOptimizer(base_config)
    
    # Run L-BFGS-B optimization
    lbfgs_result = optimizer.optimize_lbfgs()
    
    print(f"\n🎯 FINAL OPTIMIZED CONFIGURATION:")
    for param, value in lbfgs_result.optimal_parameters.items():
        print(f"  {param}: {value:.6f}")
    
    print(f"\n✅ Parameter optimization demonstration completed!")
    
    return lbfgs_result

if __name__ == "__main__":
    result = run_optimization_demo()
