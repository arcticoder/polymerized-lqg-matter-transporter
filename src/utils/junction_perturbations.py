#!/usr/bin/env python3
"""
Junction-Condition Perturbations with Polymer/Casimir Corrections
===============================================================

Advanced analysis of spacetime junction conditions with LQG polymer
and Casimir effect modifications. Handles matching conditions across
matter transporter boundaries with quantum corrections.

Implements:
- Israel junction conditions: [K_μν] = 8πG(T_μν - ½gμν T) across boundary
- Polymer corrections: K_μν^poly = K_μν^classical × [1 - α sin²(μK/μ₀)]
- Casimir stress: T_μν^Casimir = ρ_Casimir × diag(-1, 1, 1, 1) 
- Matching conditions: [g_μν] = 0, [∂_n g_μν] = δ(surface) × source

Mathematical Foundation:
Based on unified-lqg repository junction analysis methodology
- Israel formalism for hypersurface matching
- Polymer discretization regularizes singular boundary terms
- Casimir negative energy provides stress-energy sources

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

import numpy as np
from scipy.integrate import quad, odeint
from scipy.optimize import fsolve, minimize
from scipy.linalg import eigh, solve
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings

class BoundaryType(Enum):
    """Enumeration of boundary surface types."""
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR = "planar"
    CUSTOM = "custom"

class MatchingCondition(Enum):
    """Enumeration of matching condition types."""
    CONTINUOUS = "continuous"        # [g_μν] = 0
    ISRAEL = "israel"               # [K_μν] = source
    THIN_SHELL = "thin_shell"       # Delta function source
    SMOOTH = "smooth"               # Higher-order matching

@dataclass
class JunctionConfig:
    """Configuration for junction condition analysis."""
    # Boundary geometry
    boundary_type: BoundaryType = BoundaryType.SPHERICAL
    boundary_radius: float = 1e-9        # Characteristic boundary size [m]
    boundary_thickness: float = 1e-10    # Interface thickness [m]
    
    # Physical parameters
    G: float = 6.67430e-11              # Gravitational constant [m³/kg⋅s²]
    c: float = 299792458                # Speed of light [m/s]
    hbar: float = 1.054571817e-34       # Reduced Planck constant [J⋅s]
    
    # LQG polymer parameters
    polymer_scale: float = 1e-35        # Polymer discretization scale μ₀ [m]
    alpha_polymer: float = 0.1          # Polymer correction strength
    apply_polymer_corrections: bool = True
    
    # Casimir effect parameters
    casimir_plates: int = 2             # Number of parallel plates
    plate_separation: float = 1e-9      # Plate separation [m]
    casimir_coupling: float = 1.0       # Casimir coupling strength
    apply_casimir_corrections: bool = True
    
    # Matching conditions
    matching_types: List[MatchingCondition] = None
    junction_source_strength: float = 1e-20  # Source term amplitude [kg/m³]
    
    # Numerical parameters
    grid_points: int = 100              # Number of radial grid points
    integration_tolerance: float = 1e-10
    max_iterations: int = 1000
    
    def __post_init__(self):
        """Initialize default matching conditions."""
        if self.matching_types is None:
            self.matching_types = [
                MatchingCondition.CONTINUOUS,
                MatchingCondition.ISRAEL
            ]

class JunctionConditions:
    """
    Junction condition analysis with polymer and Casimir corrections.
    
    Analyzes spacetime matching across boundaries:
    1. Metric continuity conditions [g_μν] = 0
    2. Extrinsic curvature jump [K_μν] = 8πG S_μν
    3. LQG polymer modifications to junction geometry
    4. Casimir stress-energy contributions
    
    Parameters:
    -----------
    config : JunctionConfig
        Junction analysis configuration
    metric_inside : Callable
        Metric tensor inside boundary g_μν^(in)(r,t)
    metric_outside : Callable  
        Metric tensor outside boundary g_μν^(out)(r,t)
    """
    
    def __init__(self, config: JunctionConfig, 
                 metric_inside: Callable, metric_outside: Callable):
        """
        Initialize junction conditions analyzer.
        
        Args:
            config: Junction configuration
            metric_inside: Interior metric function g_μν^(in)(r,t)
            metric_outside: Exterior metric function g_μν^(out)(r,t)
        """
        self.config = config
        self.metric_inside = metric_inside
        self.metric_outside = metric_outside
        
        # Initialize coordinate system
        self._setup_coordinates()
        
        # Initialize surface geometry
        self._setup_boundary_surface()
        
        # Compute junction conditions
        self._initialize_junction_analysis()
        
        print(f"Junction conditions analyzer initialized:")
        print(f"  Boundary type: {config.boundary_type.value}")
        print(f"  Boundary radius: {config.boundary_radius:.2e} m")
        print(f"  Polymer scale: {config.polymer_scale:.2e} m")
        print(f"  Casimir plates: {config.casimir_plates}")
        print(f"  Matching conditions: {[mc.value for mc in config.matching_types]}")
    
    def _setup_coordinates(self):
        """Setup coordinate system for junction analysis."""
        # Radial coordinate grid
        r_min = self.config.boundary_radius * 0.1
        r_max = self.config.boundary_radius * 2.0
        
        self.r_grid = np.linspace(r_min, r_max, self.config.grid_points)
        self.dr = self.r_grid[1] - self.r_grid[0]
        
        # Find boundary index
        self.boundary_index = np.argmin(np.abs(self.r_grid - self.config.boundary_radius))
        self.r_boundary = self.r_grid[self.boundary_index]
        
        # Time coordinate (for time-dependent analysis)
        self.t_ref = 0.0
        
        print(f"  Coordinate grid: {len(self.r_grid)} points from {r_min:.2e} to {r_max:.2e} m")
        print(f"  Boundary at r = {self.r_boundary:.2e} m (index {self.boundary_index})")
    
    def _setup_boundary_surface(self):
        """Setup boundary surface geometry."""
        if self.config.boundary_type == BoundaryType.SPHERICAL:
            # Spherical boundary: r = R
            self.surface_equation = lambda r, theta, phi: r - self.config.boundary_radius
            self.normal_vector = lambda r, theta, phi: np.array([1.0, 0.0, 0.0, 0.0])  # Radial normal
            
        elif self.config.boundary_type == BoundaryType.CYLINDRICAL:
            # Cylindrical boundary: ρ = R
            self.surface_equation = lambda rho, phi, z: rho - self.config.boundary_radius
            self.normal_vector = lambda rho, phi, z: np.array([0.0, 1.0, 0.0, 0.0])  # ρ normal
            
        elif self.config.boundary_type == BoundaryType.PLANAR:
            # Planar boundary: z = z₀
            self.surface_equation = lambda x, y, z: z - self.config.boundary_radius
            self.normal_vector = lambda x, y, z: np.array([0.0, 0.0, 0.0, 1.0])  # z normal
            
        else:
            # Custom boundary (placeholder)
            self.surface_equation = lambda r, theta, phi: r - self.config.boundary_radius
            self.normal_vector = lambda r, theta, phi: np.array([1.0, 0.0, 0.0, 0.0])
        
        print(f"  Surface type: {self.config.boundary_type.value}")
    
    def _initialize_junction_analysis(self):
        """Initialize junction condition computations."""
        # Compute metric values at boundary
        self.g_inside_boundary = self.metric_inside(self.r_boundary, self.t_ref)
        self.g_outside_boundary = self.metric_outside(self.r_boundary, self.t_ref)
        
        # Check metric continuity
        self.metric_jump = self.g_outside_boundary - self.g_inside_boundary
        self.metric_continuity_error = np.linalg.norm(self.metric_jump)
        
        print(f"  Metric continuity error: {self.metric_continuity_error:.2e}")
    
    def compute_extrinsic_curvature(self, metric_func: Callable, 
                                  side: str = 'inside') -> np.ndarray:
        """
        Compute extrinsic curvature tensor K_μν on boundary surface.
        
        K_μν = ½ n^ρ (∇_μ g_ρν + ∇_ν g_ρμ - ∇_ρ g_μν)
        
        Args:
            metric_func: Metric tensor function g_μν(r,t)
            side: 'inside' or 'outside' for boundary approach direction
            
        Returns:
            Extrinsic curvature tensor K_μν (4×4 matrix)
        """
        # Use finite differences for derivatives
        eps = self.dr * 0.1
        
        if side == 'inside':
            r_eval = self.r_boundary - eps
        else:
            r_eval = self.r_boundary + eps
        
        # Metric and derivatives at evaluation point
        g = metric_func(r_eval, self.t_ref)
        
        # Radial derivatives (finite difference)
        g_plus = metric_func(r_eval + eps, self.t_ref)
        g_minus = metric_func(r_eval - eps, self.t_ref)
        dg_dr = (g_plus - g_minus) / (2 * eps)
        
        # Normal vector (radial direction for spherical boundary)
        n = np.array([1.0, 0.0, 0.0, 0.0])  # (t, r, θ, φ) components
        
        # Extrinsic curvature (simplified calculation)
        K = np.zeros((4, 4))
        
        # For spherical symmetry: K_μν = ½ n^r ∂_r g_μν
        for mu in range(4):
            for nu in range(4):
                K[mu, nu] = 0.5 * n[1] * dg_dr[mu, nu]  # n^r = 1 for radial normal
        
        return K
    
    def polymer_corrected_curvature(self, K_classical: np.ndarray) -> np.ndarray:
        """
        Apply LQG polymer corrections to extrinsic curvature.
        
        K_μν^poly = K_μν^classical × [1 - α sin²(μK/μ₀)]
        
        Args:
            K_classical: Classical extrinsic curvature tensor
            
        Returns:
            Polymer-corrected extrinsic curvature
        """
        if not self.config.apply_polymer_corrections:
            return K_classical
        
        # Characteristic curvature scale
        K_scale = np.linalg.norm(K_classical)
        mu_0 = self.config.polymer_scale
        
        # Polymer correction factor
        if K_scale > 0:
            argument = K_scale / mu_0
            correction_factor = 1.0 - self.config.alpha_polymer * np.sin(argument)**2
        else:
            correction_factor = 1.0
        
        K_polymer = correction_factor * K_classical
        
        return K_polymer
    
    def casimir_stress_tensor(self) -> np.ndarray:
        """
        Compute Casimir stress-energy tensor for parallel plates.
        
        T_μν^Casimir = ρ_Casimir × diag(-1, 1, 1, 1)
        
        Returns:
            Casimir stress-energy tensor T_μν (4×4 matrix)
        """
        if not self.config.apply_casimir_corrections:
            return np.zeros((4, 4))
        
        # Casimir energy density between parallel plates
        a = self.config.plate_separation
        rho_casimir = -(np.pi**2 * self.config.hbar * self.config.c) / (720 * a**4)
        
        # Multi-plate enhancement (√N scaling)
        if self.config.casimir_plates > 1:
            enhancement = np.sqrt(self.config.casimir_plates)
            rho_casimir *= enhancement
        
        # Apply coupling strength
        rho_casimir *= self.config.casimir_coupling
        
        # Casimir stress-energy tensor: T_μν = ρ diag(-1, 1, 1, 1)
        T_casimir = np.zeros((4, 4))
        T_casimir[0, 0] = -rho_casimir  # Energy density
        T_casimir[1, 1] = rho_casimir   # Radial pressure
        T_casimir[2, 2] = rho_casimir   # Tangential pressure
        T_casimir[3, 3] = rho_casimir   # Tangential pressure
        
        return T_casimir
    
    def israel_junction_conditions(self) -> Dict:
        """
        Compute Israel junction conditions [K_μν] = 8πG S_μν.
        
        Returns:
            Junction condition analysis results
        """
        print("Computing Israel junction conditions...")
        
        # Compute extrinsic curvature on both sides
        K_inside = self.compute_extrinsic_curvature(self.metric_inside, 'inside')
        K_outside = self.compute_extrinsic_curvature(self.metric_outside, 'outside')
        
        # Apply polymer corrections
        K_inside_poly = self.polymer_corrected_curvature(K_inside)
        K_outside_poly = self.polymer_corrected_curvature(K_outside)
        
        # Jump in extrinsic curvature
        K_jump = K_outside_poly - K_inside_poly
        
        # Surface stress-energy tensor S_μν from Israel equation
        # [K_μν] = 8πG (S_μν - ½ h_μν S) where h_μν is induced metric
        
        # Induced metric on boundary (3D)
        h_boundary = self.g_inside_boundary[1:, 1:]  # Remove time components for spatial surface
        
        # Surface stress-energy (simplified)
        S_surface = K_jump / (8 * np.pi * self.config.G)
        
        # Add Casimir contributions
        T_casimir = self.casimir_stress_tensor()
        
        # Total source term
        total_source = S_surface + T_casimir * self.config.junction_source_strength
        
        # Constraint satisfaction
        constraint_violation = np.linalg.norm(K_jump - 8 * np.pi * self.config.G * total_source)
        
        return {
            'K_inside': K_inside,
            'K_outside': K_outside,
            'K_inside_polymer': K_inside_poly,
            'K_outside_polymer': K_outside_poly,
            'K_jump': K_jump,
            'surface_stress_energy': S_surface,
            'casimir_stress_energy': T_casimir,
            'total_source': total_source,
            'constraint_violation': constraint_violation,
            'israel_satisfied': constraint_violation < self.config.integration_tolerance
        }
    
    def metric_continuity_analysis(self) -> Dict:
        """
        Analyze metric continuity conditions [g_μν] = 0.
        
        Returns:
            Metric continuity analysis results
        """
        print("Analyzing metric continuity conditions...")
        
        # Metric values at boundary
        g_in = self.g_inside_boundary
        g_out = self.g_outside_boundary
        
        # Jump conditions
        metric_jump = g_out - g_in
        
        # Component-wise analysis
        component_errors = {}
        for mu in range(4):
            for nu in range(mu, 4):  # Symmetric tensor
                component_errors[f'g_{mu}{nu}'] = metric_jump[mu, nu]
        
        # Overall continuity measure
        continuity_error = np.linalg.norm(metric_jump)
        continuity_satisfied = continuity_error < self.config.integration_tolerance
        
        # Determinant jump (coordinate invariant measure)
        det_in = np.linalg.det(g_in)
        det_out = np.linalg.det(g_out)
        determinant_jump = det_out - det_in
        
        return {
            'metric_inside': g_in,
            'metric_outside': g_out,
            'metric_jump': metric_jump,
            'component_errors': component_errors,
            'continuity_error': continuity_error,
            'continuity_satisfied': continuity_satisfied,
            'determinant_jump': determinant_jump
        }
    
    def thin_shell_analysis(self) -> Dict:
        """
        Analyze thin shell junction with delta function sources.
        
        Returns:
            Thin shell analysis results
        """
        print("Analyzing thin shell junction conditions...")
        
        # Surface energy-momentum tensor with delta function
        # T_μν = S_μν δ(surface)
        
        # Integrate across shell thickness
        shell_thickness = self.config.boundary_thickness
        
        # Surface density (integrated over thickness)
        surface_density = self.config.junction_source_strength * shell_thickness
        
        # Shell stress-energy tensor
        S_shell = np.zeros((4, 4))
        S_shell[0, 0] = surface_density      # Surface energy density
        S_shell[1, 1] = -surface_density/3   # Radial stress
        S_shell[2, 2] = -surface_density/3   # Tangential stress  
        S_shell[3, 3] = -surface_density/3   # Tangential stress
        
        # Add Casimir contribution
        if self.config.apply_casimir_corrections:
            T_casimir = self.casimir_stress_tensor()
            S_shell += T_casimir
        
        # Junction condition: [K_μν] = 8πG S_μν
        K_jump_required = 8 * np.pi * self.config.G * S_shell
        
        # Compute actual jump
        israel_results = self.israel_junction_conditions()
        K_jump_actual = israel_results['K_jump']
        
        # Match condition
        shell_error = np.linalg.norm(K_jump_actual - K_jump_required)
        shell_satisfied = shell_error < self.config.integration_tolerance
        
        return {
            'shell_thickness': shell_thickness,
            'surface_density': surface_density,
            'shell_stress_energy': S_shell,
            'required_K_jump': K_jump_required,
            'actual_K_jump': K_jump_actual,
            'shell_error': shell_error,
            'shell_satisfied': shell_satisfied
        }
    
    def comprehensive_junction_analysis(self) -> Dict:
        """
        Perform comprehensive junction condition analysis.
        
        Returns:
            Complete junction analysis results
        """
        print(f"Performing comprehensive junction analysis...")
        
        results = {
            'boundary_radius': self.config.boundary_radius,
            'polymer_corrections': self.config.apply_polymer_corrections,
            'casimir_corrections': self.config.apply_casimir_corrections
        }
        
        # Metric continuity
        if MatchingCondition.CONTINUOUS in self.config.matching_types:
            continuity = self.metric_continuity_analysis()
            results['metric_continuity'] = continuity
        
        # Israel conditions
        if MatchingCondition.ISRAEL in self.config.matching_types:
            israel = self.israel_junction_conditions()
            results['israel_conditions'] = israel
        
        # Thin shell
        if MatchingCondition.THIN_SHELL in self.config.matching_types:
            thin_shell = self.thin_shell_analysis()
            results['thin_shell'] = thin_shell
        
        # Overall satisfaction
        all_satisfied = True
        error_summary = []
        
        if 'metric_continuity' in results:
            satisfied = results['metric_continuity']['continuity_satisfied']
            error = results['metric_continuity']['continuity_error']
            all_satisfied &= satisfied
            error_summary.append(f"Metric continuity: {error:.2e}")
        
        if 'israel_conditions' in results:
            satisfied = results['israel_conditions']['israel_satisfied']
            error = results['israel_conditions']['constraint_violation']
            all_satisfied &= satisfied
            error_summary.append(f"Israel conditions: {error:.2e}")
        
        if 'thin_shell' in results:
            satisfied = results['thin_shell']['shell_satisfied']
            error = results['thin_shell']['shell_error']
            all_satisfied &= satisfied
            error_summary.append(f"Thin shell: {error:.2e}")
        
        results['all_conditions_satisfied'] = all_satisfied
        results['error_summary'] = error_summary
        
        print(f"  All conditions satisfied: {all_satisfied}")
        for error_msg in error_summary:
            print(f"    {error_msg}")
        
        return results
    
    def optimize_junction_parameters(self, target_variables: List[str]) -> Dict:
        """
        Optimize junction parameters to satisfy matching conditions.
        
        Args:
            target_variables: List of parameter names to optimize
            
        Returns:
            Optimization results
        """
        print(f"Optimizing junction parameters: {target_variables}")
        
        def objective(params):
            """Objective function: minimize junction condition violations."""
            # Update configuration with new parameters
            old_values = {}
            for i, var_name in enumerate(target_variables):
                old_values[var_name] = getattr(self.config, var_name)
                setattr(self.config, var_name, params[i])
            
            try:
                # Compute junction analysis
                analysis = self.comprehensive_junction_analysis()
                
                # Total error
                total_error = 0.0
                if 'metric_continuity' in analysis:
                    total_error += analysis['metric_continuity']['continuity_error']
                if 'israel_conditions' in analysis:
                    total_error += analysis['israel_conditions']['constraint_violation']
                if 'thin_shell' in analysis:
                    total_error += analysis['thin_shell']['shell_error']
                
                return total_error
                
            except Exception as e:
                # Restore old values on error
                for var_name, old_val in old_values.items():
                    setattr(self.config, var_name, old_val)
                return 1e6  # Large penalty
            finally:
                # Always restore old values for next iteration
                for var_name, old_val in old_values.items():
                    setattr(self.config, var_name, old_val)
        
        # Initial parameter values
        initial_params = [getattr(self.config, var_name) for var_name in target_variables]
        
        # Parameter bounds (10% variation around initial values)
        bounds = []
        for param in initial_params:
            lower = param * 0.9 if param > 0 else param * 1.1
            upper = param * 1.1 if param > 0 else param * 0.9
            bounds.append((lower, upper))
        
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.config.max_iterations}
            )
            
            if result.success:
                # Update configuration with optimal values
                optimal_params = {}
                for i, var_name in enumerate(target_variables):
                    optimal_params[var_name] = result.x[i]
                    setattr(self.config, var_name, result.x[i])
                
                # Final analysis with optimal parameters
                final_analysis = self.comprehensive_junction_analysis()
                
                return {
                    'success': True,
                    'optimal_parameters': optimal_params,
                    'final_error': result.fun,
                    'optimization_iterations': result.nit,
                    'final_analysis': final_analysis
                }
            else:
                return {
                    'success': False,
                    'error': result.message,
                    'initial_parameters': dict(zip(target_variables, initial_params))
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'initial_parameters': dict(zip(target_variables, initial_params))
            }

# Utility functions
def create_test_metrics() -> Tuple[Callable, Callable]:
    """
    Create test metric functions for demonstration.
    
    Returns:
        (metric_inside, metric_outside) function pair
    """
    def metric_inside(r, t):
        """Simple interior metric (approximately flat)."""
        g = np.zeros((4, 4))
        g[0, 0] = -1.0              # -dt²
        g[1, 1] = 1.0               # dr²
        g[2, 2] = r**2              # r²dθ²
        g[3, 3] = r**2 * np.sin(np.pi/4)**2  # r²sin²θ dφ²
        return g
    
    def metric_outside(r, t):
        """Simple exterior metric (weak field approximation)."""
        G = 6.67430e-11
        M = 1e-15  # Small mass scale
        rs = 2 * G * M / (299792458**2)  # Schwarzschild radius
        
        g = np.zeros((4, 4))
        g[0, 0] = -(1 - rs/r)       # -dt²
        g[1, 1] = 1/(1 - rs/r)      # dr²  
        g[2, 2] = r**2              # r²dθ²
        g[3, 3] = r**2 * np.sin(np.pi/4)**2  # r²sin²θ dφ²
        return g
    
    return metric_inside, metric_outside

if __name__ == "__main__":
    # Demonstration of junction-condition perturbations
    print("Junction-Condition Perturbations Demonstration")
    print("=" * 50)
    
    # Create test metrics
    metric_inside, metric_outside = create_test_metrics()
    
    # Configuration
    config = JunctionConfig(
        boundary_type=BoundaryType.SPHERICAL,
        boundary_radius=1e-9,
        boundary_thickness=1e-10,
        polymer_scale=1e-35,
        alpha_polymer=0.1,
        casimir_plates=3,
        plate_separation=1e-9,
        apply_polymer_corrections=True,
        apply_casimir_corrections=True,
        matching_types=[
            MatchingCondition.CONTINUOUS,
            MatchingCondition.ISRAEL,
            MatchingCondition.THIN_SHELL
        ]
    )
    
    # Initialize junction analyzer
    junction = JunctionConditions(config, metric_inside, metric_outside)
    
    print(f"\nBoundary setup complete:")
    print(f"  Boundary radius: {config.boundary_radius:.2e} m")
    print(f"  Grid points: {len(junction.r_grid)}")
    
    # Comprehensive junction analysis
    analysis = junction.comprehensive_junction_analysis()
    
    print(f"\nJunction Analysis Results:")
    print(f"  All conditions satisfied: {analysis['all_conditions_satisfied']}")
    
    if 'metric_continuity' in analysis:
        continuity = analysis['metric_continuity']
        print(f"  Metric continuity error: {continuity['continuity_error']:.2e}")
        print(f"  Determinant jump: {continuity['determinant_jump']:.2e}")
    
    if 'israel_conditions' in analysis:
        israel = analysis['israel_conditions']
        print(f"  Israel constraint violation: {israel['constraint_violation']:.2e}")
        print(f"  K jump magnitude: {np.linalg.norm(israel['K_jump']):.2e}")
        
        # Polymer correction effects
        K_classical_norm = np.linalg.norm(israel['K_inside'] - israel['K_outside'])
        K_polymer_norm = np.linalg.norm(israel['K_inside_polymer'] - israel['K_outside_polymer'])
        if K_classical_norm > 0:
            polymer_effect = (K_polymer_norm - K_classical_norm) / K_classical_norm * 100
            print(f"  Polymer correction effect: {polymer_effect:.1f}%")
    
    if 'thin_shell' in analysis:
        shell = analysis['thin_shell']
        print(f"  Thin shell error: {shell['shell_error']:.2e}")
        print(f"  Surface density: {shell['surface_density']:.2e} kg/m²")
    
    # Casimir effect analysis
    T_casimir = junction.casimir_stress_tensor()
    casimir_energy_density = abs(T_casimir[0, 0])
    
    print(f"\nCasimir Effect Analysis:")
    print(f"  Energy density: {casimir_energy_density:.2e} J/m³")
    print(f"  Plate separation: {config.plate_separation:.2e} m")
    print(f"  Number of plates: {config.casimir_plates}")
    
    # Parameter optimization
    print(f"\nOptimizing junction parameters...")
    
    optimization_targets = ['alpha_polymer', 'casimir_coupling', 'junction_source_strength']
    optimization = junction.optimize_junction_parameters(optimization_targets)
    
    print(f"Optimization Results:")
    if optimization['success']:
        print(f"  Success: {optimization['success']}")
        print(f"  Final error: {optimization['final_error']:.2e}")
        print(f"  Iterations: {optimization['optimization_iterations']}")
        print(f"  Optimal parameters:")
        for param, value in optimization['optimal_parameters'].items():
            print(f"    {param}: {value:.3e}")
        
        final_satisfied = optimization['final_analysis']['all_conditions_satisfied']
        print(f"  Final conditions satisfied: {final_satisfied}")
    else:
        print(f"  Optimization failed: {optimization.get('error', 'Unknown error')}")
    
    print("\n✅ Junction-condition perturbations demonstration complete!")
