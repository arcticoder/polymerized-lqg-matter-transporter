"""
Newton-Raphson Iterative Solver for Enhanced Stargate Transporter

This module implements Newton-Raphson iterative methods for nonlinear
Einstein field equations with adaptive step sizing and convergence acceleration.

Mathematical Framework:
    F(φ) = G_μν[φ] - 8πT_μν = 0
    Newton update: φ^(n+1) = φ^n - [J_F(φ^n)]^(-1) F(φ^n)
    
Where J_F is the Jacobian of the Einstein tensor operator.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, vmap, hessian
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import time
from scipy.sparse.linalg import spsolve, LinearOperator
from scipy.sparse import csc_matrix
import warnings

# Import our enhanced transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter

class NewtonRaphsonIterativeSolver:
    """
    Newton-Raphson solver for nonlinear Einstein field equations.
    
    Provides quadratic convergence for strongly nonlinear spacetime
    geometries with adaptive damping and line search algorithms.
    """
    
    def __init__(self, transporter: EnhancedStargateTransporter,
                 solver_config: Optional[Dict] = None,
                 field_config: Optional[Dict] = None):
        """
        Initialize Newton-Raphson iterative solver.
        
        Args:
            transporter: Enhanced stargate transporter instance
            solver_config: Solver-specific configuration
            field_config: Field discretization configuration
        """
        self.transporter = transporter
        
        # Default solver configuration
        if solver_config is None:
            solver_config = {
                'max_iterations': 100,        # Maximum Newton iterations
                'tolerance': 1e-10,           # Convergence tolerance
                'initial_damping': 1.0,       # Initial damping parameter
                'min_damping': 1e-4,          # Minimum damping parameter
                'damping_reduction': 0.5,     # Damping reduction factor
                'line_search': True,          # Enable line search
                'jacobian_method': 'forward', # 'forward', 'reverse', 'central'
                'preconditioning': True,      # Enable preconditioning
                'acceleration': 'anderson'    # 'none', 'aitken', 'anderson'
            }
        self.solver_config = solver_config
        
        # Default field configuration
        if field_config is None:
            field_config = {
                'nx': 64, 'ny': 64, 'nz': 64,  # Grid resolution
                'field_components': 10,         # Metric tensor components (symmetric 4x4)
                'boundary_conditions': 'asymptotically_flat'
            }
        self.field_config = field_config
        
        # Setup computational grid
        self.grid = self._setup_computational_grid()
        
        # Initialize field variables
        self.field_dimension = (field_config['nx'] * field_config['ny'] * 
                               field_config['nz'] * field_config['field_components'])
        
        # Precompiled functions
        self._setup_compiled_functions()
        
        # Iteration tracking
        self.iteration_history = []
        self.residual_history = []
        self.damping_history = []
        self.jacobian_condition_numbers = []
        
        # Anderson acceleration state
        if solver_config['acceleration'] == 'anderson':
            self.anderson_memory = 5  # Memory depth
            self.anderson_history = []
            
        print(f"NewtonRaphsonIterativeSolver initialized:")
        print(f"  Grid resolution: {field_config['nx']}×{field_config['ny']}×{field_config['nz']}")
        print(f"  Field components: {field_config['field_components']}")
        print(f"  Total DOFs: {self.field_dimension:,}")
        print(f"  Jacobian method: {solver_config['jacobian_method']}")
        print(f"  Acceleration: {solver_config['acceleration']}")
        
    def _setup_computational_grid(self) -> Dict:
        """Setup computational grid for field discretization."""
        
        config = self.transporter.config
        nx, ny, nz = self.field_config['nx'], self.field_config['ny'], self.field_config['nz']
        
        # Physical domain (asymptotically flat boundary conditions)
        x_max = config.R_payload * 3  # Extend beyond payload
        y_max = config.R_payload * 3
        z_max = config.L_corridor
        
        # Create coordinate arrays
        x = jnp.linspace(-x_max, x_max, nx)
        y = jnp.linspace(-y_max, y_max, ny)
        z = jnp.linspace(-z_max/2, z_max/2, nz)
        
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        
        # Create meshgrid
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        return {
            'x': x, 'y': y, 'z': z,
            'X': X, 'Y': Y, 'Z': Z,
            'dx': dx, 'dy': dy, 'dz': dz,
            'nx': nx, 'ny': ny, 'nz': nz
        }
    
    def _setup_compiled_functions(self):
        """Setup JIT-compiled functions for performance."""
        
        # Einstein tensor function
        @jit
        def einstein_tensor_residual(field_vector, stress_energy_vector):
            """Compute Einstein tensor residual F(φ) = G_μν - 8πT_μν."""
            return self._compute_einstein_residual(field_vector, stress_energy_vector)
        
        # Jacobian computation
        if self.solver_config['jacobian_method'] == 'forward':
            jacobian_func = jacfwd(einstein_tensor_residual)
        elif self.solver_config['jacobian_method'] == 'reverse':
            jacobian_func = jacrev(einstein_tensor_residual)
        else:  # central differences
            jacobian_func = lambda f, s: self._central_difference_jacobian(f, s)
            
        @jit
        def compute_jacobian(field_vector, stress_energy_vector):
            """Compute Jacobian matrix."""
            return jacobian_func(field_vector, stress_energy_vector)
        
        # Newton step computation
        @jit
        def newton_step(field_vector, residual_vector, jacobian_matrix):
            """Compute Newton step: Δφ = -J^(-1) F."""
            return self._solve_linear_system(jacobian_matrix, -residual_vector)
        
        # Line search function
        @jit
        def line_search_step(field_vector, newton_direction, stress_energy_vector, damping):
            """Perform line search to ensure convergence."""
            return self._armijo_line_search(field_vector, newton_direction, 
                                          stress_energy_vector, damping)
        
        # Store compiled functions
        self.einstein_tensor_residual = einstein_tensor_residual
        self.compute_jacobian = compute_jacobian
        self.newton_step = newton_step
        self.line_search_step = line_search_step
        
        print(f"  ✅ Compiled functions ready")
    
    @jit
    def _compute_einstein_residual(self, field_vector: jnp.ndarray, 
                                 stress_energy_vector: jnp.ndarray) -> jnp.ndarray:
        """Compute Einstein tensor residual for the field equation."""
        
        # Reshape field vector to metric tensor components
        nx, ny, nz = self.grid['nx'], self.grid['ny'], self.grid['nz']
        n_components = self.field_config['field_components']
        
        field_3d = field_vector.reshape((nx, ny, nz, n_components))
        stress_energy_3d = stress_energy_vector.reshape((nx, ny, nz, n_components))
        
        # Compute Einstein tensor from metric components (simplified)
        einstein_tensor = self._compute_discretized_einstein_tensor(field_3d)
        
        # Einstein field equation: G_μν = 8π T_μν
        residual = einstein_tensor - 8 * jnp.pi * stress_energy_3d
        
        return residual.flatten()
    
    @jit
    def _compute_discretized_einstein_tensor(self, metric_components: jnp.ndarray) -> jnp.ndarray:
        """Compute Einstein tensor using finite difference discretization."""
        
        dx, dy, dz = self.grid['dx'], self.grid['dy'], self.grid['dz']
        
        # Simplified Einstein tensor computation
        # In practice, would compute Christoffel symbols and Riemann curvature
        
        # Mock Einstein tensor based on second derivatives of metric
        einstein_tensor = jnp.zeros_like(metric_components)
        
        for component in range(metric_components.shape[-1]):
            g = metric_components[:, :, :, component]
            
            # Second derivatives (Laplacian)
            d2g_dx2 = (jnp.roll(g, -1, axis=0) - 2*g + jnp.roll(g, 1, axis=0)) / dx**2
            d2g_dy2 = (jnp.roll(g, -1, axis=1) - 2*g + jnp.roll(g, 1, axis=1)) / dy**2
            d2g_dz2 = (jnp.roll(g, -1, axis=2) - 2*g + jnp.roll(g, 1, axis=2)) / dz**2
            
            # Simplified Einstein tensor component
            G_component = d2g_dx2 + d2g_dy2 + d2g_dz2
            
            # Add nonlinear terms (simplified)
            G_component = G_component + 0.1 * g**2 - 0.05 * g**3
            
            einstein_tensor = einstein_tensor.at[:, :, :, component].set(G_component)
            
        return einstein_tensor
    
    @jit
    def _solve_linear_system(self, jacobian: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        """Solve linear system J·Δφ = rhs using appropriate method."""
        
        # For demonstration, use direct solve (in practice, would use iterative)
        try:
            # Check if Jacobian is well-conditioned
            condition_number = jnp.linalg.cond(jacobian)
            
            if condition_number < 1e12:
                # Direct solve
                solution = jnp.linalg.solve(jacobian, rhs)
            else:
                # Use pseudo-inverse for ill-conditioned systems
                solution = jnp.linalg.pinv(jacobian) @ rhs
                
            return solution
            
        except:
            # Fallback to gradient descent step
            return -rhs * 1e-6
    
    @jit
    def _armijo_line_search(self, field: jnp.ndarray, direction: jnp.ndarray,
                           stress_energy: jnp.ndarray, damping: float) -> Tuple[jnp.ndarray, float]:
        """Armijo line search for step size selection."""
        
        c1 = 1e-4  # Armijo constant
        max_backtracks = 10
        
        # Initial residual norm
        initial_residual = self.einstein_tensor_residual(field, stress_energy)
        initial_norm = jnp.linalg.norm(initial_residual)
        
        # Directional derivative
        directional_derivative = -jnp.linalg.norm(initial_residual)**2
        
        alpha = damping
        for i in range(max_backtracks):
            # Test step
            test_field = field + alpha * direction
            test_residual = self.einstein_tensor_residual(test_field, stress_energy)
            test_norm = jnp.linalg.norm(test_residual)
            
            # Armijo condition
            armijo_rhs = initial_norm + c1 * alpha * directional_derivative
            
            if test_norm <= armijo_rhs:
                return test_field, alpha
                
            # Reduce step size
            alpha *= 0.5
            
        # If line search fails, return original damped step
        return field + alpha * direction, alpha
    
    def _anderson_acceleration(self, field_new: jnp.ndarray, 
                              field_old: jnp.ndarray) -> jnp.ndarray:
        """Apply Anderson acceleration for convergence acceleration."""
        
        # Add to history
        self.anderson_history.append({
            'field': field_new.copy(),
            'residual': field_new - field_old
        })
        
        # Keep only recent history
        if len(self.anderson_history) > self.anderson_memory:
            self.anderson_history.pop(0)
            
        if len(self.anderson_history) < 2:
            return field_new
            
        # Anderson acceleration calculation
        m = len(self.anderson_history) - 1  # Number of previous iterations
        
        # Build residual differences matrix
        residual_diffs = []
        field_diffs = []
        
        for i in range(1, len(self.anderson_history)):
            prev_res = self.anderson_history[i-1]['residual']
            curr_res = self.anderson_history[i]['residual']
            residual_diffs.append(curr_res - prev_res)
            
            prev_field = self.anderson_history[i-1]['field']
            curr_field = self.anderson_history[i]['field']
            field_diffs.append(curr_field - prev_field)
            
        if not residual_diffs:
            return field_new
            
        # Solve least squares problem for Anderson coefficients
        R_matrix = jnp.stack(residual_diffs, axis=1)  # Stack as columns
        current_residual = self.anderson_history[-1]['residual']
        
        try:
            # Solve R @ alpha = r for alpha
            alpha = jnp.linalg.lstsq(R_matrix.T, current_residual, rcond=None)[0]
            
            # Anderson update
            field_accelerated = field_new.copy()
            for i, coeff in enumerate(alpha):
                field_accelerated = field_accelerated - coeff * jnp.array(field_diffs[i])
                
            return field_accelerated
            
        except:
            # If Anderson fails, return unaccelerated
            return field_new
    
    def solve_nonlinear_field(self, stress_energy_function: Callable, t: float,
                             initial_field: Optional[jnp.ndarray] = None) -> Dict:
        """
        Solve nonlinear Einstein field equation using Newton-Raphson method.
        
        Args:
            stress_energy_function: Stress-energy tensor function T_μν(x,y,z,t)
            t: Current time
            initial_field: Initial field guess
            
        Returns:
            Solution and convergence analysis
        """
        start_time = time.time()
        
        print(f"\n🧮 NEWTON-RAPHSON NONLINEAR SOLVER")
        print("-" * 50)
        print(f"Time: {t:.3f}s")
        print(f"DOFs: {self.field_dimension:,}")
        print(f"Max iterations: {self.solver_config['max_iterations']}")
        print(f"Tolerance: {self.solver_config['tolerance']:.1e}")
        
        # Initialize field
        if initial_field is None:
            # Start with flat spacetime + small perturbation
            field = jnp.zeros(self.field_dimension)
            # Add small random perturbation
            key = jax.random.PRNGKey(42)
            perturbation = jax.random.normal(key, (self.field_dimension,)) * 1e-8
            field = field + perturbation
        else:
            field = initial_field.flatten()
            
        # Compute stress-energy tensor
        stress_energy = stress_energy_function(self.grid['X'], self.grid['Y'], 
                                             self.grid['Z'], t)
        stress_energy_vector = stress_energy.flatten()
        
        # Newton-Raphson iteration
        max_iterations = self.solver_config['max_iterations']
        tolerance = self.solver_config['tolerance']
        damping = self.solver_config['initial_damping']
        
        residual_norm = float('inf')
        iteration = 0
        convergence_achieved = False
        
        print(f"Starting Newton-Raphson iterations...")
        iteration_start = time.time()
        
        while iteration < max_iterations and residual_norm > tolerance:
            iteration_start_single = time.time()
            
            # Compute residual
            residual = self.einstein_tensor_residual(field, stress_energy_vector)
            residual_norm = float(jnp.linalg.norm(residual))
            
            # Check convergence
            if residual_norm <= tolerance:
                convergence_achieved = True
                break
                
            # Compute Jacobian
            jacobian_start = time.time()
            jacobian = self.compute_jacobian(field, stress_energy_vector)
            jacobian_time = time.time() - jacobian_start
            
            # Condition number analysis
            try:
                condition_number = float(jnp.linalg.cond(jacobian))
                self.jacobian_condition_numbers.append(condition_number)
            except:
                condition_number = float('inf')
                
            # Compute Newton step
            newton_direction = self.newton_step(field, residual, jacobian)
            
            # Line search if enabled
            if self.solver_config['line_search']:
                field_new, actual_damping = self.line_search_step(
                    field, newton_direction, stress_energy_vector, damping
                )
            else:
                field_new = field + damping * newton_direction
                actual_damping = damping
                
            # Apply acceleration if enabled
            if self.solver_config['acceleration'] == 'anderson' and iteration > 0:
                field_new = self._anderson_acceleration(field_new, field)
            elif self.solver_config['acceleration'] == 'aitken' and iteration > 1:
                # Aitken acceleration (simplified)
                if len(self.iteration_history) >= 2:
                    prev_change = field - self.iteration_history[-1]
                    curr_change = field_new - field
                    
                    if jnp.linalg.norm(curr_change - prev_change) > 1e-14:
                        aitken_factor = jnp.dot(prev_change, curr_change - prev_change) / jnp.linalg.norm(curr_change - prev_change)**2
                        field_new = field - aitken_factor * prev_change
            
            # Update field
            field = field_new
            
            # Adaptive damping
            if iteration > 0 and residual_norm > self.residual_history[-1]:
                damping = max(self.solver_config['min_damping'], 
                            damping * self.solver_config['damping_reduction'])
            else:
                damping = min(1.0, damping * 1.1)  # Increase damping if convergence is good
                
            # Store iteration data
            iteration_time_single = time.time() - iteration_start_single
            self.iteration_history.append(field.copy())
            self.residual_history.append(residual_norm)
            self.damping_history.append(actual_damping)
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: residual = {residual_norm:.2e}, "
                      f"damping = {actual_damping:.2e}, "
                      f"cond = {condition_number:.1e}, "
                      f"time = {iteration_time_single:.3f}s")
                
            iteration += 1
        
        total_iteration_time = time.time() - iteration_start
        total_time = time.time() - start_time
        
        # Reshape solution back to field format
        nx, ny, nz, nc = (self.grid['nx'], self.grid['ny'], 
                         self.grid['nz'], self.field_config['field_components'])
        solution_field = field.reshape((nx, ny, nz, nc))
        
        print(f"\n📊 NEWTON-RAPHSON COMPLETE")
        print("-" * 50)
        print(f"Converged: {'✅' if convergence_achieved else '❌'}")
        print(f"Iterations: {iteration}")
        print(f"Final residual: {residual_norm:.2e}")
        print(f"Target tolerance: {tolerance:.2e}")
        print(f"Average condition number: {np.mean(self.jacobian_condition_numbers):.1e}")
        print(f"Final damping: {damping:.2e}")
        print(f"Iteration time: {total_iteration_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Time per iteration: {total_iteration_time/iteration:.3f}s")
        
        # Convergence analysis
        if len(self.residual_history) > 1:
            convergence_rate = self._analyze_convergence_rate()
        else:
            convergence_rate = None
            
        return {
            'solution': solution_field,
            'converged': convergence_achieved,
            'iterations': iteration,
            'final_residual': residual_norm,
            'convergence_rate': convergence_rate,
            'condition_numbers': self.jacobian_condition_numbers,
            'timing': {
                'total_time': total_time,
                'iteration_time': total_iteration_time,
                'time_per_iteration': total_iteration_time / iteration if iteration > 0 else 0
            },
            'performance': {
                'quadratic_convergence': self._check_quadratic_convergence(),
                'damping_effectiveness': self._analyze_damping_effectiveness(),
                'acceleration_benefit': self._estimate_acceleration_benefit()
            }
        }
    
    def _analyze_convergence_rate(self) -> Dict:
        """Analyze convergence rate from residual history."""
        
        residuals = np.array(self.residual_history)
        
        if len(residuals) < 3:
            return {'type': 'insufficient_data'}
            
        # Estimate convergence order
        log_residuals = np.log10(residuals + 1e-16)
        
        # Linear regression for convergence rate
        iterations = np.arange(len(residuals))
        
        # Fit exponential decay: log(r_n) ≈ log(r_0) + α*n
        if len(iterations) > 1:
            coeffs = np.polyfit(iterations, log_residuals, 1)
            linear_rate = -coeffs[0]
        else:
            linear_rate = 0
            
        # Check for quadratic convergence (Newton's method)
        quadratic_indicator = 0
        if len(residuals) > 3:
            for i in range(2, len(residuals)):
                if residuals[i-1] > 1e-15 and residuals[i-2] > 1e-15:
                    ratio = residuals[i] / (residuals[i-1]**2 / residuals[i-2])
                    if 0.1 < ratio < 10:  # Reasonable range for quadratic convergence
                        quadratic_indicator += 1
                        
        quadratic_percentage = quadratic_indicator / max(1, len(residuals) - 2) * 100
        
        return {
            'type': 'analysis_complete',
            'linear_rate': linear_rate,
            'quadratic_percentage': quadratic_percentage,
            'final_reduction': residuals[0] / residuals[-1] if residuals[-1] > 0 else float('inf')
        }
    
    def _check_quadratic_convergence(self) -> bool:
        """Check if quadratic convergence is achieved."""
        
        if len(self.residual_history) < 4:
            return False
            
        # Check last few iterations for quadratic behavior
        residuals = self.residual_history[-4:]
        
        for i in range(2, len(residuals)):
            if residuals[i-1] > 1e-15 and residuals[i-2] > 1e-15:
                expected_quadratic = residuals[i-1]**2 / residuals[i-2]
                actual = residuals[i]
                
                # Allow factor of 10 tolerance for quadratic convergence
                if not (0.1 * expected_quadratic <= actual <= 10 * expected_quadratic):
                    return False
                    
        return True
    
    def _analyze_damping_effectiveness(self) -> float:
        """Analyze effectiveness of adaptive damping."""
        
        if len(self.damping_history) < 2:
            return 0.0
            
        # Count how often damping helped convergence
        improvements = 0
        total_adaptations = 0
        
        for i in range(1, len(self.residual_history)):
            if self.damping_history[i] != self.damping_history[i-1]:
                total_adaptations += 1
                if self.residual_history[i] < self.residual_history[i-1]:
                    improvements += 1
                    
        return improvements / total_adaptations if total_adaptations > 0 else 1.0
    
    def _estimate_acceleration_benefit(self) -> float:
        """Estimate benefit from convergence acceleration."""
        
        # Simplified estimate based on convergence rate improvement
        if self.solver_config['acceleration'] == 'none':
            return 0.0
            
        if len(self.residual_history) < 5:
            return 0.0
            
        # Compare convergence in first half vs second half
        mid_point = len(self.residual_history) // 2
        first_half = self.residual_history[:mid_point]
        second_half = self.residual_history[mid_point:]
        
        if len(first_half) > 1 and len(second_half) > 1:
            first_rate = (first_half[0] / first_half[-1]) ** (1/len(first_half))
            second_rate = (second_half[0] / second_half[-1]) ** (1/len(second_half))
            
            return max(0, second_rate / first_rate - 1) * 100  # Percentage improvement
        else:
            return 0.0

def main():
    """Demonstration of Newton-Raphson iterative solver."""
    from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
    
    print("="*70)
    print("NEWTON-RAPHSON ITERATIVE SOLVER DEMONSTRATION")
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
    
    # Initialize Newton-Raphson solver
    solver_config = {
        'max_iterations': 50,
        'tolerance': 1e-8,
        'initial_damping': 0.8,
        'line_search': True,
        'jacobian_method': 'forward',
        'acceleration': 'anderson'
    }
    
    field_config = {
        'nx': 32, 'ny': 32, 'nz': 32,  # Reduced for demonstration
        'field_components': 6,  # Symmetric 4x4 metric has 10 components, use 6 for demo
        'boundary_conditions': 'asymptotically_flat'
    }
    
    solver = NewtonRaphsonIterativeSolver(transporter, solver_config, field_config)
    
    # Define stress-energy function
    def stress_energy_function(X, Y, Z, t):
        """Nonlinear stress-energy tensor for exotic matter."""
        r_cyl = jnp.sqrt(X**2 + Y**2)
        
        # Nonlinear source with time dependence
        base_source = jnp.exp(-(r_cyl - 1.5)**2 / 0.8) * jnp.exp(-(Z**2) / 30.0)
        
        # Add nonlinearity and time dependence
        nonlinear_factor = 1 + 0.1 * jnp.sin(t * 2 * jnp.pi) * base_source
        source = base_source * nonlinear_factor * 1e-5
        
        # Expand to field components (symmetric tensor)
        source_expanded = jnp.zeros(X.shape + (field_config['field_components'],))
        for i in range(field_config['field_components']):
            component_factor = 1.0 + 0.1 * (i % 3) / 3.0  # Different for each component
            source_expanded = source_expanded.at[:, :, :, i].set(source * component_factor)
            
        return source_expanded
    
    # Solve nonlinear field equation
    t = 1.5
    result = solver.solve_nonlinear_field(stress_energy_function, t)
    
    # Analyze results
    print(f"\n📊 SOLUTION ANALYSIS")
    print("-" * 50)
    solution = result['solution']
    print(f"Solution shape: {solution.shape}")
    print(f"Solution range: [{jnp.min(solution):.2e}, {jnp.max(solution):.2e}]")
    print(f"Solution norm: {jnp.linalg.norm(solution):.2e}")
    
    print(f"\n⚡ CONVERGENCE ANALYSIS")
    print("-" * 50)
    conv_rate = result['convergence_rate']
    perf = result['performance']
    
    if conv_rate and conv_rate['type'] == 'analysis_complete':
        print(f"Linear convergence rate: {conv_rate['linear_rate']:.2f}")
        print(f"Quadratic convergence: {conv_rate['quadratic_percentage']:.1f}%")
        print(f"Residual reduction: {conv_rate['final_reduction']:.1e}×")
    
    print(f"Quadratic convergence achieved: {'✅' if perf['quadratic_convergence'] else '❌'}")
    print(f"Damping effectiveness: {perf['damping_effectiveness']:.1%}")
    print(f"Acceleration benefit: {perf['acceleration_benefit']:.1f}%")
    
    print(f"\n🔧 JACOBIAN ANALYSIS")
    print("-" * 50)
    cond_numbers = result['condition_numbers']
    if cond_numbers:
        print(f"Average condition number: {np.mean(cond_numbers):.1e}")
        print(f"Max condition number: {np.max(cond_numbers):.1e}")
        print(f"Min condition number: {np.min(cond_numbers):.1e}")
        
        well_conditioned = np.mean(cond_numbers) < 1e8
        print(f"Well-conditioned: {'✅' if well_conditioned else '⚠️'}")
    
    # Test different acceleration methods
    print(f"\n🚀 ACCELERATION METHOD COMPARISON")
    print("-" * 50)
    
    acceleration_methods = ['none', 'aitken', 'anderson']
    for method in acceleration_methods:
        test_config = {**solver_config, 'acceleration': method, 'max_iterations': 30}
        test_solver = NewtonRaphsonIterativeSolver(transporter, test_config, field_config)
        
        test_result = test_solver.solve_nonlinear_field(stress_energy_function, t)
        
        print(f"{method:8s}: {test_result['iterations']:2d} iter, "
              f"residual = {test_result['final_residual']:.1e}, "
              f"time = {test_result['timing']['total_time']:.2f}s")
    
    target_performance = {
        'convergence': True,
        'quadratic_convergence': True,
        'final_residual': 1e-6,
        'max_iterations': 100
    }
    
    print(f"\n🎯 TARGET ACHIEVEMENT")
    print("-" * 50)
    print(f"Convergence: {'✅' if result['converged'] else '❌'} "
          f"(target: {'✅' if target_performance['convergence'] else '❌'})")
    print(f"Quadratic convergence: {'✅' if perf['quadratic_convergence'] else '❌'} "
          f"(target: {'✅' if target_performance['quadratic_convergence'] else '❌'})")
    print(f"Final residual: {result['final_residual']:.1e} "
          f"(target: ≤{target_performance['final_residual']:.1e}) "
          f"{'✅' if result['final_residual'] <= target_performance['final_residual'] else '⚠️'}")
    print(f"Iterations: {result['iterations']} "
          f"(target: ≤{target_performance['max_iterations']}) "
          f"{'✅' if result['iterations'] <= target_performance['max_iterations'] else '⚠️'}")
    
    return solver

if __name__ == "__main__":
    newton_solver = main()
