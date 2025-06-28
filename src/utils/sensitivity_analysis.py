#!/usr/bin/env python3
"""
Sensitivity-Gradient Analysis with JAX Automatic Differentiation
==============================================================

Advanced sensitivity analysis for matter transporter parameters using
JAX automatic differentiation. Provides gradient-based optimization
and uncertainty quantification for system performance.

Implements:
- Parameter sensitivity: ∂f/∂p_i for all parameters p_i
- Gradient-based optimization: p* = argmin f(p) using JAX gradients
- Hessian analysis: ∂²f/∂p_i∂p_j for second-order sensitivity
- Uncertainty propagation: δf ≈ Σ (∂f/∂p_i) δp_i + ½ Σ (∂²f/∂p_i∂p_j) δp_i δp_j

Mathematical Foundation:
Enhanced from unified-lqg repository gradient analysis methodology
- JAX provides exact gradients via automatic differentiation
- Forward and reverse mode AD for different computational patterns
- Efficient computation of high-order derivatives

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jacfwd, jacrev, vmap
    from jax.scipy.optimize import minimize as jax_minimize
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to numpy if JAX not available
    import numpy as jnp
    JAX_AVAILABLE = False
    print("Warning: JAX not available, using numpy fallback")

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings

class SensitivityMethod(Enum):
    """Enumeration of sensitivity analysis methods."""
    FINITE_DIFFERENCE = "finite_difference"
    AUTOMATIC_DIFF = "automatic_diff"
    COMPLEX_STEP = "complex_step"
    PERTURBATION = "perturbation"

@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    # Differentiation parameters
    method: SensitivityMethod = SensitivityMethod.AUTOMATIC_DIFF
    finite_diff_step: float = 1e-8      # Finite difference step size
    complex_step_size: float = 1e-15    # Complex step size
    
    # Parameter ranges and uncertainties
    parameter_bounds: Dict[str, Tuple[float, float]] = None
    parameter_uncertainties: Dict[str, float] = None
    nominal_values: Dict[str, float] = None
    
    # Analysis options
    compute_hessian: bool = True         # Compute second-order derivatives
    uncertainty_propagation: bool = True # Propagate parameter uncertainties
    monte_carlo_samples: int = 10000     # MC samples for uncertainty analysis
    
    # Optimization settings
    optimization_method: str = 'L-BFGS-B'
    max_iterations: int = 1000
    tolerance: float = 1e-8
    
    # JAX settings
    use_jax: bool = JAX_AVAILABLE        # Use JAX if available
    jit_compile: bool = True             # JIT compile JAX functions
    
    def __post_init__(self):
        """Initialize default values."""
        if self.parameter_bounds is None:
            self.parameter_bounds = {}
        if self.parameter_uncertainties is None:
            self.parameter_uncertainties = {}
        if self.nominal_values is None:
            self.nominal_values = {}

class SensitivityAnalyzer:
    """
    Sensitivity and gradient analysis framework.
    
    Provides comprehensive parameter sensitivity analysis:
    1. First-order gradients ∂f/∂p_i
    2. Second-order Hessian matrix ∂²f/∂p_i∂p_j  
    3. Uncertainty propagation through parameter space
    4. Gradient-based optimization with exact derivatives
    
    Parameters:
    -----------
    config : SensitivityConfig
        Sensitivity analysis configuration
    objective_function : Callable
        Function f(params) to analyze
    """
    
    def __init__(self, config: SensitivityConfig, objective_function: Callable):
        """
        Initialize sensitivity analyzer.
        
        Args:
            config: Sensitivity analysis configuration
            objective_function: Target function for analysis
        """
        self.config = config
        self.objective_function = objective_function
        
        # Setup differentiation method
        self._setup_differentiation()
        
        # Initialize parameter space
        self._initialize_parameters()
        
        print(f"Sensitivity analyzer initialized:")
        print(f"  Method: {config.method.value}")
        print(f"  JAX available: {JAX_AVAILABLE}")
        print(f"  Parameters: {len(self.parameter_names)}")
        print(f"  Hessian computation: {config.compute_hessian}")
    
    def _setup_differentiation(self):
        """Setup differentiation functions based on method."""
        if self.config.use_jax and JAX_AVAILABLE:
            # JAX automatic differentiation
            if self.config.jit_compile:
                self.objective_jax = jax.jit(self._jax_objective_wrapper)
                self.gradient_func = jax.jit(grad(self._jax_objective_wrapper))
                if self.config.compute_hessian:
                    self.hessian_func = jax.jit(hessian(self._jax_objective_wrapper))
            else:
                self.objective_jax = self._jax_objective_wrapper
                self.gradient_func = grad(self._jax_objective_wrapper)
                if self.config.compute_hessian:
                    self.hessian_func = hessian(self._jax_objective_wrapper)
            
            print(f"  Using JAX automatic differentiation (JIT: {self.config.jit_compile})")
            
        elif self.config.method == SensitivityMethod.FINITE_DIFFERENCE:
            self.gradient_func = self._finite_difference_gradient
            if self.config.compute_hessian:
                self.hessian_func = self._finite_difference_hessian
            print(f"  Using finite difference (step: {self.config.finite_diff_step:.2e})")
            
        elif self.config.method == SensitivityMethod.COMPLEX_STEP:
            self.gradient_func = self._complex_step_gradient
            print(f"  Using complex step differentiation (step: {self.config.complex_step_size:.2e})")
            
        else:
            # Fallback to finite differences
            self.gradient_func = self._finite_difference_gradient
            if self.config.compute_hessian:
                self.hessian_func = self._finite_difference_hessian
            print(f"  Fallback to finite difference method")
    
    def _jax_objective_wrapper(self, params_array: jnp.ndarray) -> float:
        """Wrapper for JAX compatibility."""
        # Convert JAX array to parameter dictionary
        params_dict = dict(zip(self.parameter_names, params_array))
        return self.objective_function(params_dict)
    
    def _initialize_parameters(self):
        """Initialize parameter space from configuration."""
        # Extract parameter names from bounds or uncertainties
        all_param_names = set()
        all_param_names.update(self.config.parameter_bounds.keys())
        all_param_names.update(self.config.parameter_uncertainties.keys())
        all_param_names.update(self.config.nominal_values.keys())
        
        self.parameter_names = sorted(list(all_param_names))
        
        # Set default values if not provided
        for param in self.parameter_names:
            if param not in self.config.nominal_values:
                if param in self.config.parameter_bounds:
                    bounds = self.config.parameter_bounds[param]
                    self.config.nominal_values[param] = (bounds[0] + bounds[1]) / 2
                else:
                    self.config.nominal_values[param] = 1.0
            
            if param not in self.config.parameter_uncertainties:
                if param in self.config.parameter_bounds:
                    bounds = self.config.parameter_bounds[param]
                    self.config.parameter_uncertainties[param] = (bounds[1] - bounds[0]) * 0.1
                else:
                    self.config.parameter_uncertainties[param] = 0.1 * abs(self.config.nominal_values[param])
        
        print(f"  Parameter names: {self.parameter_names}")
    
    def _params_dict_to_array(self, params_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        return np.array([params_dict[name] for name in self.parameter_names])
    
    def _params_array_to_dict(self, params_array: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return dict(zip(self.parameter_names, params_array))
    
    def compute_gradient(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Compute gradient ∂f/∂p_i for all parameters.
        
        Args:
            params: Parameter values
            
        Returns:
            Gradient dictionary {param_name: ∂f/∂param}
        """
        if self.config.use_jax and JAX_AVAILABLE:
            # JAX automatic differentiation
            params_array = self._params_dict_to_array(params)
            grad_array = self.gradient_func(params_array)
            return dict(zip(self.parameter_names, grad_array))
        else:
            # Fallback methods
            return self.gradient_func(params)
    
    def _finite_difference_gradient(self, params: Dict[str, float]) -> Dict[str, float]:
        """Compute gradient using finite differences."""
        gradient = {}
        
        for param_name in self.parameter_names:
            # Forward difference
            params_plus = params.copy()
            params_plus[param_name] += self.config.finite_diff_step
            
            params_minus = params.copy()
            params_minus[param_name] -= self.config.finite_diff_step
            
            f_plus = self.objective_function(params_plus)
            f_minus = self.objective_function(params_minus)
            
            gradient[param_name] = (f_plus - f_minus) / (2 * self.config.finite_diff_step)
        
        return gradient
    
    def _complex_step_gradient(self, params: Dict[str, float]) -> Dict[str, float]:
        """Compute gradient using complex step differentiation."""
        gradient = {}
        
        for param_name in self.parameter_names:
            # Complex step
            params_complex = params.copy()
            params_complex[param_name] = complex(params[param_name], self.config.complex_step_size)
            
            try:
                f_complex = self.objective_function(params_complex)
                gradient[param_name] = f_complex.imag / self.config.complex_step_size
            except (TypeError, AttributeError):
                # Fallback to finite difference if complex numbers not supported
                gradient[param_name] = self._finite_difference_gradient({param_name: params[param_name]})[param_name]
        
        return gradient
    
    def compute_hessian(self, params: Dict[str, float]) -> np.ndarray:
        """
        Compute Hessian matrix ∂²f/∂p_i∂p_j.
        
        Args:
            params: Parameter values
            
        Returns:
            Hessian matrix (n_params × n_params)
        """
        if not self.config.compute_hessian:
            warnings.warn("Hessian computation disabled in configuration")
            return np.zeros((len(self.parameter_names), len(self.parameter_names)))
        
        if self.config.use_jax and JAX_AVAILABLE:
            # JAX automatic differentiation
            params_array = self._params_dict_to_array(params)
            return np.array(self.hessian_func(params_array))
        else:
            # Finite difference Hessian
            return self._finite_difference_hessian(params)
    
    def _finite_difference_hessian(self, params: Dict[str, float]) -> np.ndarray:
        """Compute Hessian using finite differences."""
        n_params = len(self.parameter_names)
        hessian = np.zeros((n_params, n_params))
        
        for i, param_i in enumerate(self.parameter_names):
            for j, param_j in enumerate(self.parameter_names):
                if i <= j:  # Exploit symmetry
                    # Second-order finite difference
                    h_i = self.config.finite_diff_step
                    h_j = self.config.finite_diff_step
                    
                    # f(x+hi+hj)
                    params_pp = params.copy()
                    params_pp[param_i] += h_i
                    params_pp[param_j] += h_j
                    f_pp = self.objective_function(params_pp)
                    
                    # f(x+hi-hj)
                    params_pm = params.copy()
                    params_pm[param_i] += h_i
                    params_pm[param_j] -= h_j
                    f_pm = self.objective_function(params_pm)
                    
                    # f(x-hi+hj)
                    params_mp = params.copy()
                    params_mp[param_i] -= h_i
                    params_mp[param_j] += h_j
                    f_mp = self.objective_function(params_mp)
                    
                    # f(x-hi-hj)
                    params_mm = params.copy()
                    params_mm[param_i] -= h_i
                    params_mm[param_j] -= h_j
                    f_mm = self.objective_function(params_mm)
                    
                    # Mixed partial derivative
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h_i * h_j)
                    hessian[j, i] = hessian[i, j]  # Symmetry
        
        return hessian
    
    def sensitivity_analysis(self, params: Dict[str, float]) -> Dict:
        """
        Comprehensive sensitivity analysis at given parameter point.
        
        Args:
            params: Parameter values for analysis
            
        Returns:
            Sensitivity analysis results
        """
        print(f"Performing sensitivity analysis at parameter point...")
        
        # Compute gradient
        gradient = self.compute_gradient(params)
        
        # Compute Hessian if requested
        if self.config.compute_hessian:
            hessian = self.compute_hessian(params)
        else:
            hessian = None
        
        # Sensitivity measures
        gradient_array = np.array([gradient[name] for name in self.parameter_names])
        gradient_magnitude = np.linalg.norm(gradient_array)
        
        # Normalized sensitivities
        normalized_gradient = {}
        for i, param_name in enumerate(self.parameter_names):
            param_value = params[param_name]
            if param_value != 0:
                normalized_gradient[param_name] = gradient[param_name] * param_value
            else:
                normalized_gradient[param_name] = gradient[param_name]
        
        # Parameter rankings by sensitivity
        sensitivity_ranking = sorted(
            self.parameter_names, 
            key=lambda p: abs(normalized_gradient[p]), 
            reverse=True
        )
        
        results = {
            'gradient': gradient,
            'normalized_gradient': normalized_gradient,
            'gradient_magnitude': gradient_magnitude,
            'sensitivity_ranking': sensitivity_ranking,
            'hessian': hessian,
            'parameter_values': params.copy()
        }
        
        # Add Hessian analysis
        if hessian is not None:
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            results['hessian_eigenvalues'] = eigenvalues
            results['hessian_eigenvectors'] = eigenvectors
            results['hessian_condition_number'] = np.max(eigenvalues) / np.max([np.min(eigenvalues), 1e-12])
        
        print(f"  Gradient magnitude: {gradient_magnitude:.2e}")
        print(f"  Most sensitive parameter: {sensitivity_ranking[0]}")
        
        return results
    
    def uncertainty_propagation(self, params: Dict[str, float]) -> Dict:
        """
        Propagate parameter uncertainties to function uncertainty.
        
        Uses first and second-order Taylor expansion:
        δf ≈ Σ (∂f/∂p_i) δp_i + ½ Σ (∂²f/∂p_i∂p_j) δp_i δp_j
        
        Args:
            params: Nominal parameter values
            
        Returns:
            Uncertainty propagation results
        """
        if not self.config.uncertainty_propagation:
            return {}
        
        print(f"Propagating parameter uncertainties...")
        
        # Compute gradient and Hessian
        gradient = self.compute_gradient(params)
        if self.config.compute_hessian:
            hessian = self.compute_hessian(params)
        else:
            hessian = None
        
        # Parameter uncertainties
        param_uncertainties = np.array([
            self.config.parameter_uncertainties.get(name, 0.0) 
            for name in self.parameter_names
        ])
        
        # First-order uncertainty (linear propagation)
        gradient_array = np.array([gradient[name] for name in self.parameter_names])
        linear_variance = np.sum((gradient_array * param_uncertainties)**2)
        linear_std = np.sqrt(linear_variance)
        
        # Second-order correction if Hessian available
        if hessian is not None:
            # Quadratic contribution to variance
            quadratic_variance = 0.0
            for i in range(len(self.parameter_names)):
                for j in range(len(self.parameter_names)):
                    quadratic_variance += (0.5 * hessian[i, j] * 
                                         param_uncertainties[i] * param_uncertainties[j])**2
            
            quadratic_std = np.sqrt(quadratic_variance)
            total_std = np.sqrt(linear_variance + quadratic_variance)
        else:
            quadratic_std = 0.0
            total_std = linear_std
        
        # Monte Carlo validation
        mc_std = self._monte_carlo_uncertainty(params) if self.config.monte_carlo_samples > 0 else None
        
        results = {
            'linear_std': linear_std,
            'quadratic_std': quadratic_std,
            'total_std': total_std,
            'monte_carlo_std': mc_std,
            'parameter_contributions': {
                name: abs(gradient[name] * self.config.parameter_uncertainties.get(name, 0.0))
                for name in self.parameter_names
            }
        }
        
        print(f"  Linear uncertainty: ±{linear_std:.2e}")
        if quadratic_std > 0:
            print(f"  Quadratic correction: ±{quadratic_std:.2e}")
            print(f"  Total uncertainty: ±{total_std:.2e}")
        if mc_std is not None:
            print(f"  Monte Carlo validation: ±{mc_std:.2e}")
        
        return results
    
    def _monte_carlo_uncertainty(self, params: Dict[str, float]) -> float:
        """Monte Carlo uncertainty estimation."""
        samples = []
        
        for _ in range(self.config.monte_carlo_samples):
            # Sample parameters from uncertainty distributions
            sample_params = {}
            for name in self.parameter_names:
                uncertainty = self.config.parameter_uncertainties.get(name, 0.0)
                sample_params[name] = np.random.normal(params[name], uncertainty)
            
            # Evaluate function
            try:
                sample_value = self.objective_function(sample_params)
                samples.append(sample_value)
            except:
                continue  # Skip failed evaluations
        
        if len(samples) > 10:
            return np.std(samples)
        else:
            return None
    
    def gradient_based_optimization(self, initial_params: Dict[str, float]) -> Dict:
        """
        Perform gradient-based optimization using exact derivatives.
        
        Args:
            initial_params: Starting parameter values
            
        Returns:
            Optimization results
        """
        print(f"Starting gradient-based optimization...")
        
        if self.config.use_jax and JAX_AVAILABLE:
            # Use JAX optimization
            return self._jax_optimization(initial_params)
        else:
            # Use scipy optimization with gradient
            return self._scipy_optimization(initial_params)
    
    def _jax_optimization(self, initial_params: Dict[str, float]) -> Dict:
        """JAX-based optimization."""
        # Convert to array format
        x0 = self._params_dict_to_array(initial_params)
        
        # Define bounds
        bounds = []
        for name in self.parameter_names:
            if name in self.config.parameter_bounds:
                bounds.append(self.config.parameter_bounds[name])
            else:
                bounds.append((None, None))
        
        try:
            # JAX optimization (placeholder - full implementation would use JAX optimizers)
            from scipy.optimize import minimize
            
            def objective_array(x):
                params = self._params_array_to_dict(x)
                return self.objective_function(params)
            
            def gradient_array(x):
                if JAX_AVAILABLE:
                    return np.array(self.gradient_func(x))
                else:
                    params = self._params_array_to_dict(x)
                    grad_dict = self.gradient_func(params)
                    return np.array([grad_dict[name] for name in self.parameter_names])
            
            result = minimize(
                objective_array,
                x0,
                method=self.config.optimization_method,
                jac=gradient_array,
                bounds=bounds if any(b != (None, None) for b in bounds) else None,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            optimal_params = self._params_array_to_dict(result.x)
            
            return {
                'success': result.success,
                'optimal_parameters': optimal_params,
                'optimal_value': result.fun,
                'n_iterations': result.nit,
                'final_gradient': self.compute_gradient(optimal_params),
                'optimization_message': result.message
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'initial_parameters': initial_params
            }
    
    def _scipy_optimization(self, initial_params: Dict[str, float]) -> Dict:
        """Scipy-based optimization with gradient."""
        x0 = self._params_dict_to_array(initial_params)
        
        def objective_array(x):
            params = self._params_array_to_dict(x)
            return self.objective_function(params)
        
        def gradient_array(x):
            params = self._params_array_to_dict(x)
            grad_dict = self.gradient_func(params)
            return np.array([grad_dict[name] for name in self.parameter_names])
        
        # Set up bounds
        bounds = []
        for name in self.parameter_names:
            if name in self.config.parameter_bounds:
                bounds.append(self.config.parameter_bounds[name])
            else:
                bounds.append((None, None))
        
        try:
            result = minimize(
                objective_array,
                x0,
                method=self.config.optimization_method,
                jac=gradient_array,
                bounds=bounds if any(b != (None, None) for b in bounds) else None,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            optimal_params = self._params_array_to_dict(result.x)
            
            return {
                'success': result.success,
                'optimal_parameters': optimal_params,
                'optimal_value': result.fun,
                'n_iterations': result.nit,
                'final_gradient': self.compute_gradient(optimal_params),
                'optimization_message': result.message
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'initial_parameters': initial_params
            }

# Utility functions
def create_test_objective(complexity: str = 'quadratic') -> Callable:
    """
    Create test objective function for demonstration.
    
    Args:
        complexity: Type of objective ('quadratic', 'rosenbrock', 'polynomial')
        
    Returns:
        Test objective function
    """
    if complexity == 'quadratic':
        def objective(params):
            x = params.get('x', 0.0)
            y = params.get('y', 0.0)
            return x**2 + 2*y**2 + 0.1*x*y + 0.01*x**3
    
    elif complexity == 'rosenbrock':
        def objective(params):
            x = params.get('x', 0.0)
            y = params.get('y', 0.0)
            return 100*(y - x**2)**2 + (1 - x)**2
    
    elif complexity == 'polynomial':
        def objective(params):
            x = params.get('x', 0.0)
            y = params.get('y', 0.0)
            z = params.get('z', 0.0)
            return x**4 + y**4 + z**4 + x*y*z + 0.1*(x**2 + y**2 + z**2)
    
    else:
        raise ValueError(f"Unknown complexity: {complexity}")
    
    return objective

if __name__ == "__main__":
    # Demonstration of sensitivity-gradient analysis
    print("Sensitivity-Gradient Analysis Demonstration")
    print("=" * 45)
    
    # Create test objective
    objective = create_test_objective('rosenbrock')
    
    # Configuration
    config = SensitivityConfig(
        method=SensitivityMethod.AUTOMATIC_DIFF if JAX_AVAILABLE else SensitivityMethod.FINITE_DIFFERENCE,
        parameter_bounds={'x': (-2.0, 2.0), 'y': (-2.0, 2.0)},
        parameter_uncertainties={'x': 0.1, 'y': 0.1},
        nominal_values={'x': 0.5, 'y': 0.5},
        compute_hessian=True,
        uncertainty_propagation=True,
        monte_carlo_samples=5000
    )
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(config, objective)
    
    # Test point for analysis
    test_params = {'x': 0.5, 'y': 0.5}
    
    print(f"\nAnalyzing sensitivity at test point: {test_params}")
    print(f"Objective value: {objective(test_params):.6f}")
    
    # Sensitivity analysis
    sensitivity = analyzer.sensitivity_analysis(test_params)
    
    print(f"\nSensitivity Analysis Results:")
    print(f"  Gradient magnitude: {sensitivity['gradient_magnitude']:.2e}")
    print(f"  Parameter gradients:")
    for param, grad in sensitivity['gradient'].items():
        print(f"    ∂f/∂{param}: {grad:.4f}")
    
    print(f"  Sensitivity ranking: {sensitivity['sensitivity_ranking']}")
    
    if sensitivity['hessian'] is not None:
        print(f"  Hessian condition number: {sensitivity['hessian_condition_number']:.2e}")
        print(f"  Hessian eigenvalues: {sensitivity['hessian_eigenvalues']}")
    
    # Uncertainty propagation
    uncertainty = analyzer.uncertainty_propagation(test_params)
    
    if uncertainty:
        print(f"\nUncertainty Propagation:")
        print(f"  Linear uncertainty: ±{uncertainty['linear_std']:.4f}")
        if uncertainty['quadratic_std'] > 0:
            print(f"  Quadratic correction: ±{uncertainty['quadratic_std']:.4f}")
            print(f"  Total uncertainty: ±{uncertainty['total_std']:.4f}")
        if uncertainty['monte_carlo_std'] is not None:
            print(f"  Monte Carlo validation: ±{uncertainty['monte_carlo_std']:.4f}")
        
        print(f"  Parameter contributions:")
        for param, contrib in uncertainty['parameter_contributions'].items():
            print(f"    {param}: ±{contrib:.4f}")
    
    # Gradient-based optimization
    print(f"\nPerforming gradient-based optimization...")
    
    initial_guess = {'x': -1.0, 'y': 1.0}  # Start away from minimum
    optimization = analyzer.gradient_based_optimization(initial_guess)
    
    print(f"Optimization Results:")
    if optimization['success']:
        print(f"  Success: {optimization['success']}")
        print(f"  Optimal value: {optimization['optimal_value']:.6f}")
        print(f"  Iterations: {optimization['n_iterations']}")
        print(f"  Optimal parameters:")
        for param, value in optimization['optimal_parameters'].items():
            print(f"    {param}: {value:.6f}")
        
        final_grad_norm = np.linalg.norm(list(optimization['final_gradient'].values()))
        print(f"  Final gradient norm: {final_grad_norm:.2e}")
    else:
        print(f"  Optimization failed: {optimization.get('error', 'Unknown error')}")
    
    print("\n✅ Sensitivity-gradient analysis demonstration complete!")
