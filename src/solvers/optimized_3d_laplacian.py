"""
Optimized 3D Laplacian Operator for Enhanced Stargate Transporter

This module implements highly optimized 3D Laplacian operators using JAX
with vectorization, memory optimization, and multiple discretization schemes.

Mathematical Framework:
    ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
    
Supports finite differences, spectral methods, and compact stencils.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import time
from functools import partial

# Import our enhanced transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter

class Optimized3DLaplacianOperator:
    """
    Highly optimized 3D Laplacian operator with multiple discretization schemes.
    
    Provides maximum performance for large-scale field computations
    with vectorized operations and memory-efficient implementations.
    """
    
    def __init__(self, grid_config: Dict, operator_config: Optional[Dict] = None):
        """
        Initialize optimized 3D Laplacian operator.
        
        Args:
            grid_config: Grid configuration {nx, ny, nz, dx, dy, dz}
            operator_config: Operator-specific configuration
        """
        self.grid_config = grid_config
        
        # Default operator configuration
        if operator_config is None:
            operator_config = {
                'discretization': 'finite_difference',  # 'finite_difference', 'spectral', 'compact'
                'stencil_order': 2,                     # 2, 4, 6, 8 for finite differences
                'boundary_condition': 'periodic',       # 'periodic', 'dirichlet', 'neumann'
                'vectorization': 'full',                # 'full', 'partial', 'none'
                'memory_optimization': True,            # Enable memory optimizations
                'parallel_mode': 'auto'                 # 'auto', 'pmap', 'vmap', 'none'
            }
        self.operator_config = operator_config
        
        # Extract grid parameters
        self.nx, self.ny, self.nz = grid_config['nx'], grid_config['ny'], grid_config['nz']
        self.dx, self.dy, self.dz = grid_config['dx'], grid_config['dy'], grid_config['dz']
        
        # Precompute stencil coefficients
        self.stencil_coeffs = self._compute_stencil_coefficients()
        
        # Setup optimized operators
        self._setup_optimized_operators()
        
        # Performance tracking
        self.timing_data = {}
        self.memory_usage = {}
        
        print(f"Optimized3DLaplacianOperator initialized:")
        print(f"  Grid: {self.nx}×{self.ny}×{self.nz} = {self.nx*self.ny*self.nz:,} points")
        print(f"  Discretization: {operator_config['discretization']}")
        print(f"  Stencil order: {operator_config['stencil_order']}")
        print(f"  Boundary condition: {operator_config['boundary_condition']}")
        print(f"  Vectorization: {operator_config['vectorization']}")
        
    def _compute_stencil_coefficients(self) -> Dict:
        """Compute finite difference stencil coefficients."""
        
        order = self.operator_config['stencil_order']
        
        if order == 2:
            # Standard 3-point stencil: [1, -2, 1]
            coeffs_x = jnp.array([1.0, -2.0, 1.0]) / self.dx**2
            coeffs_y = jnp.array([1.0, -2.0, 1.0]) / self.dy**2
            coeffs_z = jnp.array([1.0, -2.0, 1.0]) / self.dz**2
            
        elif order == 4:
            # 5-point stencil: [-1/12, 4/3, -5/2, 4/3, -1/12]
            coeffs_x = jnp.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / self.dx**2
            coeffs_y = jnp.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / self.dy**2
            coeffs_z = jnp.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / self.dz**2
            
        elif order == 6:
            # 7-point stencil
            coeffs_x = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]) / self.dx**2
            coeffs_y = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]) / self.dy**2
            coeffs_z = jnp.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]) / self.dz**2
            
        elif order == 8:
            # 9-point stencil
            coeffs_x = jnp.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]) / self.dx**2
            coeffs_y = jnp.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]) / self.dy**2
            coeffs_z = jnp.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]) / self.dz**2
            
        else:
            raise ValueError(f"Unsupported stencil order: {order}")
            
        return {
            'x': coeffs_x,
            'y': coeffs_y,
            'z': coeffs_z,
            'order': order
        }
    
    def _setup_optimized_operators(self):
        """Setup optimized Laplacian operators based on configuration."""
        
        discretization = self.operator_config['discretization']
        
        if discretization == 'finite_difference':
            self._setup_finite_difference_operators()
        elif discretization == 'spectral':
            self._setup_spectral_operators()
        elif discretization == 'compact':
            self._setup_compact_operators()
        else:
            raise ValueError(f"Unknown discretization: {discretization}")
            
        print(f"  ✅ Optimized operators compiled")
    
    def _setup_finite_difference_operators(self):
        """Setup finite difference Laplacian operators."""
        
        order = self.stencil_coeffs['order']
        boundary = self.operator_config['boundary_condition']
        
        if order == 2:
            if boundary == 'periodic':
                @jit
                def laplacian_2nd_periodic(field):
                    return self._laplacian_2nd_periodic_impl(field)
            else:
                @jit
                def laplacian_2nd_dirichlet(field):
                    return self._laplacian_2nd_dirichlet_impl(field)
                laplacian_2nd_periodic = laplacian_2nd_dirichlet
                
        elif order == 4:
            @jit
            def laplacian_4th_order(field):
                return self._laplacian_4th_order_impl(field)
            laplacian_2nd_periodic = laplacian_4th_order
            
        elif order == 6:
            @jit
            def laplacian_6th_order(field):
                return self._laplacian_6th_order_impl(field)
            laplacian_2nd_periodic = laplacian_6th_order
            
        elif order == 8:
            @jit  
            def laplacian_8th_order(field):
                return self._laplacian_8th_order_impl(field)
            laplacian_2nd_periodic = laplacian_8th_order
        
        # Vectorized versions
        if self.operator_config['vectorization'] == 'full':
            @jit
            def vectorized_laplacian(fields):
                """Apply Laplacian to multiple fields simultaneously."""
                return vmap(laplacian_2nd_periodic)(fields)
                
            self.apply_laplacian = laplacian_2nd_periodic
            self.apply_laplacian_vectorized = vectorized_laplacian
            
        else:
            self.apply_laplacian = laplacian_2nd_periodic
            
        # Parallel versions if requested
        if self.operator_config['parallel_mode'] == 'pmap':
            @pmap
            def parallel_laplacian(fields):
                return vmap(laplacian_2nd_periodic)(fields)
            self.apply_laplacian_parallel = parallel_laplacian
    
    @jit
    def _laplacian_2nd_periodic_impl(self, field: jnp.ndarray) -> jnp.ndarray:
        """Optimized 2nd-order finite difference Laplacian with periodic BC."""
        
        # Precompute inverse grid spacings squared
        dx2_inv = 1.0 / self.dx**2
        dy2_inv = 1.0 / self.dy**2  
        dz2_inv = 1.0 / self.dz**2
        
        # Second derivatives using JAX roll (periodic boundaries)
        d2_dx2 = (jnp.roll(field, -1, axis=0) - 2*field + jnp.roll(field, 1, axis=0)) * dx2_inv
        d2_dy2 = (jnp.roll(field, -1, axis=1) - 2*field + jnp.roll(field, 1, axis=1)) * dy2_inv
        d2_dz2 = (jnp.roll(field, -1, axis=2) - 2*field + jnp.roll(field, 1, axis=2)) * dz2_inv
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    @jit
    def _laplacian_2nd_dirichlet_impl(self, field: jnp.ndarray) -> jnp.ndarray:
        """Optimized 2nd-order finite difference Laplacian with Dirichlet BC."""
        
        dx2_inv = 1.0 / self.dx**2
        dy2_inv = 1.0 / self.dy**2
        dz2_inv = 1.0 / self.dz**2
        
        # Initialize Laplacian
        laplacian = jnp.zeros_like(field)
        
        # Interior points only (boundaries handled separately)
        interior = field[1:-1, 1:-1, 1:-1]
        
        # X direction
        d2_dx2_interior = (field[2:, 1:-1, 1:-1] - 2*interior + field[:-2, 1:-1, 1:-1]) * dx2_inv
        
        # Y direction  
        d2_dy2_interior = (field[1:-1, 2:, 1:-1] - 2*interior + field[1:-1, :-2, 1:-1]) * dy2_inv
        
        # Z direction
        d2_dz2_interior = (field[1:-1, 1:-1, 2:] - 2*interior + field[1:-1, 1:-1, :-2]) * dz2_inv
        
        # Set interior values
        laplacian = laplacian.at[1:-1, 1:-1, 1:-1].set(d2_dx2_interior + d2_dy2_interior + d2_dz2_interior)
        
        return laplacian
    
    @jit
    def _laplacian_4th_order_impl(self, field: jnp.ndarray) -> jnp.ndarray:
        """Optimized 4th-order finite difference Laplacian."""
        
        # 4th-order stencil coefficients
        c = self.stencil_coeffs
        
        # Apply stencil in each direction
        d2_dx2 = (c['x'][0] * jnp.roll(field, -2, axis=0) +
                  c['x'][1] * jnp.roll(field, -1, axis=0) +
                  c['x'][2] * field +
                  c['x'][3] * jnp.roll(field, 1, axis=0) +
                  c['x'][4] * jnp.roll(field, 2, axis=0))
        
        d2_dy2 = (c['y'][0] * jnp.roll(field, -2, axis=1) +
                  c['y'][1] * jnp.roll(field, -1, axis=1) +
                  c['y'][2] * field +
                  c['y'][3] * jnp.roll(field, 1, axis=1) +
                  c['y'][4] * jnp.roll(field, 2, axis=1))
        
        d2_dz2 = (c['z'][0] * jnp.roll(field, -2, axis=2) +
                  c['z'][1] * jnp.roll(field, -1, axis=2) +
                  c['z'][2] * field +
                  c['z'][3] * jnp.roll(field, 1, axis=2) +
                  c['z'][4] * jnp.roll(field, 2, axis=2))
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    @jit
    def _laplacian_6th_order_impl(self, field: jnp.ndarray) -> jnp.ndarray:
        """Optimized 6th-order finite difference Laplacian."""
        
        c = self.stencil_coeffs
        
        # 6th-order stencil
        d2_dx2 = jnp.zeros_like(field)
        for i, coeff in enumerate(c['x']):
            offset = i - len(c['x']) // 2
            d2_dx2 = d2_dx2 + coeff * jnp.roll(field, offset, axis=0)
            
        d2_dy2 = jnp.zeros_like(field)
        for i, coeff in enumerate(c['y']):
            offset = i - len(c['y']) // 2
            d2_dy2 = d2_dy2 + coeff * jnp.roll(field, offset, axis=1)
            
        d2_dz2 = jnp.zeros_like(field)
        for i, coeff in enumerate(c['z']):
            offset = i - len(c['z']) // 2
            d2_dz2 = d2_dz2 + coeff * jnp.roll(field, offset, axis=2)
            
        return d2_dx2 + d2_dy2 + d2_dz2
    
    @jit
    def _laplacian_8th_order_impl(self, field: jnp.ndarray) -> jnp.ndarray:
        """Optimized 8th-order finite difference Laplacian."""
        
        c = self.stencil_coeffs
        
        # Unrolled 8th-order stencil for maximum performance
        d2_dx2 = (c['x'][0] * jnp.roll(field, -4, axis=0) +
                  c['x'][1] * jnp.roll(field, -3, axis=0) +
                  c['x'][2] * jnp.roll(field, -2, axis=0) +
                  c['x'][3] * jnp.roll(field, -1, axis=0) +
                  c['x'][4] * field +
                  c['x'][5] * jnp.roll(field, 1, axis=0) +
                  c['x'][6] * jnp.roll(field, 2, axis=0) +
                  c['x'][7] * jnp.roll(field, 3, axis=0) +
                  c['x'][8] * jnp.roll(field, 4, axis=0))
        
        d2_dy2 = (c['y'][0] * jnp.roll(field, -4, axis=1) +
                  c['y'][1] * jnp.roll(field, -3, axis=1) +
                  c['y'][2] * jnp.roll(field, -2, axis=1) +
                  c['y'][3] * jnp.roll(field, -1, axis=1) +
                  c['y'][4] * field +
                  c['y'][5] * jnp.roll(field, 1, axis=1) +
                  c['y'][6] * jnp.roll(field, 2, axis=1) +
                  c['y'][7] * jnp.roll(field, 3, axis=1) +
                  c['y'][8] * jnp.roll(field, 4, axis=1))
        
        d2_dz2 = (c['z'][0] * jnp.roll(field, -4, axis=2) +
                  c['z'][1] * jnp.roll(field, -3, axis=2) +
                  c['z'][2] * jnp.roll(field, -2, axis=2) +
                  c['z'][3] * jnp.roll(field, -1, axis=2) +
                  c['z'][4] * field +
                  c['z'][5] * jnp.roll(field, 1, axis=2) +
                  c['z'][6] * jnp.roll(field, 2, axis=2) +
                  c['z'][7] * jnp.roll(field, 3, axis=2) +
                  c['z'][8] * jnp.roll(field, 4, axis=2))
        
        return d2_dx2 + d2_dy2 + d2_dz2
    
    def _setup_spectral_operators(self):
        """Setup spectral Laplacian operators using FFT."""
        
        # Precompute wavenumber grids
        kx = jnp.fft.fftfreq(self.nx, self.dx) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(self.ny, self.dy) * 2 * jnp.pi
        kz = jnp.fft.fftfreq(self.nz, self.dz) * 2 * jnp.pi
        
        # Create 3D wavenumber arrays
        Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
        
        # Laplacian in Fourier space: -k²
        self.k_squared = -(Kx**2 + Ky**2 + Kz**2)
        
        @jit
        def spectral_laplacian(field):
            """Spectral Laplacian using FFT."""
            # Forward FFT
            field_hat = jnp.fft.fftn(field)
            
            # Apply Laplacian in Fourier space
            laplacian_hat = self.k_squared * field_hat
            
            # Inverse FFT
            laplacian = jnp.real(jnp.fft.ifftn(laplacian_hat))
            
            return laplacian
            
        self.apply_laplacian = spectral_laplacian
        
        # Vectorized spectral version
        @jit
        def vectorized_spectral_laplacian(fields):
            return vmap(spectral_laplacian)(fields)
            
        self.apply_laplacian_vectorized = vectorized_spectral_laplacian
    
    def _setup_compact_operators(self):
        """Setup compact finite difference operators."""
        
        # Compact scheme: α·φ_{i-1} + φ_i + α·φ_{i+1} = β/h²(φ_{i+1} - 2φ_i + φ_{i-1})
        # For 4th-order accuracy: α = 1/10, β = 6/5
        
        alpha = 1.0 / 10.0
        beta = 6.0 / 5.0
        
        dx2_inv = 1.0 / self.dx**2
        dy2_inv = 1.0 / self.dy**2
        dz2_inv = 1.0 / self.dz**2
        
        @jit
        def compact_laplacian(field):
            """Compact finite difference Laplacian."""
            
            # This is a simplified implementation
            # Full compact schemes require solving tridiagonal systems
            
            # Approximate compact stencil
            d2_dx2 = beta * dx2_inv * (jnp.roll(field, -1, axis=0) - 2*field + jnp.roll(field, 1, axis=0))
            d2_dy2 = beta * dy2_inv * (jnp.roll(field, -1, axis=1) - 2*field + jnp.roll(field, 1, axis=1))
            d2_dz2 = beta * dz2_inv * (jnp.roll(field, -1, axis=2) - 2*field + jnp.roll(field, 1, axis=2))
            
            # Apply compact operator (simplified)
            laplacian = d2_dx2 + d2_dy2 + d2_dz2
            
            # Correction for compact scheme (approximate)
            correction = alpha * (jnp.roll(laplacian, -1, axis=0) + jnp.roll(laplacian, 1, axis=0) +
                                jnp.roll(laplacian, -1, axis=1) + jnp.roll(laplacian, 1, axis=1) +
                                jnp.roll(laplacian, -1, axis=2) + jnp.roll(laplacian, 1, axis=2))
            
            return laplacian + correction
        
        self.apply_laplacian = compact_laplacian
    
    def benchmark_performance(self, field_shapes: List[Tuple], num_trials: int = 10) -> Dict:
        """
        Benchmark Laplacian operator performance.
        
        Args:
            field_shapes: List of field shapes to test
            num_trials: Number of trials for timing
            
        Returns:
            Performance benchmark results
        """
        print(f"\n⚡ LAPLACIAN OPERATOR BENCHMARKS")
        print("-" * 50)
        
        results = {}
        
        for shape in field_shapes:
            print(f"Testing shape {shape}...")
            
            # Generate test field
            key = jax.random.PRNGKey(42)
            test_field = jax.random.normal(key, shape)
            
            # Benchmark single application
            start_time = time.time()
            for _ in range(num_trials):
                result = self.apply_laplacian(test_field)
                result.block_until_ready()  # Ensure computation completes
            single_time = (time.time() - start_time) / num_trials
            
            # Benchmark vectorized application if available
            if hasattr(self, 'apply_laplacian_vectorized') and len(shape) == 3:
                # Create batch of fields
                batch_size = 4
                test_fields = jnp.stack([test_field] * batch_size)
                
                start_time = time.time()
                for _ in range(num_trials):
                    result_vec = self.apply_laplacian_vectorized(test_fields)
                    result_vec.block_until_ready()
                vectorized_time = (time.time() - start_time) / num_trials
                
                speedup = single_time * batch_size / vectorized_time
            else:
                vectorized_time = None
                speedup = None
            
            # Calculate throughput
            elements = np.prod(shape)
            throughput = elements / single_time / 1e6  # Million elements per second
            
            # Memory usage estimate
            memory_usage = elements * 8 * 4 / (1024**3)  # GB (input + output + temporaries)
            
            results[shape] = {
                'single_time': single_time,
                'vectorized_time': vectorized_time,
                'speedup': speedup,
                'throughput_meps': throughput,  # Million elements per second
                'memory_usage_gb': memory_usage,
                'elements': elements
            }
            
            print(f"  Single: {single_time*1000:.2f} ms")
            if vectorized_time:
                print(f"  Vectorized: {vectorized_time*1000:.2f} ms ({speedup:.1f}× speedup)")
            print(f"  Throughput: {throughput:.1f} MEPS")
            print(f"  Memory: {memory_usage:.3f} GB")
            
        return results
    
    def accuracy_analysis(self, analytical_function: Callable) -> Dict:
        """
        Analyze accuracy of Laplacian operator against analytical solution.
        
        Args:
            analytical_function: Function providing analytical Laplacian
            
        Returns:
            Accuracy analysis results
        """
        print(f"\n🎯 LAPLACIAN ACCURACY ANALYSIS")
        print("-" * 50)
        
        # Create coordinate arrays
        x = jnp.linspace(-2*jnp.pi, 2*jnp.pi, self.nx)
        y = jnp.linspace(-2*jnp.pi, 2*jnp.pi, self.ny)
        z = jnp.linspace(-2*jnp.pi, 2*jnp.pi, self.nz)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Test function: u(x,y,z) = sin(x)cos(y)sin(z)
        test_function = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
        
        # Analytical Laplacian: ∇²u = -3·sin(x)cos(y)sin(z)
        analytical_laplacian = -3 * test_function
        
        # Numerical Laplacian
        numerical_laplacian = self.apply_laplacian(test_function)
        
        # Error analysis
        absolute_error = jnp.abs(numerical_laplacian - analytical_laplacian)
        relative_error = absolute_error / (jnp.abs(analytical_laplacian) + 1e-12)
        
        # Error norms
        l1_error = jnp.mean(absolute_error)
        l2_error = jnp.sqrt(jnp.mean(absolute_error**2))
        linf_error = jnp.max(absolute_error)
        
        # Relative error norms
        rel_l1_error = jnp.mean(relative_error)
        rel_l2_error = jnp.sqrt(jnp.mean(relative_error**2))
        rel_linf_error = jnp.max(relative_error)
        
        print(f"Discretization: {self.operator_config['discretization']}")
        print(f"Stencil order: {self.operator_config['stencil_order']}")
        print(f"Grid resolution: {self.nx}×{self.ny}×{self.nz}")
        
        print(f"\nAbsolute errors:")
        print(f"  L1 error: {l1_error:.2e}")
        print(f"  L2 error: {l2_error:.2e}")
        print(f"  L∞ error: {linf_error:.2e}")
        
        print(f"\nRelative errors:")
        print(f"  L1 error: {rel_l1_error:.2%}")
        print(f"  L2 error: {rel_l2_error:.2%}")
        print(f"  L∞ error: {rel_linf_error:.2%}")
        
        # Order of accuracy estimate
        if self.operator_config['discretization'] == 'finite_difference':
            theoretical_order = self.operator_config['stencil_order']
            h = min(self.dx, self.dy, self.dz)
            expected_error = h**theoretical_order
            observed_order = -jnp.log(l2_error) / jnp.log(h)
            
            print(f"\nOrder of accuracy:")
            print(f"  Theoretical: {theoretical_order}")
            print(f"  Observed: {float(observed_order):.2f}")
            print(f"  Expected error: {expected_error:.2e}")
        
        return {
            'absolute_errors': {
                'l1': float(l1_error),
                'l2': float(l2_error),
                'linf': float(linf_error)
            },
            'relative_errors': {
                'l1': float(rel_l1_error),
                'l2': float(rel_l2_error),
                'linf': float(rel_linf_error)
            },
            'error_field': absolute_error,
            'numerical_solution': numerical_laplacian,
            'analytical_solution': analytical_laplacian
        }

def main():
    """Demonstration of optimized 3D Laplacian operator."""
    from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
    
    print("="*70)
    print("OPTIMIZED 3D LAPLACIAN OPERATOR DEMONSTRATION")
    print("="*70)
    
    # Grid configuration
    grid_config = {
        'nx': 64, 'ny': 64, 'nz': 64,
        'dx': 0.1, 'dy': 0.1, 'dz': 0.1
    }
    
    # Test different operator configurations
    configs = [
        {'discretization': 'finite_difference', 'stencil_order': 2, 'boundary_condition': 'periodic'},
        {'discretization': 'finite_difference', 'stencil_order': 4, 'boundary_condition': 'periodic'},
        {'discretization': 'finite_difference', 'stencil_order': 6, 'boundary_condition': 'periodic'},
        {'discretization': 'spectral', 'boundary_condition': 'periodic'},
        {'discretization': 'compact', 'boundary_condition': 'periodic'}
    ]
    
    # Performance comparison
    print(f"\n🏁 OPERATOR PERFORMANCE COMPARISON")
    print("-" * 70)
    
    performance_results = {}
    
    for i, operator_config in enumerate(configs):
        try:
            print(f"\n--- Configuration {i+1}: {operator_config['discretization']} ---")
            
            laplacian_op = Optimized3DLaplacianOperator(grid_config, operator_config)
            
            # Benchmark performance
            test_shapes = [(32, 32, 32), (64, 64, 64)]
            benchmark = laplacian_op.benchmark_performance(test_shapes, num_trials=5)
            
            # Accuracy analysis
            accuracy = laplacian_op.accuracy_analysis(lambda x, y, z: -3 * jnp.sin(x) * jnp.cos(y) * jnp.sin(z))
            
            performance_results[f"config_{i+1}"] = {
                'config': operator_config,
                'benchmark': benchmark,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")
            continue
    
    # Summary comparison
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"{'Config':<15} {'Method':<12} {'Order':<6} {'Time (ms)':<12} {'Throughput':<12} {'L2 Error':<12}")
    print("-" * 70)
    
    for config_name, results in performance_results.items():
        if 'benchmark' in results and 'accuracy' in results:
            config = results['config']
            benchmark = results['benchmark']
            accuracy = results['accuracy']
            
            # Get results for 64³ grid
            grid_key = (64, 64, 64)
            if grid_key in benchmark:
                time_ms = benchmark[grid_key]['single_time'] * 1000
                throughput = benchmark[grid_key]['throughput_meps']
                l2_error = accuracy['absolute_errors']['l2']
                
                method = config['discretization'][:6]
                order = config.get('stencil_order', 'N/A')
                
                print(f"{config_name:<15} {method:<12} {str(order):<6} {time_ms:<12.2f} {throughput:<12.1f} {l2_error:<12.2e}")
    
    # Best configuration selection
    print(f"\n🏆 OPTIMAL CONFIGURATION SELECTION")
    print("-" * 50)
    
    best_performance = None
    best_accuracy = None
    best_balanced = None
    
    for config_name, results in performance_results.items():
        if 'benchmark' not in results or 'accuracy' not in results:
            continue
            
        grid_key = (64, 64, 64)
        if grid_key not in results['benchmark']:
            continue
            
        throughput = results['benchmark'][grid_key]['throughput_meps']
        l2_error = results['accuracy']['absolute_errors']['l2']
        
        # Best performance (highest throughput)
        if best_performance is None or throughput > best_performance[1]:
            best_performance = (config_name, throughput, results['config'])
            
        # Best accuracy (lowest error)
        if best_accuracy is None or l2_error < best_accuracy[1]:
            best_accuracy = (config_name, l2_error, results['config'])
            
        # Best balanced (throughput/error ratio)
        balance_metric = throughput / (l2_error + 1e-12)
        if best_balanced is None or balance_metric > best_balanced[1]:
            best_balanced = (config_name, balance_metric, results['config'])
    
    if best_performance:
        print(f"Best performance: {best_performance[0]} ({best_performance[1]:.1f} MEPS)")
        print(f"  Configuration: {best_performance[2]}")
        
    if best_accuracy:
        print(f"Best accuracy: {best_accuracy[0]} (L2 error: {best_accuracy[1]:.2e})")
        print(f"  Configuration: {best_accuracy[2]}")
        
    if best_balanced:
        print(f"Best balanced: {best_balanced[0]} (metric: {best_balanced[1]:.1e})")
        print(f"  Configuration: {best_balanced[2]}")
    
    # Target achievements
    target_performance = {
        'throughput_meps': 100.0,  # Million elements per second
        'l2_error': 1e-6,          # Absolute L2 error
        'memory_usage_gb': 2.0     # Memory usage limit
    }
    
    print(f"\n🎯 TARGET ACHIEVEMENT")
    print("-" * 50)
    
    achieved_configs = []
    for config_name, results in performance_results.items():
        if 'benchmark' not in results or 'accuracy' not in results:
            continue
            
        grid_key = (64, 64, 64)
        if grid_key not in results['benchmark']:
            continue
            
        throughput = results['benchmark'][grid_key]['throughput_meps']
        l2_error = results['accuracy']['absolute_errors']['l2']
        memory = results['benchmark'][grid_key]['memory_usage_gb']
        
        meets_throughput = throughput >= target_performance['throughput_meps']
        meets_accuracy = l2_error <= target_performance['l2_error']
        meets_memory = memory <= target_performance['memory_usage_gb']
        
        if meets_throughput and meets_accuracy and meets_memory:
            achieved_configs.append(config_name)
            
        status_throughput = "✅" if meets_throughput else "❌"
        status_accuracy = "✅" if meets_accuracy else "❌"
        status_memory = "✅" if meets_memory else "❌"
        
        print(f"{config_name}: Throughput {status_throughput} Accuracy {status_accuracy} Memory {status_memory}")
    
    if achieved_configs:
        print(f"\n🎉 Configurations meeting all targets: {', '.join(achieved_configs)}")
    else:
        print(f"\n⚠️ No configurations meet all performance targets")
    
    return performance_results

if __name__ == "__main__":
    results = main()
