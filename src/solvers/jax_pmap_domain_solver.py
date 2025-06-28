"""
JAX Pmap Domain Decomposition Solver for Enhanced Stargate Transporter

This module implements parallel domain decomposition using JAX pmap
for scalable Einstein field equation solving across multiple GPUs.

Mathematical Framework:
    ∇²φ = ρ(x,t) with domain decomposition Ω = ∪ᵢ Ωᵢ
    
Parallel solver with Schwarz alternating method and interface coupling.

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap, grad
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import time

# Import our enhanced transporter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_stargate_transporter import EnhancedStargateTransporter

class JAXPmapDomainDecompositionSolver:
    """
    Parallel domain decomposition solver using JAX pmap.
    
    Provides scalable solution of partial differential equations
    with multi-GPU parallelization and optimal load balancing.
    """
    
    def __init__(self, transporter: EnhancedStargateTransporter,
                 decomp_config: Optional[Dict] = None,
                 solver_config: Optional[Dict] = None):
        """
        Initialize parallel domain decomposition solver.
        
        Args:
            transporter: Enhanced stargate transporter instance
            decomp_config: Domain decomposition configuration
            solver_config: Solver-specific configuration
        """
        self.transporter = transporter
        
        # Default decomposition configuration
        if decomp_config is None:
            decomp_config = {
                'n_domains_x': 2,        # Domains in x direction
                'n_domains_y': 2,        # Domains in y direction  
                'n_domains_z': 2,        # Domains in z direction
                'overlap_points': 2,     # Overlap between domains
                'boundary_type': 'dirichlet',  # Boundary condition type
                'load_balance': True     # Enable dynamic load balancing
            }
        self.decomp_config = decomp_config
        
        # Default solver configuration
        if solver_config is None:
            solver_config = {
                'max_iterations': 1000,     # Maximum solver iterations
                'tolerance': 1e-8,          # Convergence tolerance
                'schwarz_iterations': 5,    # Schwarz method iterations
                'relaxation_parameter': 1.0, # Over/under-relaxation
                'preconditioner': 'jacobi', # Preconditioning method
                'krylov_subspace': 30       # Krylov subspace dimension
            }
        self.solver_config = solver_config
        
        # Grid and domain setup
        self.global_grid = self._setup_global_grid()
        self.domain_grids = self._decompose_domain()
        
        # Check available devices
        self.devices = jax.local_devices()
        self.n_devices = len(self.devices)
        self.total_domains = (decomp_config['n_domains_x'] * 
                            decomp_config['n_domains_y'] * 
                            decomp_config['n_domains_z'])
        
        print(f"JAXPmapDomainDecompositionSolver initialized:")
        print(f"  Available devices: {self.n_devices}")
        print(f"  Total domains: {self.total_domains}")
        print(f"  Domains per device: {self.total_domains / self.n_devices:.1f}")
        print(f"  Overlap points: {decomp_config['overlap_points']}")
        
        # Initialize parallel functions
        self._setup_parallel_functions()
        
        # Performance tracking
        self.iteration_history = []
        self.residual_history = []
        self.timing_data = {}
        
    def _setup_global_grid(self) -> Dict:
        """Setup global computational grid."""
        
        config = self.transporter.config
        
        # Grid resolution
        nx, ny, nz = 128, 128, 128  # High resolution for accuracy
        
        # Physical domain
        x_min, x_max = -config.R_payload * 2, config.R_payload * 2
        y_min, y_max = -config.R_payload * 2, config.R_payload * 2
        z_min, z_max = -config.L_corridor/2, config.L_corridor/2
        
        # Create coordinate arrays
        x = jnp.linspace(x_min, x_max, nx)
        y = jnp.linspace(y_min, y_max, ny)  
        z = jnp.linspace(z_min, z_max, nz)
        
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        
        # Create meshgrid
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        return {
            'x': x, 'y': y, 'z': z,
            'X': X, 'Y': Y, 'Z': Z,
            'dx': dx, 'dy': dy, 'dz': dz,
            'nx': nx, 'ny': ny, 'nz': nz,
            'domain_bounds': [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        }
    
    def _decompose_domain(self) -> List[Dict]:
        """Decompose global domain into subdomains."""
        
        nx_total, ny_total, nz_total = (self.global_grid['nx'], 
                                       self.global_grid['ny'],
                                       self.global_grid['nz'])
        
        nx_domains = self.decomp_config['n_domains_x']
        ny_domains = self.decomp_config['n_domains_y']
        nz_domains = self.decomp_config['n_domains_z']
        overlap = self.decomp_config['overlap_points']
        
        # Calculate domain sizes
        nx_per_domain = nx_total // nx_domains
        ny_per_domain = ny_total // ny_domains
        nz_per_domain = nz_total // nz_domains
        
        domain_grids = []
        
        for i in range(nx_domains):
            for j in range(ny_domains):
                for k in range(nz_domains):
                    # Calculate indices with overlap
                    ix_start = max(0, i * nx_per_domain - overlap)
                    ix_end = min(nx_total, (i + 1) * nx_per_domain + overlap)
                    
                    iy_start = max(0, j * ny_per_domain - overlap)
                    iy_end = min(ny_total, (j + 1) * ny_per_domain + overlap)
                    
                    iz_start = max(0, k * nz_per_domain - overlap)
                    iz_end = min(nz_total, (k + 1) * nz_per_domain + overlap)
                    
                    # Extract subdomain grid
                    domain_grid = {
                        'domain_id': (i, j, k),
                        'device_id': (i * ny_domains * nz_domains + j * nz_domains + k) % self.n_devices,
                        'ix_range': (ix_start, ix_end),
                        'iy_range': (iy_start, iy_end),
                        'iz_range': (iz_start, iz_end),
                        'x': self.global_grid['x'][ix_start:ix_end],
                        'y': self.global_grid['y'][iy_start:iy_end],
                        'z': self.global_grid['z'][iz_start:iz_end],
                        'dx': self.global_grid['dx'],
                        'dy': self.global_grid['dy'],
                        'dz': self.global_grid['dz'],
                        'is_boundary': {
                            'x_min': i == 0,
                            'x_max': i == nx_domains - 1,
                            'y_min': j == 0,
                            'y_max': j == ny_domains - 1,
                            'z_min': k == 0,
                            'z_max': k == nz_domains - 1
                        }
                    }
                    
                    domain_grids.append(domain_grid)
                    
        return domain_grids
    
    def _setup_parallel_functions(self):
        """Setup JAX pmap parallel functions."""
        
        # Parallel Laplacian operator
        @pmap
        def parallel_laplacian(field_domains):
            """Apply Laplacian operator to each domain."""
            return vmap(self._domain_laplacian)(field_domains)
        
        # Parallel residual computation
        @pmap
        def parallel_residual(field_domains, source_domains):
            """Compute residual for each domain."""
            return vmap(self._domain_residual)(field_domains, source_domains)
        
        # Parallel Schwarz iteration
        @pmap
        def parallel_schwarz_step(field_domains, source_domains, boundary_data):
            """Perform one Schwarz iteration step."""
            return vmap(self._schwarz_domain_step)(field_domains, source_domains, boundary_data)
        
        # Parallel boundary exchange
        @pmap
        def parallel_boundary_exchange(field_domains):
            """Exchange boundary data between domains."""
            return vmap(self._extract_boundary_data)(field_domains)
        
        self.parallel_laplacian = parallel_laplacian
        self.parallel_residual = parallel_residual
        self.parallel_schwarz_step = parallel_schwarz_step
        self.parallel_boundary_exchange = parallel_boundary_exchange
        
        print(f"  ✅ Parallel functions compiled for {self.n_devices} devices")
    
    @jit
    def _domain_laplacian(self, field: jnp.ndarray) -> jnp.ndarray:
        """Compute Laplacian on a single domain using finite differences."""
        
        # Get domain grid spacing
        dx = self.global_grid['dx']
        dy = self.global_grid['dy']
        dz = self.global_grid['dz']
        
        # Second derivatives using central differences
        d2_dx2 = (jnp.roll(field, -1, axis=0) - 2*field + jnp.roll(field, 1, axis=0)) / dx**2
        d2_dy2 = (jnp.roll(field, -1, axis=1) - 2*field + jnp.roll(field, 1, axis=1)) / dy**2
        d2_dz2 = (jnp.roll(field, -1, axis=2) - 2*field + jnp.roll(field, 1, axis=2)) / dz**2
        
        # Laplacian
        laplacian = d2_dx2 + d2_dy2 + d2_dz2
        
        return laplacian
    
    @jit
    def _domain_residual(self, field: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
        """Compute residual for Poisson equation: r = ∇²φ - ρ."""
        laplacian = self._domain_laplacian(field)
        residual = laplacian - source
        return residual
    
    @jit
    def _schwarz_domain_step(self, field: jnp.ndarray, source: jnp.ndarray, 
                           boundary_data: jnp.ndarray) -> jnp.ndarray:
        """Perform one Schwarz iteration step on a domain."""
        
        # Solve local problem with Dirichlet boundary conditions
        # Using Gauss-Seidel relaxation
        
        dx2 = self.global_grid['dx']**2
        dy2 = self.global_grid['dy']**2
        dz2 = self.global_grid['dz']**2
        
        # Relaxation coefficient
        omega = self.solver_config['relaxation_parameter']
        
        # Gauss-Seidel update
        updated_field = field.copy()
        
        for iteration in range(self.solver_config['schwarz_iterations']):
            # Interior points update
            for i in range(1, field.shape[0]-1):
                for j in range(1, field.shape[1]-1):
                    for k in range(1, field.shape[2]-1):
                        # Discrete Laplacian stencil
                        laplacian_neighbors = (
                            (updated_field[i+1,j,k] + updated_field[i-1,j,k]) / dx2 +
                            (updated_field[i,j+1,k] + updated_field[i,j-1,k]) / dy2 +
                            (updated_field[i,j,k+1] + updated_field[i,j,k-1]) / dz2
                        )
                        
                        denominator = 2 * (1/dx2 + 1/dy2 + 1/dz2)
                        new_value = (laplacian_neighbors - source[i,j,k]) / denominator
                        
                        # Relaxation
                        updated_field = updated_field.at[i,j,k].set(
                            (1 - omega) * updated_field[i,j,k] + omega * new_value
                        )
        
        return updated_field
    
    @jit
    def _extract_boundary_data(self, field: jnp.ndarray) -> jnp.ndarray:
        """Extract boundary data for domain coupling."""
        
        # Extract faces of the domain (simplified)
        boundary_faces = jnp.concatenate([
            field[0, :, :].flatten(),   # x_min face
            field[-1, :, :].flatten(),  # x_max face
            field[:, 0, :].flatten(),   # y_min face
            field[:, -1, :].flatten(),  # y_max face
            field[:, :, 0].flatten(),   # z_min face
            field[:, :, -1].flatten()   # z_max face
        ])
        
        return boundary_faces
    
    def solve_field_equation(self, source_function: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
                           t: float, initial_field: Optional[jnp.ndarray] = None) -> Dict:
        """
        Solve field equation using parallel domain decomposition.
        
        Args:
            source_function: Source term function ρ(x,y,z,t)
            t: Current time
            initial_field: Initial field guess
            
        Returns:
            Solution and performance data
        """
        start_time = time.time()
        
        print(f"\n🧮 PARALLEL FIELD EQUATION SOLVER")
        print("-" * 50)
        print(f"Time: {t:.3f}s")
        print(f"Domains: {self.total_domains}")
        print(f"Devices: {self.n_devices}")
        
        # Initialize field
        if initial_field is None:
            field = jnp.zeros((self.global_grid['nx'], 
                             self.global_grid['ny'], 
                             self.global_grid['nz']))
        else:
            field = initial_field
            
        # Compute source term
        source = source_function(self.global_grid['X'], 
                               self.global_grid['Y'], 
                               self.global_grid['Z'], t)
        
        # Decompose field and source into domains
        field_domains = self._distribute_to_domains(field)
        source_domains = self._distribute_to_domains(source)
        
        # Reshape for pmap (group domains by device)
        field_domains_pmap = self._reshape_for_pmap(field_domains)
        source_domains_pmap = self._reshape_for_pmap(source_domains)
        
        setup_time = time.time() - start_time
        
        # Schwarz alternating method
        max_iterations = self.solver_config['max_iterations']
        tolerance = self.solver_config['tolerance']
        
        residual_norm = float('inf')
        iteration = 0
        
        print(f"Starting iterations...")
        iteration_start = time.time()
        
        while iteration < max_iterations and residual_norm > tolerance:
            # Exchange boundary data
            boundary_data_pmap = self.parallel_boundary_exchange(field_domains_pmap)
            
            # Perform Schwarz iteration
            field_domains_pmap = self.parallel_schwarz_step(
                field_domains_pmap, source_domains_pmap, boundary_data_pmap
            )
            
            # Compute global residual
            residuals_pmap = self.parallel_residual(field_domains_pmap, source_domains_pmap)
            residual_norm = float(jnp.sqrt(jnp.sum(residuals_pmap**2)))
            
            # Store iteration data
            self.iteration_history.append(iteration)
            self.residual_history.append(residual_norm)
            
            if iteration % 50 == 0:
                print(f"  Iteration {iteration:4d}: residual = {residual_norm:.2e}")
                
            iteration += 1
        
        iteration_time = time.time() - iteration_start
        
        # Assemble global solution
        assembly_start = time.time()
        field_domains_final = self._reshape_from_pmap(field_domains_pmap)
        global_field = self._assemble_from_domains(field_domains_final)
        assembly_time = time.time() - assembly_start
        
        total_time = time.time() - start_time
        
        # Performance analysis
        convergence_achieved = residual_norm <= tolerance
        iterations_per_second = iteration / iteration_time if iteration_time > 0 else 0
        
        print(f"\n📊 SOLUTION COMPLETE")
        print("-" * 50)
        print(f"Converged: {'✅' if convergence_achieved else '❌'}")
        print(f"Iterations: {iteration}")
        print(f"Final residual: {residual_norm:.2e}")
        print(f"Target tolerance: {tolerance:.2e}")
        print(f"Setup time: {setup_time:.3f}s")
        print(f"Iteration time: {iteration_time:.3f}s")
        print(f"Assembly time: {assembly_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Iterations/second: {iterations_per_second:.1f}")
        
        # Store timing data
        self.timing_data = {
            'setup_time': setup_time,
            'iteration_time': iteration_time,
            'assembly_time': assembly_time,
            'total_time': total_time,
            'iterations_per_second': iterations_per_second
        }
        
        return {
            'solution': global_field,
            'converged': convergence_achieved,
            'iterations': iteration,
            'final_residual': residual_norm,
            'timing': self.timing_data,
            'performance': {
                'parallel_efficiency': self._compute_parallel_efficiency(),
                'memory_usage': self._estimate_memory_usage(),
                'computational_intensity': iteration * self.total_domains / total_time
            }
        }
    
    def _distribute_to_domains(self, global_field: jnp.ndarray) -> List[jnp.ndarray]:
        """Distribute global field to domain grids."""
        
        domain_fields = []
        
        for domain_grid in self.domain_grids:
            ix_start, ix_end = domain_grid['ix_range']
            iy_start, iy_end = domain_grid['iy_range']
            iz_start, iz_end = domain_grid['iz_range']
            
            domain_field = global_field[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end]
            domain_fields.append(domain_field)
            
        return domain_fields
    
    def _assemble_from_domains(self, domain_fields: List[jnp.ndarray]) -> jnp.ndarray:
        """Assemble global field from domain solutions."""
        
        global_field = jnp.zeros((self.global_grid['nx'], 
                                 self.global_grid['ny'], 
                                 self.global_grid['nz']))
        
        overlap = self.decomp_config['overlap_points']
        
        for i, domain_field in enumerate(domain_fields):
            domain_grid = self.domain_grids[i]
            
            # Calculate non-overlapping region for this domain
            ix_start, ix_end = domain_grid['ix_range']
            iy_start, iy_end = domain_grid['iy_range']
            iz_start, iz_end = domain_grid['iz_range']
            
            # Adjust for overlap removal
            if not domain_grid['is_boundary']['x_min']:
                ix_start += overlap
            if not domain_grid['is_boundary']['x_max']:
                ix_end -= overlap
            if not domain_grid['is_boundary']['y_min']:
                iy_start += overlap
            if not domain_grid['is_boundary']['y_max']:
                iy_end -= overlap
            if not domain_grid['is_boundary']['z_min']:
                iz_start += overlap
            if not domain_grid['is_boundary']['z_max']:
                iz_end -= overlap
                
            # Calculate corresponding indices in domain field
            local_ix_start = ix_start - domain_grid['ix_range'][0]
            local_ix_end = local_ix_start + (ix_end - ix_start)
            local_iy_start = iy_start - domain_grid['iy_range'][0]
            local_iy_end = local_iy_start + (iy_end - iy_start)
            local_iz_start = iz_start - domain_grid['iz_range'][0]
            local_iz_end = local_iz_start + (iz_end - iz_start)
            
            # Copy non-overlapping region
            global_field = global_field.at[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end].set(
                domain_field[local_ix_start:local_ix_end, 
                           local_iy_start:local_iy_end,
                           local_iz_start:local_iz_end]
            )
            
        return global_field
    
    def _reshape_for_pmap(self, domain_data: List) -> jnp.ndarray:
        """Reshape domain data for pmap parallelization."""
        
        # Group domains by device
        domains_per_device = max(1, self.total_domains // self.n_devices)
        
        # Pad if necessary
        while len(domain_data) < self.n_devices * domains_per_device:
            domain_data.append(jnp.zeros_like(domain_data[0]))
            
        # Reshape to (n_devices, domains_per_device, ...)
        reshaped = []
        for device_id in range(self.n_devices):
            device_domains = []
            for i in range(domains_per_device):
                domain_idx = device_id * domains_per_device + i
                if domain_idx < len(domain_data):
                    device_domains.append(domain_data[domain_idx])
                else:
                    device_domains.append(jnp.zeros_like(domain_data[0]))
            reshaped.append(jnp.stack(device_domains))
            
        return jnp.stack(reshaped)
    
    def _reshape_from_pmap(self, pmap_data: jnp.ndarray) -> List:
        """Reshape pmap data back to domain list."""
        
        domain_data = []
        domains_per_device = pmap_data.shape[1]
        
        for device_id in range(self.n_devices):
            for i in range(domains_per_device):
                domain_idx = device_id * domains_per_device + i
                if domain_idx < self.total_domains:
                    domain_data.append(pmap_data[device_id, i])
                    
        return domain_data
    
    def _compute_parallel_efficiency(self) -> float:
        """Compute parallel efficiency metric."""
        
        # Simplified efficiency based on domain distribution
        ideal_domains_per_device = self.total_domains / self.n_devices
        actual_domains_per_device = max([len([d for d in self.domain_grids if d['device_id'] == i]) 
                                       for i in range(self.n_devices)])
        
        efficiency = ideal_domains_per_device / actual_domains_per_device
        return min(1.0, efficiency)
    
    def _estimate_memory_usage(self) -> Dict:
        """Estimate memory usage for the solver."""
        
        # Calculate memory per domain
        domain_size = np.prod([len(self.domain_grids[0]['x']),
                              len(self.domain_grids[0]['y']),
                              len(self.domain_grids[0]['z'])])
        
        bytes_per_element = 8  # float64
        memory_per_domain = domain_size * bytes_per_element * 2  # field + source
        total_memory = memory_per_domain * self.total_domains
        
        return {
            'memory_per_domain_mb': memory_per_domain / (1024**2),
            'total_memory_mb': total_memory / (1024**2),
            'memory_per_device_mb': total_memory / self.n_devices / (1024**2)
        }

def main():
    """Demonstration of JAX pmap domain decomposition solver."""
    from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
    
    print("="*70)
    print("JAX PMAP DOMAIN DECOMPOSITION SOLVER DEMONSTRATION")
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
    
    # Initialize parallel solver
    decomp_config = {
        'n_domains_x': 2,
        'n_domains_y': 2,
        'n_domains_z': 2,
        'overlap_points': 2,
        'boundary_type': 'dirichlet'
    }
    
    solver_config = {
        'max_iterations': 200,
        'tolerance': 1e-6,
        'schwarz_iterations': 3,
        'relaxation_parameter': 1.0
    }
    
    solver = JAXPmapDomainDecompositionSolver(transporter, decomp_config, solver_config)
    
    # Define source function (stress-energy density)
    def source_function(X, Y, Z, t):
        """Exotic matter stress-energy source."""
        r_cyl = jnp.sqrt(X**2 + Y**2)
        
        # Gaussian source in cylindrical coordinates
        source = jnp.exp(-(r_cyl - 1.0)**2 / 0.5) * jnp.exp(-(Z**2) / 25.0)
        source = source * jnp.sin(t * 2 * jnp.pi) * 1e-6  # Time-dependent
        
        return source
    
    # Solve field equation
    t = 2.0
    result = solver.solve_field_equation(source_function, t)
    
    # Analyze results
    print(f"\n📊 SOLUTION ANALYSIS")
    print("-" * 50)
    solution = result['solution']
    print(f"Solution shape: {solution.shape}")
    print(f"Solution range: [{jnp.min(solution):.2e}, {jnp.max(solution):.2e}]")
    print(f"Solution norm: {jnp.linalg.norm(solution):.2e}")
    
    print(f"\n⚡ PERFORMANCE METRICS")
    print("-" * 50)
    perf = result['performance']
    timing = result['timing']
    
    print(f"Parallel efficiency: {perf['parallel_efficiency']:.1%}")
    print(f"Memory per device: {perf['memory_usage']['memory_per_device_mb']:.1f} MB")
    print(f"Computational intensity: {perf['computational_intensity']:.1f} ops/s")
    print(f"Iterations per second: {timing['iterations_per_second']:.1f}")
    
    # Test scalability with different domain counts
    print(f"\n🔧 SCALABILITY TEST")
    print("-" * 50)
    
    domain_configs = [
        {'n_domains_x': 1, 'n_domains_y': 1, 'n_domains_z': 1},
        {'n_domains_x': 2, 'n_domains_y': 1, 'n_domains_z': 1},
        {'n_domains_x': 2, 'n_domains_y': 2, 'n_domains_z': 1}
    ]
    
    for i, test_config in enumerate(domain_configs):
        test_decomp = {**decomp_config, **test_config}
        test_solver = JAXPmapDomainDecompositionSolver(transporter, test_decomp, solver_config)
        
        test_result = test_solver.solve_field_equation(source_function, t)
        
        domains = test_config['n_domains_x'] * test_config['n_domains_y'] * test_config['n_domains_z']
        time_taken = test_result['timing']['total_time']
        
        print(f"Domains: {domains:2d}, Time: {time_taken:.3f}s, "
              f"Efficiency: {test_result['performance']['parallel_efficiency']:.1%}")
    
    target_performance = {
        'convergence': True,
        'parallel_efficiency': 0.8,
        'memory_per_device_mb': 500.0
    }
    
    print(f"\n🎯 TARGET ACHIEVEMENT")
    print("-" * 50)
    print(f"Convergence: {'✅' if result['converged'] else '❌'} "
          f"(target: {'✅' if target_performance['convergence'] else '❌'})")
    print(f"Parallel efficiency: {perf['parallel_efficiency']:.1%} "
          f"(target: {target_performance['parallel_efficiency']:.1%}) "
          f"{'✅' if perf['parallel_efficiency'] >= target_performance['parallel_efficiency'] else '⚠️'}")
    print(f"Memory per device: {perf['memory_usage']['memory_per_device_mb']:.1f} MB "
          f"(target: ≤{target_performance['memory_per_device_mb']:.0f} MB) "
          f"{'✅' if perf['memory_usage']['memory_per_device_mb'] <= target_performance['memory_per_device_mb'] else '⚠️'}")
    
    return solver

if __name__ == "__main__":
    solver = main()
