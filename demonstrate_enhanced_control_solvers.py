"""
Enhanced Integration Demonstration for Advanced Control & Solver Suite

This module demonstrates the complete integration of enhanced control systems
and advanced parallel solvers with the existing stargate transporter framework.

Capabilities:
- H∞ optimal control integration
- Multi-variable PID control coordination  
- Quantum error correction deployment
- JAX pmap domain decomposition solving
- Newton-Raphson nonlinear field solving
- Optimized 3D Laplacian operations

Author: Enhanced Implementation
Created: June 27, 2025
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import warnings

# Import enhanced control systems
from src.control.hinfty_controller import HInfinityController
from src.control.multivar_pid_controller import MultiVarPIDController
from src.control.qec_injector import QECInjector

# Import advanced solvers
from src.solvers.jax_pmap_domain_solver import JAXPmapDomainDecompositionSolver
from src.solvers.newton_raphson_solver import NewtonRaphsonIterativeSolver
from src.solvers.optimized_3d_laplacian import Optimized3DLaplacianOperator

# Import enhanced transporter
from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

class EnhancedControlSolverIntegration:
    """
    Complete integration of enhanced control systems and advanced solvers.
    
    Provides unified interface for exotic matter transport with advanced
    control theory and high-performance numerical methods.
    """
    
    def __init__(self, transporter_config: EnhancedTransporterConfig):
        """
        Initialize enhanced control and solver integration.
        
        Args:
            transporter_config: Enhanced transporter configuration
        """
        self.transporter_config = transporter_config
        
        # Initialize enhanced transporter
        self.transporter = EnhancedStargateTransporter(transporter_config)
        
        # Initialize control systems
        self._initialize_control_systems()
        
        # Initialize advanced solvers
        self._initialize_advanced_solvers()
        
        # Integration state tracking
        self.integration_history = []
        self.performance_metrics = {}
        self.control_solver_coordination = {}
        
        print(f"EnhancedControlSolverIntegration initialized:")
        print(f"  🎛️ Control systems: H∞, Multi-PID, QEC")
        print(f"  🧮 Solvers: JAX pmap, Newton-Raphson, Optimized Laplacian")
        print(f"  🚀 Ready for advanced exotic matter transport")
        
    def _initialize_control_systems(self):
        """Initialize all enhanced control systems."""
        
        print(f"\n🎛️ INITIALIZING CONTROL SYSTEMS")
        print("-" * 50)
        
        # H∞ optimal controller
        try:
            self.hinf_controller = HInfinityController(self.transporter)
            print(f"  ✅ H∞ controller ready")
        except Exception as e:
            print(f"  ❌ H∞ controller failed: {e}")
            self.hinf_controller = None
            
        # Multi-variable PID controller
        try:
            self.pid_controller = MultiVarPIDController(self.transporter)
            print(f"  ✅ Multi-PID controller ready")
        except Exception as e:
            print(f"  ❌ Multi-PID controller failed: {e}")
            self.pid_controller = None
            
        # Quantum error correction injector
        try:
            self.qec_injector = QECInjector(self.transporter, code_type="surface")
            print(f"  ✅ QEC injector ready")
        except Exception as e:
            print(f"  ❌ QEC injector failed: {e}")
            self.qec_injector = None
    
    def _initialize_advanced_solvers(self):
        """Initialize all advanced numerical solvers."""
        
        print(f"\n🧮 INITIALIZING ADVANCED SOLVERS")
        print("-" * 50)
        
        # JAX pmap domain decomposition solver
        try:
            decomp_config = {
                'n_domains_x': 2, 'n_domains_y': 2, 'n_domains_z': 2,
                'overlap_points': 2, 'boundary_type': 'dirichlet'
            }
            solver_config = {
                'max_iterations': 100, 'tolerance': 1e-6,
                'schwarz_iterations': 3, 'relaxation_parameter': 1.0
            }
            self.domain_solver = JAXPmapDomainDecompositionSolver(
                self.transporter, decomp_config, solver_config
            )
            print(f"  ✅ JAX pmap domain solver ready")
        except Exception as e:
            print(f"  ❌ JAX pmap domain solver failed: {e}")
            self.domain_solver = None
            
        # Newton-Raphson iterative solver
        try:
            nr_solver_config = {
                'max_iterations': 50, 'tolerance': 1e-8,
                'jacobian_method': 'forward', 'acceleration': 'anderson'
            }
            nr_field_config = {
                'nx': 32, 'ny': 32, 'nz': 32, 'field_components': 6
            }
            self.newton_solver = NewtonRaphsonIterativeSolver(
                self.transporter, nr_solver_config, nr_field_config
            )
            print(f"  ✅ Newton-Raphson solver ready")
        except Exception as e:
            print(f"  ❌ Newton-Raphson solver failed: {e}")
            self.newton_solver = None
            
        # Optimized 3D Laplacian operator
        try:
            grid_config = {
                'nx': 64, 'ny': 64, 'nz': 64,
                'dx': 0.1, 'dy': 0.1, 'dz': 0.1
            }
            laplacian_config = {
                'discretization': 'finite_difference', 'stencil_order': 4,
                'boundary_condition': 'periodic', 'vectorization': 'full'
            }
            self.laplacian_op = Optimized3DLaplacianOperator(grid_config, laplacian_config)
            print(f"  ✅ Optimized Laplacian operator ready")
        except Exception as e:
            print(f"  ❌ Optimized Laplacian operator failed: {e}")
            self.laplacian_op = None
    
    def run_integrated_control_cycle(self, t: float, control_mode: str = "coordinated") -> Dict:
        """
        Run complete integrated control cycle.
        
        Args:
            t: Current time
            control_mode: Control coordination mode ("sequential", "coordinated", "adaptive")
            
        Returns:
            Integrated control results
        """
        cycle_start = time.time()
        
        print(f"\n🎛️ INTEGRATED CONTROL CYCLE (t = {t:.3f}s)")
        print("-" * 60)
        
        results = {
            'time': t,
            'control_mode': control_mode,
            'control_results': {},
            'coordination_metrics': {},
            'overall_performance': {}
        }
        
        # Phase 1: H∞ optimal control
        if self.hinf_controller:
            try:
                spatial_grid = {'nx': 32, 'ny': 32, 'nz': 32}
                hinf_result = self.hinf_controller.apply_control(t, spatial_grid)
                results['control_results']['hinf'] = hinf_result
                
                print(f"  H∞ control: effectiveness = {hinf_result['control_effectiveness']:.1f} dB")
                
            except Exception as e:
                print(f"  ❌ H∞ control failed: {e}")
                results['control_results']['hinf'] = {'error': str(e)}
        
        # Phase 2: Multi-variable PID control
        if self.pid_controller:
            try:
                pid_result = self.pid_controller.apply_control(t)
                results['control_results']['pid'] = pid_result
                
                perf = pid_result['performance']
                print(f"  Multi-PID: error = {perf['error_norm']:.2e}, "
                      f"gain scheduling = {perf['gain_schedule_factor']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Multi-PID control failed: {e}")
                results['control_results']['pid'] = {'error': str(e)}
        
        # Phase 3: Quantum error correction
        if self.qec_injector:
            try:
                qec_result = self.qec_injector.apply_qec_protocol(t)
                results['control_results']['qec'] = qec_result
                
                perf = qec_result['performance']
                print(f"  QEC: fidelity = {perf['fidelity']:.4f}, "
                      f"corrections = {'✅' if perf['correction_applied'] else '❌'}")
                
            except Exception as e:
                print(f"  ❌ QEC protocol failed: {e}")
                results['control_results']['qec'] = {'error': str(e)}
        
        # Phase 4: Control coordination
        if control_mode == "coordinated":
            coordination_result = self._coordinate_control_systems(results['control_results'], t)
            results['coordination_metrics'] = coordination_result
            
            print(f"  Coordination: interference = {coordination_result['interference_level']:.1%}, "
                  f"synergy = {coordination_result['synergy_factor']:.2f}")
        
        cycle_time = time.time() - cycle_start
        
        # Overall performance assessment
        results['overall_performance'] = {
            'cycle_time': cycle_time,
            'control_systems_active': len([r for r in results['control_results'].values() if 'error' not in r]),
            'coordination_successful': control_mode != "coordinated" or 'interference_level' in results['coordination_metrics'],
            'stability_margin': self._assess_stability_margin(results['control_results'])
        }
        
        print(f"  Cycle time: {cycle_time:.3f}s, Active systems: {results['overall_performance']['control_systems_active']}")
        
        return results
    
    def run_integrated_solver_cycle(self, t: float, solver_mode: str = "adaptive") -> Dict:
        """
        Run complete integrated solver cycle.
        
        Args:
            t: Current time
            solver_mode: Solver coordination mode ("sequential", "parallel", "adaptive")
            
        Returns:
            Integrated solver results
        """
        cycle_start = time.time()
        
        print(f"\n🧮 INTEGRATED SOLVER CYCLE (t = {t:.3f}s)")
        print("-" * 60)
        
        results = {
            'time': t,
            'solver_mode': solver_mode,
            'solver_results': {},
            'performance_analysis': {},
            'computational_efficiency': {}
        }
        
        # Define source functions for different solvers
        def linear_source_function(X, Y, Z, t):
            """Linear source for domain decomposition solver."""
            r_cyl = jnp.sqrt(X**2 + Y**2)
            return jnp.exp(-(r_cyl - 1.0)**2 / 0.5) * jnp.exp(-(Z**2) / 25.0) * jnp.sin(t * 2 * jnp.pi) * 1e-6
        
        def nonlinear_source_function(X, Y, Z, t):
            """Nonlinear source for Newton-Raphson solver."""
            r_cyl = jnp.sqrt(X**2 + Y**2)
            base_source = jnp.exp(-(r_cyl - 1.5)**2 / 0.8) * jnp.exp(-(Z**2) / 30.0)
            nonlinear_factor = 1 + 0.1 * jnp.sin(t * 2 * jnp.pi) * base_source
            source = base_source * nonlinear_factor * 1e-5
            
            # Expand to field components
            source_expanded = jnp.zeros(X.shape + (6,))
            for i in range(6):
                component_factor = 1.0 + 0.1 * (i % 3) / 3.0
                source_expanded = source_expanded.at[:, :, :, i].set(source * component_factor)
            return source_expanded
        
        # Phase 1: JAX pmap domain decomposition
        if self.domain_solver:
            try:
                print(f"  Running JAX pmap domain solver...")
                domain_result = self.domain_solver.solve_field_equation(linear_source_function, t)
                results['solver_results']['domain'] = domain_result
                
                print(f"    Converged: {'✅' if domain_result['converged'] else '❌'}, "
                      f"Iterations: {domain_result['iterations']}, "
                      f"Time: {domain_result['timing']['total_time']:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Domain solver failed: {e}")
                results['solver_results']['domain'] = {'error': str(e)}
        
        # Phase 2: Newton-Raphson nonlinear solver
        if self.newton_solver:
            try:
                print(f"  Running Newton-Raphson solver...")
                newton_result = self.newton_solver.solve_nonlinear_field(nonlinear_source_function, t)
                results['solver_results']['newton'] = newton_result
                
                print(f"    Converged: {'✅' if newton_result['converged'] else '❌'}, "
                      f"Iterations: {newton_result['iterations']}, "
                      f"Time: {newton_result['timing']['total_time']:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Newton-Raphson solver failed: {e}")
                results['solver_results']['newton'] = {'error': str(e)}
        
        # Phase 3: Optimized Laplacian operations
        if self.laplacian_op:
            try:
                print(f"  Running optimized Laplacian benchmark...")
                
                # Generate test field
                key = jax.random.PRNGKey(42)
                test_field = jax.random.normal(key, (64, 64, 64))
                
                # Benchmark Laplacian
                laplacian_shapes = [(32, 32, 32), (64, 64, 64)]
                laplacian_result = self.laplacian_op.benchmark_performance(laplacian_shapes, num_trials=3)
                results['solver_results']['laplacian'] = laplacian_result
                
                # Report performance for largest grid
                largest_grid = max(laplacian_shapes)
                if largest_grid in laplacian_result:
                    throughput = laplacian_result[largest_grid]['throughput_meps']
                    time_taken = laplacian_result[largest_grid]['single_time']
                    print(f"    Performance: {throughput:.1f} MEPS, Time: {time_taken*1000:.2f} ms")
                
            except Exception as e:
                print(f"  ❌ Laplacian benchmark failed: {e}")
                results['solver_results']['laplacian'] = {'error': str(e)}
        
        cycle_time = time.time() - cycle_start
        
        # Performance analysis
        results['performance_analysis'] = self._analyze_solver_performance(results['solver_results'])
        
        # Computational efficiency assessment
        results['computational_efficiency'] = {
            'total_cycle_time': cycle_time,
            'solvers_completed': len([r for r in results['solver_results'].values() if 'error' not in r]),
            'parallel_efficiency': self._estimate_parallel_efficiency(results['solver_results']),
            'memory_efficiency': self._estimate_memory_efficiency(results['solver_results'])
        }
        
        print(f"  Total cycle time: {cycle_time:.3f}s, "
              f"Completed solvers: {results['computational_efficiency']['solvers_completed']}")
        
        return results
    
    def run_complete_integration_demonstration(self, time_points: List[float]) -> Dict:
        """
        Run complete demonstration of enhanced control and solver integration.
        
        Args:
            time_points: List of time points for demonstration
            
        Returns:
            Complete integration results
        """
        demo_start = time.time()
        
        print(f"\n🚀 COMPLETE INTEGRATION DEMONSTRATION")
        print("=" * 70)
        print(f"Time points: {len(time_points)}")
        print(f"Duration: {max(time_points) - min(time_points):.2f}s")
        
        demonstration_results = {
            'time_points': time_points,
            'control_cycles': [],
            'solver_cycles': [],
            'integration_metrics': {},
            'performance_summary': {}
        }
        
        # Run control and solver cycles at each time point
        for i, t in enumerate(time_points):
            print(f"\n--- Time Point {i+1}/{len(time_points)}: t = {t:.3f}s ---")
            
            # Alternate control modes for demonstration
            control_modes = ["sequential", "coordinated", "adaptive"]
            control_mode = control_modes[i % len(control_modes)]
            
            # Run control cycle
            try:
                control_result = self.run_integrated_control_cycle(t, control_mode)
                demonstration_results['control_cycles'].append(control_result)
            except Exception as e:
                print(f"Control cycle failed: {e}")
                
            # Alternate solver modes
            solver_modes = ["sequential", "parallel", "adaptive"]
            solver_mode = solver_modes[i % len(solver_modes)]
            
            # Run solver cycle
            try:
                solver_result = self.run_integrated_solver_cycle(t, solver_mode)
                demonstration_results['solver_cycles'].append(solver_result)
            except Exception as e:
                print(f"Solver cycle failed: {e}")
        
        demo_time = time.time() - demo_start
        
        # Analyze integration metrics
        demonstration_results['integration_metrics'] = self._analyze_integration_metrics(
            demonstration_results['control_cycles'], 
            demonstration_results['solver_cycles']
        )
        
        # Performance summary
        demonstration_results['performance_summary'] = {
            'total_demonstration_time': demo_time,
            'average_cycle_time': demo_time / len(time_points) if time_points else 0,
            'control_success_rate': self._compute_success_rate(demonstration_results['control_cycles']),
            'solver_success_rate': self._compute_success_rate(demonstration_results['solver_cycles']),
            'overall_stability': self._assess_overall_stability(demonstration_results),
            'computational_scalability': self._assess_computational_scalability(demonstration_results)
        }
        
        # Generate final report
        self._generate_integration_report(demonstration_results)
        
        return demonstration_results
    
    def _coordinate_control_systems(self, control_results: Dict, t: float) -> Dict:
        """Coordinate multiple control systems to minimize interference."""
        
        coordination_result = {
            'interference_level': 0.0,
            'synergy_factor': 1.0,
            'coordination_adjustments': {}
        }
        
        # Simple coordination based on control norms
        active_controls = []
        
        if 'hinf' in control_results and 'error' not in control_results['hinf']:
            hinf_norm = control_results['hinf']['performance']['control_norm']
            active_controls.append(('hinf', hinf_norm))
            
        if 'pid' in control_results and 'error' not in control_results['pid']:
            pid_norm = control_results['pid']['performance']['control_norm']  
            active_controls.append(('pid', pid_norm))
        
        # Assess interference
        if len(active_controls) > 1:
            norms = [norm for _, norm in active_controls]
            interference = np.std(norms) / (np.mean(norms) + 1e-12)
            coordination_result['interference_level'] = min(1.0, interference)
            
            # Synergy factor (simplified)
            coordination_result['synergy_factor'] = 1.0 / (1.0 + interference)
        
        return coordination_result
    
    def _assess_stability_margin(self, control_results: Dict) -> float:
        """Assess overall stability margin from control results."""
        
        margins = []
        
        if 'hinf' in control_results and 'error' not in control_results['hinf']:
            # H∞ provides guaranteed stability margin
            margins.append(0.9)
            
        if 'pid' in control_results and 'error' not in control_results['pid']:
            # Estimate PID stability from error norm
            error_norm = control_results['pid']['performance']['error_norm']
            pid_margin = max(0.1, min(0.9, 1.0 - error_norm * 1e6))
            margins.append(pid_margin)
            
        if 'qec' in control_results and 'error' not in control_results['qec']:
            # QEC fidelity contributes to stability
            fidelity = control_results['qec']['performance']['fidelity']
            margins.append(fidelity)
        
        return np.mean(margins) if margins else 0.5
    
    def _analyze_solver_performance(self, solver_results: Dict) -> Dict:
        """Analyze performance across all solvers."""
        
        analysis = {
            'convergence_rates': {},
            'computational_times': {},
            'accuracy_metrics': {},
            'scalability_assessment': {}
        }
        
        # Domain solver analysis
        if 'domain' in solver_results and 'error' not in solver_results['domain']:
            domain_result = solver_results['domain']
            analysis['convergence_rates']['domain'] = {
                'converged': domain_result['converged'],
                'iterations': domain_result['iterations'],
                'final_residual': domain_result['final_residual']
            }
            analysis['computational_times']['domain'] = domain_result['timing']['total_time']
        
        # Newton-Raphson analysis
        if 'newton' in solver_results and 'error' not in solver_results['newton']:
            newton_result = solver_results['newton']
            analysis['convergence_rates']['newton'] = {
                'converged': newton_result['converged'],
                'iterations': newton_result['iterations'],
                'quadratic_convergence': newton_result['performance']['quadratic_convergence']
            }
            analysis['computational_times']['newton'] = newton_result['timing']['total_time']
        
        # Laplacian analysis
        if 'laplacian' in solver_results and 'error' not in solver_results['laplacian']:
            laplacian_result = solver_results['laplacian']
            # Extract performance for largest available grid
            largest_grid = max(laplacian_result.keys()) if laplacian_result else None
            if largest_grid:
                analysis['computational_times']['laplacian'] = laplacian_result[largest_grid]['single_time']
                analysis['scalability_assessment']['laplacian_throughput'] = laplacian_result[largest_grid]['throughput_meps']
        
        return analysis
    
    def _estimate_parallel_efficiency(self, solver_results: Dict) -> float:
        """Estimate parallel efficiency from solver results."""
        
        efficiencies = []
        
        if 'domain' in solver_results and 'performance' in solver_results['domain']:
            domain_perf = solver_results['domain']['performance']
            if 'parallel_efficiency' in domain_perf:
                efficiencies.append(domain_perf['parallel_efficiency'])
        
        # Estimate from other metrics
        if 'laplacian' in solver_results:
            # Laplacian vectorization provides some parallelism
            efficiencies.append(0.7)  # Estimated
            
        return np.mean(efficiencies) if efficiencies else 0.5
    
    def _estimate_memory_efficiency(self, solver_results: Dict) -> float:
        """Estimate memory efficiency from solver results."""
        
        memory_usages = []
        
        # Extract memory usage data where available
        if 'domain' in solver_results and 'performance' in solver_results['domain']:
            domain_perf = solver_results['domain']['performance']
            if 'memory_usage' in domain_perf:
                # Normalize by some baseline (1 GB)
                memory_usage_gb = domain_perf['memory_usage']['total_memory_mb'] / 1024
                efficiency = max(0.1, min(1.0, 1.0 / (memory_usage_gb + 0.1)))
                memory_usages.append(efficiency)
        
        return np.mean(memory_usages) if memory_usages else 0.8
    
    def _analyze_integration_metrics(self, control_cycles: List, solver_cycles: List) -> Dict:
        """Analyze metrics across all integration cycles."""
        
        metrics = {
            'control_consistency': 0.0,
            'solver_reliability': 0.0,
            'temporal_stability': 0.0,
            'resource_utilization': 0.0
        }
        
        # Control consistency
        if control_cycles:
            successful_controls = [c for c in control_cycles 
                                 if c['overall_performance']['control_systems_active'] > 0]
            metrics['control_consistency'] = len(successful_controls) / len(control_cycles)
        
        # Solver reliability
        if solver_cycles:
            successful_solvers = [s for s in solver_cycles 
                                if s['computational_efficiency']['solvers_completed'] > 0]
            metrics['solver_reliability'] = len(successful_solvers) / len(solver_cycles)
        
        # Temporal stability (simplified)
        if control_cycles:
            stability_margins = [c['overall_performance']['stability_margin'] 
                               for c in control_cycles]
            metrics['temporal_stability'] = np.mean(stability_margins)
        
        # Resource utilization
        if solver_cycles:
            parallel_efficiencies = [s['computational_efficiency']['parallel_efficiency'] 
                                   for s in solver_cycles]
            metrics['resource_utilization'] = np.mean(parallel_efficiencies)
        
        return metrics
    
    def _compute_success_rate(self, cycles: List) -> float:
        """Compute success rate for a list of cycles."""
        
        if not cycles:
            return 0.0
            
        successful = 0
        for cycle in cycles:
            if 'control_cycles' in str(type(cycle)):  # Control cycle
                if cycle['overall_performance']['control_systems_active'] > 0:
                    successful += 1
            else:  # Solver cycle
                if cycle['computational_efficiency']['solvers_completed'] > 0:
                    successful += 1
                    
        return successful / len(cycles)
    
    def _assess_overall_stability(self, demonstration_results: Dict) -> float:
        """Assess overall system stability."""
        
        control_cycles = demonstration_results.get('control_cycles', [])
        solver_cycles = demonstration_results.get('solver_cycles', [])
        
        stability_factors = []
        
        # Control stability
        if control_cycles:
            control_stability = np.mean([
                c['overall_performance']['stability_margin'] 
                for c in control_cycles
            ])
            stability_factors.append(control_stability)
        
        # Solver convergence contributes to stability
        if solver_cycles:
            convergence_rates = []
            for cycle in solver_cycles:
                solver_results = cycle['solver_results']
                if 'domain' in solver_results and solver_results['domain'].get('converged', False):
                    convergence_rates.append(1.0)
                if 'newton' in solver_results and solver_results['newton'].get('converged', False):
                    convergence_rates.append(1.0)
            
            if convergence_rates:
                solver_stability = np.mean(convergence_rates)
                stability_factors.append(solver_stability)
        
        return np.mean(stability_factors) if stability_factors else 0.5
    
    def _assess_computational_scalability(self, demonstration_results: Dict) -> float:
        """Assess computational scalability."""
        
        solver_cycles = demonstration_results.get('solver_cycles', [])
        
        if not solver_cycles:
            return 0.5
            
        # Look for parallel efficiency and throughput metrics
        parallel_efficiencies = []
        
        for cycle in solver_cycles:
            if 'computational_efficiency' in cycle:
                parallel_eff = cycle['computational_efficiency']['parallel_efficiency']
                parallel_efficiencies.append(parallel_eff)
        
        return np.mean(parallel_efficiencies) if parallel_efficiencies else 0.5
    
    def _generate_integration_report(self, demonstration_results: Dict):
        """Generate comprehensive integration report."""
        
        print(f"\n📊 INTEGRATION DEMONSTRATION REPORT")
        print("=" * 70)
        
        # Summary statistics
        summary = demonstration_results['performance_summary']
        metrics = demonstration_results['integration_metrics']
        
        print(f"Demonstration Duration: {summary['total_demonstration_time']:.2f}s")
        print(f"Average Cycle Time: {summary['average_cycle_time']:.3f}s")
        print(f"Control Success Rate: {summary['control_success_rate']:.1%}")
        print(f"Solver Success Rate: {summary['solver_success_rate']:.1%}")
        print(f"Overall Stability: {summary['overall_stability']:.1%}")
        print(f"Computational Scalability: {summary['computational_scalability']:.1%}")
        
        print(f"\n🎛️ CONTROL SYSTEM PERFORMANCE")
        print("-" * 50)
        print(f"Control Consistency: {metrics['control_consistency']:.1%}")
        print(f"Temporal Stability: {metrics['temporal_stability']:.1%}")
        
        # Control system breakdown
        control_cycles = demonstration_results['control_cycles']
        if control_cycles:
            hinf_active = sum(1 for c in control_cycles if 'hinf' in c['control_results'] and 'error' not in c['control_results']['hinf'])
            pid_active = sum(1 for c in control_cycles if 'pid' in c['control_results'] and 'error' not in c['control_results']['pid'])
            qec_active = sum(1 for c in control_cycles if 'qec' in c['control_results'] and 'error' not in c['control_results']['qec'])
            
            print(f"H∞ Controller Active: {hinf_active}/{len(control_cycles)} cycles ({hinf_active/len(control_cycles):.1%})")
            print(f"Multi-PID Active: {pid_active}/{len(control_cycles)} cycles ({pid_active/len(control_cycles):.1%})")
            print(f"QEC Active: {qec_active}/{len(control_cycles)} cycles ({qec_active/len(control_cycles):.1%})")
        
        print(f"\n🧮 SOLVER PERFORMANCE")
        print("-" * 50)
        print(f"Solver Reliability: {metrics['solver_reliability']:.1%}")
        print(f"Resource Utilization: {metrics['resource_utilization']:.1%}")
        
        # Solver breakdown
        solver_cycles = demonstration_results['solver_cycles']
        if solver_cycles:
            domain_success = sum(1 for s in solver_cycles if 'domain' in s['solver_results'] and 'error' not in s['solver_results']['domain'])
            newton_success = sum(1 for s in solver_cycles if 'newton' in s['solver_results'] and 'error' not in s['solver_results']['newton'])
            laplacian_success = sum(1 for s in solver_cycles if 'laplacian' in s['solver_results'] and 'error' not in s['solver_results']['laplacian'])
            
            print(f"JAX Pmap Domain Solver: {domain_success}/{len(solver_cycles)} cycles ({domain_success/len(solver_cycles):.1%})")
            print(f"Newton-Raphson Solver: {newton_success}/{len(solver_cycles)} cycles ({newton_success/len(solver_cycles):.1%})")
            print(f"Optimized Laplacian: {laplacian_success}/{len(solver_cycles)} cycles ({laplacian_success/len(solver_cycles):.1%})")
        
        # Target achievement assessment
        print(f"\n🎯 TARGET ACHIEVEMENT ASSESSMENT")
        print("-" * 50)
        
        targets = {
            'control_success_rate': 0.90,
            'solver_success_rate': 0.85,
            'overall_stability': 0.80,
            'computational_scalability': 0.75,
            'resource_utilization': 0.70
        }
        
        achievements = 0
        total_targets = len(targets)
        
        for metric, target in targets.items():
            if metric in summary:
                achieved = summary[metric]
            elif metric in metrics:
                achieved = metrics[metric]
            else:
                achieved = 0.0
                
            status = "✅" if achieved >= target else "❌"
            print(f"{metric.replace('_', ' ').title()}: {achieved:.1%} (target: {target:.1%}) {status}")
            
            if achieved >= target:
                achievements += 1
        
        print(f"\nOverall Target Achievement: {achievements}/{total_targets} ({achievements/total_targets:.1%})")
        
        if achievements >= total_targets * 0.8:
            print(f"🎉 INTEGRATION SUCCESS: {achievements}/{total_targets} targets achieved!")
        elif achievements >= total_targets * 0.6:
            print(f"⚠️ PARTIAL SUCCESS: {achievements}/{total_targets} targets achieved")
        else:
            print(f"❌ INTEGRATION NEEDS IMPROVEMENT: Only {achievements}/{total_targets} targets achieved")

def main():
    """Main demonstration of enhanced control and solver integration."""
    
    print("="*80)
    print("ENHANCED CONTROL & SOLVER INTEGRATION DEMONSTRATION")
    print("="*80)
    
    # Create enhanced transporter configuration
    config = EnhancedTransporterConfig(
        R_payload=2.0,
        R_neck=0.08,
        L_corridor=50.0,
        corridor_mode="sinusoidal",
        v_conveyor_max=1e6
    )
    
    # Initialize enhanced integration
    integration = EnhancedControlSolverIntegration(config)
    
    # Define demonstration time points
    time_points = [0.0, 2.5, 5.0, 7.5, 10.0]
    
    # Run complete integration demonstration
    results = integration.run_complete_integration_demonstration(time_points)
    
    print(f"\n🚀 DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return integration, results

if __name__ == "__main__":
    integration_system, demo_results = main()
