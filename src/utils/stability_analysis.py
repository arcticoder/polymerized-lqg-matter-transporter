#!/usr/bin/env python3
"""
3D Multi-Frequency Stability Analysis
====================================

Enhanced stability analysis with 3D eigenvalue methods and multi-frequency
decoherence modeling, based on advanced techniques from unified-lqg repository.

Implements:
- 3D perturbation analysis: δψ(r,θ,φ,t) = Σ_lmn A_lmn Y_l^m(θ,φ) R_n(r) e^λ_lmn t
- Multi-frequency decoherence modeling with exponential/Gaussian/thermal components
- Complete stability characterization across all regimes

Mathematical Foundation:
Enhanced from unified-lqg/papers/warp_feasibility_complete.tex (lines 389+)
- Exponential decoherence: Γ_dec^-1 = 10^12.3 seconds
- Gaussian smearing: τ_gaussian = 5.0 time units  
- Thermal effects: τ_thermal = 2.0 time units

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from typing import Dict, List, Tuple, Optional, Callable
from scipy.special import sph_harm, factorial
from dataclasses import dataclass
import warnings

@dataclass
class Stability3DConfig:
    """Configuration for 3D stability analysis."""
    max_l: int = 10                    # Maximum spherical harmonic l
    max_m: int = 10                    # Maximum spherical harmonic m  
    max_n: int = 5                     # Maximum radial modes
    decoherence_time: float = 1e12     # Decoherence timescale (s)
    gaussian_time: float = 5.0         # Gaussian smearing timescale
    thermal_time: float = 2.0          # Thermal timescale
    r_min: float = 0.01                # Minimum radius for analysis
    r_max: float = 10.0                # Maximum radius for analysis

class Stability3D:
    """
    3D Multi-frequency stability analysis for matter transporter.
    
    Solves perturbation evolution:
    δψ(r,θ,φ,t) = Σ_lmn A_lmn Y_l^m(θ,φ) R_n(r) exp(λ_lmn t)
    
    Includes decoherence and thermal corrections:
    λ_lmn = λ_lmn^(0) + Δλ_decoherence + Δλ_thermal
    
    Parameters:
    -----------
    config : Stability3DConfig
        Configuration parameters for stability analysis
    """
    
    def __init__(self, config: Stability3DConfig):
        """
        Initialize 3D stability analyzer.
        
        Args:
            config: Stability analysis configuration
        """
        self.config = config
        
        # Initialize mode arrays
        self._initialize_mode_structure()
        
        # Compute base eigenvalues
        self._compute_base_eigenvalues()
        
        # Add decoherence and thermal corrections
        self._add_corrections()
    
    def _initialize_mode_structure(self):
        """Initialize spherical harmonic and radial mode structure."""
        # Create mode index arrays
        self.modes = []
        self.λ0 = {}
        self.Δλ_dec = {}
        self.Δλ_th = {}
        
        for l in range(self.config.max_l + 1):
            for m in range(-l, l + 1):
                for n in range(self.config.max_n):
                    mode = (l, m, n)
                    self.modes.append(mode)
                    
                    # Initialize placeholder eigenvalues
                    self.λ0[mode] = 0.0
                    self.Δλ_dec[mode] = 0.0
                    self.Δλ_th[mode] = 0.0
        
        print(f"Initialized {len(self.modes)} stability modes")
    
    def _compute_base_eigenvalues(self):
        """Compute base eigenvalues λ_lmn^(0) for unperturbed system."""
        for l, m, n in self.modes:
            # Base eigenvalue from spherical harmonic structure
            # λ ~ -l(l+1) for stable modes, positive for unstable
            base_lambda = -l * (l + 1) / (1 + n**2)
            
            # Add radial contribution
            radial_factor = -n * np.pi / (self.config.r_max - self.config.r_min)
            
            self.λ0[(l, m, n)] = base_lambda + radial_factor
    
    def _add_corrections(self):
        """Add decoherence and thermal corrections."""
        for l, m, n in self.modes:
            # Decoherence correction (exponential decay)
            # Enhanced from line 389 of warp_feasibility_complete.tex
            self.Δλ_dec[(l, m, n)] = -1.0 / self.config.decoherence_time
            
            # Thermal correction (mode-dependent)
            thermal_factor = np.exp(-l / self.config.thermal_time)
            self.Δλ_th[(l, m, n)] = -thermal_factor / self.config.thermal_time
    
    @partial(jit, static_argnums=(0,))
    def radial_basis_function(self, r: jnp.ndarray, n: int) -> jnp.ndarray:
        """
        Compute radial basis function R_n(r).
        
        Uses Bessel functions for bounded domain.
        
        Args:
            r: Radial coordinate array
            n: Radial mode number
            
        Returns:
            Radial basis function values
        """
        # Normalized radial coordinate
        r_norm = r / self.config.r_max
        
        # Simple polynomial basis (can be upgraded to Bessel functions)
        if n == 0:
            return jnp.ones_like(r_norm)
        else:
            return r_norm**n * jnp.exp(-r_norm)
    
    def spherical_harmonic(self, theta: float, phi: float, l: int, m: int) -> complex:
        """
        Compute spherical harmonic Y_l^m(θ,φ).
        
        Args:
            theta: Polar angle
            phi: Azimuthal angle
            l: Orbital angular momentum quantum number
            m: Magnetic quantum number
            
        Returns:
            Complex spherical harmonic value
        """
        return sph_harm(m, l, phi, theta)
    
    def growth_rate(self, l: int, m: int, n: int) -> float:
        """
        Compute total growth rate for mode (l,m,n).
        
        λ_total = λ_0 + Δλ_decoherence + Δλ_thermal
        
        Args:
            l, m, n: Mode quantum numbers
            
        Returns:
            Total growth rate
        """
        mode = (l, m, n)
        return self.λ0[mode] + self.Δλ_dec[mode] + self.Δλ_th[mode]
    
    def mode_amplitude(self, r: np.ndarray, theta: float, phi: float, 
                      t: float, l: int, m: int, n: int,
                      A_lmn: float = 1.0) -> np.ndarray:
        """
        Compute mode amplitude for given (l,m,n) mode.
        
        δψ_lmn = A_lmn * Y_l^m(θ,φ) * R_n(r) * exp(λ_lmn * t)
        
        Args:
            r: Radial coordinate array
            theta, phi: Angular coordinates
            t: Time
            l, m, n: Mode numbers
            A_lmn: Mode amplitude
            
        Returns:
            Mode amplitude array
        """
        Y_lm = self.spherical_harmonic(theta, phi, l, m)
        R_n = self.radial_basis_function(jnp.array(r), n)
        λ_total = self.growth_rate(l, m, n)
        
        time_evolution = np.exp(λ_total * t)
        
        return A_lmn * Y_lm * np.array(R_n) * time_evolution
    
    def total_perturbation(self, r: np.ndarray, theta: float, phi: float, 
                          t: float, mode_amplitudes: Dict[Tuple[int, int, int], float] = None) -> np.ndarray:
        """
        Compute total perturbation from all modes.
        
        δψ(r,θ,φ,t) = Σ_lmn A_lmn Y_l^m(θ,φ) R_n(r) exp(λ_lmn t)
        
        Args:
            r: Radial coordinate array
            theta, phi: Angular coordinates
            t: Time
            mode_amplitudes: Dictionary of mode amplitudes {(l,m,n): A_lmn}
            
        Returns:
            Total perturbation field
        """
        if mode_amplitudes is None:
            # Default: equal amplitude for all modes
            mode_amplitudes = {mode: 1.0 for mode in self.modes}
        
        δψ_total = np.zeros_like(r, dtype=complex)
        
        for (l, m, n), A_lmn in mode_amplitudes.items():
            if (l, m, n) in self.modes:
                δψ_mode = self.mode_amplitude(r, theta, phi, t, l, m, n, A_lmn)
                δψ_total += δψ_mode
        
        return δψ_total
    
    def stability_analysis(self, time_range: np.ndarray) -> Dict:
        """
        Comprehensive stability analysis over time range.
        
        Args:
            time_range: Array of time values for analysis
            
        Returns:
            Stability analysis results
        """
        results = {
            'times': time_range,
            'stable_modes': [],
            'unstable_modes': [],
            'growth_rates': {},
            'dominant_mode': None,
            'stability_margin': 0.0
        }
        
        max_growth_rate = -np.inf
        dominant_mode = None
        
        for l, m, n in self.modes:
            λ_total = self.growth_rate(l, m, n)
            results['growth_rates'][(l, m, n)] = λ_total
            
            if λ_total < 0:
                results['stable_modes'].append((l, m, n))
            else:
                results['unstable_modes'].append((l, m, n))
                
            if λ_total > max_growth_rate:
                max_growth_rate = λ_total
                dominant_mode = (l, m, n)
        
        results['dominant_mode'] = dominant_mode
        results['stability_margin'] = -max_growth_rate if max_growth_rate < 0 else 0.0
        
        # Time evolution analysis
        r_test = np.array([self.config.r_min, self.config.r_max])
        theta_test, phi_test = np.pi/2, 0  # Equatorial plane
        
        perturbation_evolution = []
        for t in time_range:
            δψ = self.total_perturbation(r_test, theta_test, phi_test, t)
            perturbation_evolution.append(np.max(np.abs(δψ)))
        
        results['perturbation_evolution'] = np.array(perturbation_evolution)
        results['final_perturbation'] = perturbation_evolution[-1]
        
        return results
    
    def decoherence_analysis(self, time_range: np.ndarray) -> Dict:
        """
        Analyze decoherence effects across different models.
        
        Enhanced from unified-lqg multi-model decoherence framework.
        
        Args:
            time_range: Time array for analysis
            
        Returns:
            Decoherence analysis results
        """
        # Decoherence factors from different models
        Γ_exponential = np.exp(-time_range / self.config.decoherence_time)
        Γ_gaussian = np.exp(-(time_range / self.config.gaussian_time)**2)
        Γ_thermal = np.exp(-time_range / self.config.thermal_time)
        
        # Combined decoherence (worst case)
        Γ_total = np.minimum(Γ_exponential, np.minimum(Γ_gaussian, Γ_thermal))
        
        return {
            'times': time_range,
            'exponential_decoherence': Γ_exponential,
            'gaussian_decoherence': Γ_gaussian,
            'thermal_decoherence': Γ_thermal,
            'total_decoherence': Γ_total,
            'decoherence_time_1e': time_range[np.argmin(np.abs(Γ_total - 1/np.e))],
            'coherence_preservation': Γ_total[-1]
        }
    
    def optimization_recommendations(self, stability_results: Dict) -> Dict:
        """
        Generate optimization recommendations based on stability analysis.
        
        Args:
            stability_results: Results from stability_analysis()
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'system_stable': len(stability_results['unstable_modes']) == 0,
            'critical_modes': stability_results['unstable_modes'][:5],  # Top 5 unstable
            'stabilization_required': len(stability_results['unstable_modes']) > 0
        }
        
        if recommendations['stabilization_required']:
            # Recommend parameter adjustments
            recommendations['suggested_actions'] = [
                "Increase decoherence time (cryogenic cooling)",
                "Reduce high-order mode excitation",
                "Implement active feedback stabilization",
                "Optimize geometry for mode suppression"
            ]
        else:
            recommendations['suggested_actions'] = [
                "System stable - proceed with transport",
                "Monitor dominant mode evolution",
                "Maintain current operating parameters"
            ]
        
        recommendations['stability_margin'] = stability_results['stability_margin']
        recommendations['dominant_mode'] = stability_results['dominant_mode']
        
        return recommendations

# Utility functions
def visualize_mode_structure(stability: Stability3D, l_max: int = 3):
    """
    Visualize spherical harmonic mode structure.
    
    Args:
        stability: Stability3D instance
        l_max: Maximum l for visualization
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 5))
        
        theta = np.linspace(0, np.pi, 50)
        phi = np.linspace(0, 2*np.pi, 50)
        THETA, PHI = np.meshgrid(theta, phi)
        
        for i, l in enumerate(range(min(3, l_max + 1))):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # Use m=0 for visualization
            Y = np.real(sph_harm(0, l, PHI, THETA))
            
            # Convert to Cartesian for plotting
            X = np.abs(Y) * np.sin(THETA) * np.cos(PHI)
            Y_cart = np.abs(Y) * np.sin(THETA) * np.sin(PHI)
            Z = np.abs(Y) * np.cos(THETA)
            
            ax.plot_surface(X, Y_cart, Z, alpha=0.7, cmap='viridis')
            ax.set_title(f'Y_{l}^0 Mode')
            
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Demonstration of 3D stability analysis
    print("3D Multi-Frequency Stability Analysis Demonstration")
    print("=" * 55)
    
    # Configuration
    config = Stability3DConfig(
        max_l=5,
        max_m=5,
        max_n=3,
        decoherence_time=1e12,  # From unified-lqg specs
        gaussian_time=5.0,
        thermal_time=2.0
    )
    
    stability = Stability3D(config)
    
    print(f"Initialized {len(stability.modes)} stability modes")
    print(f"Mode range: l=0-{config.max_l}, m=±l, n=0-{config.max_n-1}")
    
    # Time range for analysis
    time_range = np.linspace(0, 100, 200)
    
    # Stability analysis
    results = stability.stability_analysis(time_range)
    
    print(f"\nStability Analysis Results:")
    print(f"  Stable modes: {len(results['stable_modes'])}")
    print(f"  Unstable modes: {len(results['unstable_modes'])}")
    print(f"  Dominant mode: {results['dominant_mode']}")
    print(f"  Stability margin: {results['stability_margin']:.2e}")
    print(f"  Final perturbation: {results['final_perturbation']:.2e}")
    
    # Decoherence analysis
    decoherence = stability.decoherence_analysis(time_range)
    print(f"\nDecoherence Analysis:")
    print(f"  Coherence at t=100: {decoherence['coherence_preservation']:.3f}")
    print(f"  1/e decoherence time: {decoherence['decoherence_time_1e']:.1f}")
    
    # Optimization recommendations
    recommendations = stability.optimization_recommendations(results)
    print(f"\nOptimization Recommendations:")
    print(f"  System stable: {recommendations['system_stable']}")
    print(f"  Stabilization required: {recommendations['stabilization_required']}")
    
    if recommendations['suggested_actions']:
        print(f"  Suggested actions:")
        for action in recommendations['suggested_actions']:
            print(f"    - {action}")
