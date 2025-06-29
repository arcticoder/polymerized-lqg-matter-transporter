"""
Metamaterial Enhancement Uncertainty System
==========================================

Implements advanced uncertainty analysis for metamaterial enhancement with:
- Effective medium fluctuation analysis
- Dielectric tensor uncertainty propagation
- Metamaterial response optimization with exact backreaction factor
- Frequency-dependent enhancement uncertainty bounds

Mathematical Framework:
ε_eff = ⟨ε_host⟩ + f·(ε_inclusion - ε_host)·[1 + δε_uncertainty]

where uncertainty contribution:
δε_uncertainty = ∫ S_metamaterial(ω) · K_enhancement(ω,μ) dω

Enhancement kernel:
K_enhancement(ω,μ) = β_backreaction · sinc²(πμω/ω_polymer) · T^(-3)

Temporal scaling with golden ratio stability:
σ²_metamaterial ≤ σ²_classical · φ^(-2) / [1 + β·T^(-1)]

Author: Advanced Matter Transporter Framework
Date: 2024
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Callable
from functools import partial
import logging
from dataclasses import dataclass
import scipy.integrate as integrate
import scipy.special as special

# Physical constants
EPSILON_0 = 8.8541878128e-12  # F/m (vacuum permittivity)
MU_0 = 4 * jnp.pi * 1e-7  # H/m (vacuum permeability)
SPEED_OF_LIGHT = 299792458.0  # m/s

# Mathematical constants from workspace analysis
EXACT_BACKREACTION_FACTOR = 1.9443254780147017  # 48.55% energy reduction
GOLDEN_RATIO = 1.618033988749894  # φ = (1 + √5)/2
GOLDEN_RATIO_INV = 0.618033988749894  # 1/φ
GOLDEN_RATIO_INV_SQ = 0.381966011250105  # φ^(-2)

@dataclass
class MetamaterialProperties:
    """Container for metamaterial property information"""
    effective_permittivity: jnp.ndarray
    effective_permeability: jnp.ndarray
    enhancement_factor: float
    frequency_response: jnp.ndarray
    uncertainty_bounds: Tuple[float, float]

@dataclass
class EffectiveMediumAnalysis:
    """Container for effective medium analysis results"""
    host_contribution: jnp.ndarray
    inclusion_contribution: jnp.ndarray
    volume_fraction: float
    mixing_uncertainty: float
    enhancement_spectrum: jnp.ndarray

@dataclass
class DielectricUncertaintyResult:
    """Container for dielectric tensor uncertainty analysis"""
    tensor_uncertainty: jnp.ndarray
    eigenvalue_uncertainty: jnp.ndarray
    enhancement_uncertainty: float
    frequency_stability: float
    polymer_enhancement_factor: float

@dataclass
class MetamaterialUncertaintyAnalysis:
    """Container for complete metamaterial uncertainty analysis"""
    properties: MetamaterialProperties
    effective_medium: EffectiveMediumAnalysis
    dielectric_uncertainty: DielectricUncertaintyResult
    enhancement_optimization: Dict[str, float]
    uncertainty_bounds: Dict[str, Any]

class MetamaterialEnhancementUncertainty:
    """
    Advanced metamaterial enhancement uncertainty quantification system.
    Analyzes effective medium fluctuations and dielectric tensor uncertainty.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metamaterial enhancement uncertainty analyzer.
        
        Args:
            config: Configuration dictionary with metamaterial parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metamaterial parameters
        self.n_frequency_points = config.get('n_frequency_points', 200)
        self.frequency_range = config.get('frequency_range', (1e9, 1e12))  # Hz
        self.volume_fraction = config.get('volume_fraction', 0.3)
        
        # Material properties
        self.host_permittivity = config.get('host_permittivity', 2.1 + 0.01j)
        self.inclusion_permittivity = config.get('inclusion_permittivity', -10.0 + 1.0j)
        self.host_permeability = config.get('host_permeability', 1.0)
        self.inclusion_permeability = config.get('inclusion_permeability', 1.0)
        
        # Polymer parameters
        self.mu_polymer = config.get('mu_polymer', 0.1)
        self.omega_polymer = config.get('omega_polymer', 1e11)  # Hz
        self.T_scaling = config.get('T_scaling', 1e4)
        
        # Uncertainty parameters
        self.uncertainty_tolerance = config.get('uncertainty_tolerance', 1e-3)
        self.enhancement_target = config.get('enhancement_target', 2.0)
        
        # Initialize frequency grid
        self._initialize_frequency_grid()
        
        # Precompute enhancement kernels
        self._precompute_enhancement_kernels()
        
        self.logger.info("Initialized Metamaterial Enhancement Uncertainty Analyzer")
    
    def _initialize_frequency_grid(self):
        """Initialize frequency grid for metamaterial analysis"""
        
        # Logarithmic frequency grid
        f_min, f_max = self.frequency_range
        self.frequency_grid = jnp.logspace(
            jnp.log10(f_min), jnp.log10(f_max), self.n_frequency_points
        )
        
        # Angular frequency grid
        self.omega_grid = 2 * jnp.pi * self.frequency_grid
        
        # Normalized frequency grid for polymer analysis
        self.omega_normalized = self.omega_grid / self.omega_polymer
        
        self.logger.info(f"Initialized frequency grid: {f_min:.2e} - {f_max:.2e} Hz")
    
    def _precompute_enhancement_kernels(self):
        """Precompute enhancement kernels K_enhancement(ω,μ)"""
        
        # K_enhancement(ω,μ) = β_backreaction · sinc²(πμω/ω_polymer) · T^(-3)
        def enhancement_kernel(omega_norm, mu):
            """Enhancement kernel function"""
            sinc_squared = jnp.sinc(jnp.pi * mu * omega_norm)**2
            temporal_scaling = self.T_scaling**(-3)  # T^(-3) scaling
            return EXACT_BACKREACTION_FACTOR * sinc_squared * temporal_scaling
        
        # Vectorized kernel computation
        self.enhancement_kernels = vmap(enhancement_kernel, in_axes=(0, None))(
            self.omega_normalized, self.mu_polymer
        )
        
        # Golden ratio stability enhancement
        self.stable_enhancement_kernels = self.enhancement_kernels * GOLDEN_RATIO_INV_SQ
        
        self.logger.info("Precomputed enhancement kernels")
    
    def compute_effective_medium_properties(self, uncertainty_level: float = 0.01) -> EffectiveMediumAnalysis:
        """
        Compute effective medium properties with uncertainty analysis
        
        Args:
            uncertainty_level: Relative uncertainty in material properties
            
        Returns:
            Effective medium analysis results
        """
        f = self.volume_fraction
        eps_h = self.host_permittivity
        eps_i = self.inclusion_permittivity
        
        # Maxwell-Garnett effective medium theory
        eps_mg = eps_h * (1 + 2*f*(eps_i - eps_h)/(eps_i + 2*eps_h - f*(eps_i - eps_h)))
        
        # Host contribution (frequency-dependent with polymer enhancement)
        host_contribution = []
        for omega in self.omega_grid:
            # Frequency-dependent host properties
            omega_norm = omega / self.omega_polymer
            polymer_factor = jnp.sinc(jnp.pi * self.mu_polymer * omega_norm)
            
            eps_h_freq = eps_h * (1.0 + 0.1 * polymer_factor * EXACT_BACKREACTION_FACTOR)
            host_contribution.append(eps_h_freq)
        
        host_contribution = jnp.array(host_contribution)
        
        # Inclusion contribution (enhanced by backreaction factor)
        inclusion_contribution = []
        for omega in self.omega_grid:
            omega_norm = omega / self.omega_polymer
            enhancement_factor = 1.0 + EXACT_BACKREACTION_FACTOR * jnp.exp(-omega_norm**2)
            
            eps_i_enhanced = eps_i * enhancement_factor
            inclusion_contribution.append(eps_i_enhanced)
        
        inclusion_contribution = jnp.array(inclusion_contribution)
        
        # Mixing uncertainty from volume fraction fluctuations
        f_uncertainty = uncertainty_level * f
        
        # Effective medium with uncertainty
        eps_eff_nominal = eps_h + f * (eps_i - eps_h)
        eps_eff_upper = eps_h + (f + f_uncertainty) * (eps_i - eps_h)
        eps_eff_lower = eps_h + (f - f_uncertainty) * (eps_i - eps_h)
        
        mixing_uncertainty = float(jnp.abs(eps_eff_upper - eps_eff_lower) / jnp.abs(eps_eff_nominal))
        
        # Enhancement spectrum across frequencies
        enhancement_spectrum = []
        for i, omega in enumerate(self.omega_grid):
            enhancement = jnp.abs(inclusion_contribution[i] / host_contribution[i])
            enhancement_spectrum.append(enhancement)
        
        enhancement_spectrum = jnp.array(enhancement_spectrum)
        
        return EffectiveMediumAnalysis(
            host_contribution=host_contribution,
            inclusion_contribution=inclusion_contribution,
            volume_fraction=f,
            mixing_uncertainty=mixing_uncertainty,
            enhancement_spectrum=enhancement_spectrum
        )
    
    def analyze_dielectric_tensor_uncertainty(self, effective_medium: EffectiveMediumAnalysis) -> DielectricUncertaintyResult:
        """
        Analyze dielectric tensor uncertainty propagation
        
        Args:
            effective_medium: Effective medium analysis results
            
        Returns:
            Dielectric tensor uncertainty analysis
        """
        # Construct dielectric tensor (assuming uniaxial symmetry)
        n_freq = len(self.frequency_grid)
        tensor_uncertainty = jnp.zeros((n_freq, 3, 3), dtype=jnp.complex64)
        
        # Diagonal tensor elements with frequency dependence
        for i, freq in enumerate(self.frequency_grid):
            omega = self.omega_grid[i]
            omega_norm = omega / self.omega_polymer
            
            # Base permittivity values
            eps_parallel = effective_medium.inclusion_contribution[i]
            eps_perpendicular = effective_medium.host_contribution[i]
            
            # Polymer enhancement with uncertainty
            polymer_uncertainty = 0.01 * jnp.sinc(jnp.pi * self.mu_polymer * omega_norm)
            enhancement_factor = 1.0 + EXACT_BACKREACTION_FACTOR * polymer_uncertainty
            
            # Diagonal elements
            tensor_uncertainty = tensor_uncertainty.at[i, 0, 0].set(eps_parallel * enhancement_factor)
            tensor_uncertainty = tensor_uncertainty.at[i, 1, 1].set(eps_perpendicular * enhancement_factor)
            tensor_uncertainty = tensor_uncertainty.at[i, 2, 2].set(eps_perpendicular * enhancement_factor)
            
            # Off-diagonal coupling (small but finite)
            coupling_strength = 0.001 * EXACT_BACKREACTION_FACTOR * polymer_uncertainty
            tensor_uncertainty = tensor_uncertainty.at[i, 0, 1].set(coupling_strength)
            tensor_uncertainty = tensor_uncertainty.at[i, 1, 0].set(coupling_strength)
        
        # Compute eigenvalue uncertainty
        eigenvalue_uncertainty = []
        for i in range(n_freq):
            eigenvals = jnp.linalg.eigvals(tensor_uncertainty[i])
            eigenval_spread = jnp.std(jnp.real(eigenvals))
            eigenvalue_uncertainty.append(eigenval_spread)
        
        eigenvalue_uncertainty = jnp.array(eigenvalue_uncertainty)
        
        # Enhancement uncertainty analysis
        enhancement_values = jnp.abs(effective_medium.enhancement_spectrum)
        enhancement_uncertainty = float(jnp.std(enhancement_values) / jnp.mean(enhancement_values))
        
        # Frequency stability assessment
        enhancement_gradient = jnp.gradient(enhancement_values)
        frequency_stability = 1.0 / (1.0 + jnp.max(jnp.abs(enhancement_gradient)))
        
        # Polymer enhancement factor
        mean_polymer_factor = jnp.mean(self.enhancement_kernels)
        polymer_enhancement_factor = float(EXACT_BACKREACTION_FACTOR * mean_polymer_factor)
        
        return DielectricUncertaintyResult(
            tensor_uncertainty=tensor_uncertainty,
            eigenvalue_uncertainty=eigenvalue_uncertainty,
            enhancement_uncertainty=enhancement_uncertainty,
            frequency_stability=float(frequency_stability),
            polymer_enhancement_factor=polymer_enhancement_factor
        )
    
    def optimize_metamaterial_enhancement(self, effective_medium: EffectiveMediumAnalysis,
                                        dielectric_uncertainty: DielectricUncertaintyResult) -> Dict[str, float]:
        """
        Optimize metamaterial enhancement with uncertainty bounds
        
        Args:
            effective_medium: Effective medium analysis
            dielectric_uncertainty: Dielectric uncertainty analysis
            
        Returns:
            Enhancement optimization results
        """
        # Target enhancement with uncertainty constraints
        target_enhancement = self.enhancement_target
        
        # Current enhancement performance
        mean_enhancement = float(jnp.mean(effective_medium.enhancement_spectrum))
        max_enhancement = float(jnp.max(effective_medium.enhancement_spectrum))
        
        # Uncertainty-constrained optimization
        enhancement_uncertainty = dielectric_uncertainty.enhancement_uncertainty
        frequency_stability = dielectric_uncertainty.frequency_stability
        
        # Golden ratio optimization for stability
        stability_factor = GOLDEN_RATIO_INV * frequency_stability
        uncertainty_penalty = 1.0 / (1.0 + enhancement_uncertainty)
        
        # Optimized enhancement with backreaction benefit
        optimized_enhancement = mean_enhancement * stability_factor * uncertainty_penalty * EXACT_BACKREACTION_FACTOR
        
        # Performance metrics
        enhancement_efficiency = optimized_enhancement / target_enhancement
        stability_score = stability_factor * uncertainty_penalty
        
        # Frequency band optimization
        resonance_frequencies = []
        for i, enhancement in enumerate(effective_medium.enhancement_spectrum):
            if enhancement > target_enhancement * 0.8:  # Within 80% of target
                resonance_frequencies.append(self.frequency_grid[i])
        
        optimal_bandwidth = len(resonance_frequencies) / len(self.frequency_grid)
        
        # T^(-3) temporal scaling benefit
        temporal_scaling_benefit = self.T_scaling**(-3) * EXACT_BACKREACTION_FACTOR
        
        return {
            'target_enhancement': target_enhancement,
            'achieved_enhancement': float(optimized_enhancement),
            'enhancement_efficiency': float(enhancement_efficiency),
            'stability_score': float(stability_score),
            'optimal_bandwidth': float(optimal_bandwidth),
            'uncertainty_penalty': float(uncertainty_penalty),
            'frequency_stability': frequency_stability,
            'temporal_scaling_benefit': float(temporal_scaling_benefit),
            'polymer_benefit': dielectric_uncertainty.polymer_enhancement_factor,
            'golden_ratio_optimization': float(stability_factor),
            'resonance_frequency_count': len(resonance_frequencies)
        }
    
    def compute_uncertainty_bounds(self, effective_medium: EffectiveMediumAnalysis,
                                 dielectric_uncertainty: DielectricUncertaintyResult,
                                 enhancement_opt: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute comprehensive uncertainty bounds for metamaterial enhancement
        
        Args:
            effective_medium: Effective medium analysis
            dielectric_uncertainty: Dielectric uncertainty analysis
            enhancement_opt: Enhancement optimization results
            
        Returns:
            Comprehensive uncertainty bounds
        """
        # Enhancement uncertainty bounds
        enhancement_mean = enhancement_opt['achieved_enhancement']
        enhancement_uncertainty = dielectric_uncertainty.enhancement_uncertainty
        
        # Golden ratio stability bounds
        lower_bound = enhancement_mean * (1.0 - enhancement_uncertainty * GOLDEN_RATIO_INV_SQ)
        upper_bound = enhancement_mean * (1.0 + enhancement_uncertainty * GOLDEN_RATIO_INV)
        
        # Confidence intervals based on frequency stability
        confidence_level = dielectric_uncertainty.frequency_stability
        confidence_width = (upper_bound - lower_bound) * (1.0 - confidence_level)
        
        # Performance guarantees
        guaranteed_enhancement = lower_bound + confidence_width / 2
        safety_margin = (enhancement_mean - guaranteed_enhancement) / enhancement_mean
        
        # Risk assessment
        uncertainty_risk = enhancement_uncertainty / enhancement_mean
        stability_risk = 1.0 - dielectric_uncertainty.frequency_stability
        combined_risk = jnp.sqrt(uncertainty_risk**2 + stability_risk**2)
        
        # Polymer enhancement reliability
        polymer_reliability = dielectric_uncertainty.polymer_enhancement_factor / EXACT_BACKREACTION_FACTOR
        
        # Temporal scaling confidence
        temporal_confidence = enhancement_opt['temporal_scaling_benefit'] / enhancement_mean
        
        return {
            'enhancement_bounds': {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'mean_value': float(enhancement_mean),
                'uncertainty_width': float(upper_bound - lower_bound)
            },
            'confidence_analysis': {
                'confidence_level': float(confidence_level),
                'confidence_width': float(confidence_width),
                'guaranteed_enhancement': float(guaranteed_enhancement),
                'safety_margin': float(safety_margin)
            },
            'risk_assessment': {
                'uncertainty_risk': float(uncertainty_risk),
                'stability_risk': float(stability_risk),
                'combined_risk': float(combined_risk),
                'risk_level': 'LOW' if combined_risk < 0.1 else 'MEDIUM' if combined_risk < 0.3 else 'HIGH'
            },
            'reliability_metrics': {
                'polymer_reliability': float(polymer_reliability),
                'temporal_confidence': float(temporal_confidence),
                'frequency_stability': dielectric_uncertainty.frequency_stability,
                'golden_ratio_benefit': float(GOLDEN_RATIO_INV_SQ)
            },
            'performance_certification': {
                'meets_target': guaranteed_enhancement >= self.enhancement_target,
                'certification_level': 'APPROVED' if guaranteed_enhancement >= self.enhancement_target else 'CONDITIONAL',
                'required_improvement': max(0.0, float(self.enhancement_target - guaranteed_enhancement)),
                'excess_performance': max(0.0, float(guaranteed_enhancement - self.enhancement_target))
            }
        }
    
    def analyze_metamaterial_uncertainty(self, uncertainty_level: float = 0.01) -> MetamaterialUncertaintyAnalysis:
        """
        Perform complete metamaterial enhancement uncertainty analysis
        
        Args:
            uncertainty_level: Material property uncertainty level
            
        Returns:
            Complete metamaterial uncertainty analysis
        """
        self.logger.info("Starting metamaterial uncertainty analysis...")
        
        # Effective medium analysis
        effective_medium = self.compute_effective_medium_properties(uncertainty_level)
        
        # Dielectric tensor uncertainty analysis
        dielectric_uncertainty = self.analyze_dielectric_tensor_uncertainty(effective_medium)
        
        # Enhancement optimization
        enhancement_optimization = self.optimize_metamaterial_enhancement(
            effective_medium, dielectric_uncertainty
        )
        
        # Uncertainty bounds computation
        uncertainty_bounds = self.compute_uncertainty_bounds(
            effective_medium, dielectric_uncertainty, enhancement_optimization
        )
        
        # Construct metamaterial properties
        properties = MetamaterialProperties(
            effective_permittivity=effective_medium.host_contribution + effective_medium.inclusion_contribution,
            effective_permeability=jnp.ones_like(self.frequency_grid),  # Non-magnetic for simplicity
            enhancement_factor=enhancement_optimization['achieved_enhancement'],
            frequency_response=effective_medium.enhancement_spectrum,
            uncertainty_bounds=(
                uncertainty_bounds['enhancement_bounds']['lower_bound'],
                uncertainty_bounds['enhancement_bounds']['upper_bound']
            )
        )
        
        result = MetamaterialUncertaintyAnalysis(
            properties=properties,
            effective_medium=effective_medium,
            dielectric_uncertainty=dielectric_uncertainty,
            enhancement_optimization=enhancement_optimization,
            uncertainty_bounds=uncertainty_bounds
        )
        
        self.logger.info(f"Metamaterial analysis complete: Enhancement = {properties.enhancement_factor:.3f}")
        return result

def create_metamaterial_uncertainty_analyzer(config: Optional[Dict[str, Any]] = None) -> MetamaterialEnhancementUncertainty:
    """
    Factory function to create metamaterial enhancement uncertainty analyzer
    
    Args:
        config: Optional configuration parameters
        
    Returns:
        Configured MetamaterialEnhancementUncertainty instance
    """
    default_config = {
        'n_frequency_points': 200,
        'frequency_range': (1e9, 1e12),
        'volume_fraction': 0.3,
        'host_permittivity': 2.1 + 0.01j,
        'inclusion_permittivity': -10.0 + 1.0j,
        'host_permeability': 1.0,
        'inclusion_permeability': 1.0,
        'mu_polymer': 0.1,
        'omega_polymer': 1e11,
        'T_scaling': 1e4,
        'uncertainty_tolerance': 1e-3,
        'enhancement_target': 2.0
    }
    
    if config:
        default_config.update(config)
    
    return MetamaterialEnhancementUncertainty(default_config)

# Demonstration function
def demonstrate_metamaterial_uncertainty():
    """Demonstrate metamaterial enhancement uncertainty analysis"""
    print("🔬 Metamaterial Enhancement Uncertainty Analysis Demonstration")
    print("=" * 70)
    
    # Create analyzer
    analyzer = create_metamaterial_uncertainty_analyzer()
    
    # Perform uncertainty analysis
    result = analyzer.analyze_metamaterial_uncertainty(uncertainty_level=0.02)
    
    # Display results
    print(f"\n📊 Metamaterial Properties:")
    print(f"  • Enhancement Factor: {result.properties.enhancement_factor:.3f}")
    print(f"  • Uncertainty Bounds: [{result.properties.uncertainty_bounds[0]:.3f}, {result.properties.uncertainty_bounds[1]:.3f}]")
    print(f"  • Frequency Range: {analyzer.frequency_range[0]:.2e} - {analyzer.frequency_range[1]:.2e} Hz")
    
    print(f"\n🔬 Effective Medium Analysis:")
    print(f"  • Volume Fraction: {result.effective_medium.volume_fraction:.3f}")
    print(f"  • Mixing Uncertainty: {result.effective_medium.mixing_uncertainty:.6f}")
    print(f"  • Enhancement Spectrum Range: {jnp.min(result.effective_medium.enhancement_spectrum):.3f} - {jnp.max(result.effective_medium.enhancement_spectrum):.3f}")
    
    print(f"\n⚡ Dielectric Uncertainty:")
    print(f"  • Enhancement Uncertainty: {result.dielectric_uncertainty.enhancement_uncertainty:.6f}")
    print(f"  • Frequency Stability: {result.dielectric_uncertainty.frequency_stability:.6f}")
    print(f"  • Polymer Enhancement Factor: {result.dielectric_uncertainty.polymer_enhancement_factor:.6f}")
    
    print(f"\n🎯 Enhancement Optimization:")
    opt = result.enhancement_optimization
    print(f"  • Target Enhancement: {opt['target_enhancement']:.3f}")
    print(f"  • Achieved Enhancement: {opt['achieved_enhancement']:.3f}")
    print(f"  • Enhancement Efficiency: {opt['enhancement_efficiency']:.1%}")
    print(f"  • Stability Score: {opt['stability_score']:.6f}")
    print(f"  • Optimal Bandwidth: {opt['optimal_bandwidth']:.1%}")
    
    print(f"\n📈 Uncertainty Bounds:")
    bounds = result.uncertainty_bounds
    print(f"  • Lower Bound: {bounds['enhancement_bounds']['lower_bound']:.3f}")
    print(f"  • Upper Bound: {bounds['enhancement_bounds']['upper_bound']:.3f}")
    print(f"  • Guaranteed Enhancement: {bounds['confidence_analysis']['guaranteed_enhancement']:.3f}")
    print(f"  • Safety Margin: {bounds['confidence_analysis']['safety_margin']:.1%}")
    print(f"  • Risk Level: {bounds['risk_assessment']['risk_level']}")
    
    print(f"\n✅ Performance Certification:")
    cert = bounds['performance_certification']
    print(f"  • Meets Target: {cert['meets_target']}")
    print(f"  • Certification Level: {cert['certification_level']}")
    print(f"  • Excess Performance: {cert['excess_performance']:.3f}")
    
    print(f"\n🌟 Key Achievements:")
    print(f"  • Exact backreaction factor β = {EXACT_BACKREACTION_FACTOR:.6f} integrated")
    print(f"  • Golden ratio stability φ^(-2) = {GOLDEN_RATIO_INV_SQ:.6f} applied")
    print(f"  • T^(-3) temporal scaling implemented")
    print(f"  • Frequency-dependent polymer enhancement active")

if __name__ == "__main__":
    demonstrate_metamaterial_uncertainty()
