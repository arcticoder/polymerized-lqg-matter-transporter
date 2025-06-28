#!/usr/bin/env python3
"""
Casimir Effect Module
====================

Implements negative energy generation through Casimir effect:
- Multi-plate Casimir energy density: ρ = -π²ℏc / (720 a⁴)
- Enhancement factor: √N for N plates
- Reduction factor: R_Casimir = (|ρ|·V_neck·√N) / (m c²)

The Casimir effect generates negative energy density between
conducting plates, which can be harnessed for matter transport
energy reduction when integrated into warp geometries.

Mathematical Foundation:
The Casimir energy arises from vacuum fluctuations of the
electromagnetic field between conducting boundaries, creating
regions of negative energy density that violate the null
energy condition locally while preserving global energy
conservation through quantum field theory.

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Physical constants
hbar = 1.0545718e-34  # Planck's constant (J⋅s)
c = 299792458.0     # Speed of light (m/s)
epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)

@dataclass
class CasimirConfig:
    """Configuration for Casimir effect calculations."""
    plate_separation: float = 1e-6    # Plate separation distance (m)
    num_plates: int = 100             # Number of plates in array
    V_neck: float = 1e-6              # Warp neck volume (m³)
    plate_area: float = 1e-4          # Individual plate area (m²)
    conductivity: float = 5.96e7      # Plate conductivity (S/m, copper default)
    temperature: float = 300.0        # Operating temperature (K)

class CasimirGenerator:
    """
    Multi-plate Casimir energy density generator.
    
    Implements the Casimir effect for negative energy generation:
    ρ = -π²ℏc / (720 a⁴) for parallel plate configuration
    
    The multi-plate enhancement provides √N scaling where N is
    the number of plates, enabling macroscopic energy densities.
    
    Parameters:
    -----------
    config : CasimirConfig
        Configuration parameters for Casimir array
    """
    
    def __init__(self, config: CasimirConfig):
        """
        Initialize Casimir generator with configuration.
        
        Args:
            config: Casimir configuration parameters
        """
        self.config = config
        self.a = config.plate_separation
        self.N = config.num_plates
        self.V_neck = config.V_neck
        
        # Validate parameters
        self._validate_parameters()
        
        # Precompute constants
        self._casimir_constant = -np.pi**2 * hbar * c / 720.0
    
    def _validate_parameters(self):
        """Validate Casimir configuration parameters."""
        if self.a <= 0:
            raise ValueError("Plate separation must be positive")
        if self.a > 1e-3:
            warnings.warn("Large plate separation may reduce Casimir effect strength")
        if self.N < 2:
            raise ValueError("Need at least 2 plates for Casimir effect")
        if self.V_neck <= 0:
            raise ValueError("Neck volume must be positive")
    
    def rho(self, a: Optional[float] = None) -> float:
        """
        Compute Casimir energy density.
        
        Formula: ρ_Casimir(a) = -π²ℏc / (720 a⁴)
        
        Args:
            a: Plate separation (uses default if None)
            
        Returns:
            Casimir energy density (J/m³), negative for attractive effect
        """
        separation = a if a is not None else self.a
        return self._casimir_constant / (separation**4)
    
    def force_per_area(self, a: Optional[float] = None) -> float:
        """
        Compute Casimir force per unit area.
        
        Formula: F/A = -π²ℏc / (240 a⁴)
        
        Args:
            a: Plate separation (uses default if None)
            
        Returns:
            Force per unit area (N/m²), negative for attractive force
        """
        separation = a if a is not None else self.a
        return -np.pi**2 * hbar * c / (240.0 * separation**4)
    
    def total_energy(self, a: Optional[float] = None) -> float:
        """
        Compute total Casimir energy in the gap.
        
        E = ρ × Volume_gap
        
        Args:
            a: Plate separation (uses default if None)
            
        Returns:
            Total Casimir energy (J)
        """
        separation = a if a is not None else self.a
        volume_gap = self.config.plate_area * separation
        return self.rho(separation) * volume_gap
    
    def multi_plate_enhancement(self) -> float:
        """
        Compute multi-plate enhancement factor.
        
        For N plates in series: Enhancement = √N
        This accounts for coherent addition of Casimir fields.
        
        Returns:
            Enhancement factor (dimensionless)
        """
        return np.sqrt(self.N)
    
    def R_casimir(self, m: float, a: Optional[float] = None) -> float:
        """
        Compute Casimir reduction factor for matter transport.
        
        Formula: R_Casimir = (√N × |ρ(a)| × V_neck) / (m c²)
        
        This represents the fractional energy reduction achievable
        through Casimir effect integration.
        
        Args:
            m: Payload mass (kg)
            a: Plate separation (uses default if None)
            
        Returns:
            Reduction factor (dimensionless, typically << 1)
        """
        if m <= 0:
            raise ValueError("Mass must be positive")
        
        rho_cas = abs(self.rho(a))
        enhancement = self.multi_plate_enhancement()
        rest_energy = m * c**2
        
        return (enhancement * rho_cas * self.V_neck) / rest_energy
    
    def dynamic_casimir_factor(self, frequency: float, amplitude: float) -> float:
        """
        Compute dynamic Casimir enhancement factor.
        
        When plates oscillate, additional negative energy can be extracted
        through parametric amplification of vacuum fluctuations.
        
        Args:
            frequency: Oscillation frequency (Hz)
            amplitude: Oscillation amplitude (m)
            
        Returns:
            Dynamic enhancement factor (≥ 1)
        """
        # Simplified model: enhancement scales with oscillation parameters
        # Full treatment requires detailed QFT calculation
        omega = 2 * np.pi * frequency
        characteristic_time = hbar / (self.rho(self.a) * self.a**3)
        
        # Dynamic enhancement when oscillation timescale ~ quantum timescale
        enhancement = 1.0 + (omega * characteristic_time) * (amplitude / self.a)**2
        
        return max(1.0, enhancement)  # Ensure enhancement ≥ 1
    
    def thermal_correction(self, temperature: Optional[float] = None) -> float:
        """
        Compute thermal correction factor.
        
        At finite temperature, thermal photons reduce Casimir effect.
        
        Args:
            temperature: Temperature in Kelvin (uses config default if None)
            
        Returns:
            Thermal correction factor (≤ 1)
        """
        T = temperature if temperature is not None else self.config.temperature
        
        # Thermal wavelength
        k_B = 1.380649e-23  # Boltzmann constant
        lambda_thermal = hbar * c / (k_B * T)
        
        # Correction factor when plate separation ~ thermal wavelength
        if self.a < lambda_thermal:
            return 1.0  # Negligible thermal effects
        else:
            return np.exp(-2 * np.pi * self.a / lambda_thermal)
    
    def casimir_stress_energy_tensor(self, coordinates: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute Casimir stress-energy tensor components.
        
        T_μν for integration into Einstein field equations.
        
        Args:
            coordinates: Spatial coordinates where to evaluate T_μν
            
        Returns:
            Dictionary with stress-energy tensor components
        """
        # For parallel plates: T₀₀ = ρ, T₁₁ = T₂₂ = -ρ/3, T₃₃ = ρ
        rho_cas = self.rho()
        
        shape = coordinates.shape if len(coordinates.shape) > 1 else (len(coordinates),)
        
        return {
            'T_00': np.full(shape, rho_cas / c**2),  # Energy density / c²
            'T_11': np.full(shape, -rho_cas / (3 * c**2)),  # Pressure components
            'T_22': np.full(shape, -rho_cas / (3 * c**2)),
            'T_33': np.full(shape, rho_cas / c**2)
        }
    
    def optimize_plate_separation(self, target_reduction: float, m: float) -> Dict[str, float]:
        """
        Optimize plate separation for target energy reduction.
        
        Args:
            target_reduction: Desired R_casimir value
            m: Payload mass
            
        Returns:
            Dictionary with optimization results
        """
        from scipy.optimize import minimize_scalar
        
        def objective(a):
            try:
                R = self.R_casimir(m, a)
                return abs(R - target_reduction)
            except:
                return 1e10  # Penalty for invalid parameters
        
        # Reasonable bounds for plate separation
        result = minimize_scalar(objective, bounds=(1e-9, 1e-3), method='bounded')
        
        optimal_a = result.x
        achieved_R = self.R_casimir(m, optimal_a)
        
        return {
            'optimal_separation': optimal_a,
            'achieved_reduction': achieved_R,
            'target_reduction': target_reduction,
            'optimization_success': result.success,
            'casimir_energy_density': self.rho(optimal_a),
            'total_casimir_energy': self.total_energy(optimal_a)
        }
    
    def performance_analysis(self, m: float) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive performance analysis of Casimir system.
        
        Args:
            m: Payload mass for analysis
            
        Returns:
            Complete performance metrics
        """
        base_reduction = self.R_casimir(m)
        thermal_factor = self.thermal_correction()
        enhancement = self.multi_plate_enhancement()
        
        # Dynamic enhancement example (1 kHz, 1% amplitude)
        dynamic_factor = self.dynamic_casimir_factor(1000.0, self.a * 0.01)
        
        # Total effective reduction
        total_reduction = base_reduction * thermal_factor * dynamic_factor
        
        return {
            'base_reduction_factor': base_reduction,
            'multi_plate_enhancement': enhancement,
            'thermal_correction': thermal_factor,
            'dynamic_enhancement': dynamic_factor,
            'total_reduction_factor': total_reduction,
            'energy_density': self.rho(),
            'force_per_area': self.force_per_area(),
            'total_energy': self.total_energy(),
            'configuration': {
                'plate_separation': self.a,
                'num_plates': self.N,
                'neck_volume': self.V_neck,
                'temperature': self.config.temperature
            }
        }

# Utility functions
def casimir_force_between_spheres(R1: float, R2: float, distance: float) -> float:
    """
    Compute Casimir force between two conducting spheres.
    
    Proximity force approximation for sphere-sphere geometry.
    """
    # Reduced radius
    R_eff = (R1 * R2) / (R1 + R2)
    
    # Force per unit area for parallel plates
    force_density = np.pi**2 * hbar * c / (240.0 * distance**4)
    
    # Geometric factor for spheres
    return 2 * np.pi * R_eff * force_density

def casimir_energy_scaling_laws(a_range: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze Casimir energy scaling across separation range.
    
    Args:
        a_range: Array of plate separations
        
    Returns:
        Scaling analysis results
    """
    config = CasimirConfig()
    generator = CasimirGenerator(config)
    
    energies = [generator.rho(a) for a in a_range]
    forces = [generator.force_per_area(a) for a in a_range]
    
    return {
        'separations': a_range,
        'energy_densities': np.array(energies),
        'forces_per_area': np.array(forces),
        'a4_scaling_check': np.array(energies) * (a_range**4),  # Should be constant
        'power_law_fit': np.polyfit(np.log(a_range), np.log(np.abs(energies)), 1)[0]  # Should be ≈ -4
    }

if __name__ == "__main__":
    # Demonstration of Casimir effect calculations
    print("Casimir Effect Demonstration")
    print("=" * 40)
    
    # Standard configuration
    config = CasimirConfig(
        plate_separation=1e-6,    # 1 μm
        num_plates=100,           # 100-plate array
        V_neck=1e-6,             # 1 μm³ neck volume
        plate_area=1e-4          # 1 cm² plates
    )
    
    casimir = CasimirGenerator(config)
    
    # Test with 1 kg payload
    m_test = 1.0
    analysis = casimir.performance_analysis(m_test)
    
    print(f"Configuration:")
    print(f"  Plate separation: {config.plate_separation*1e6:.1f} μm")
    print(f"  Number of plates: {config.num_plates}")
    print(f"  Neck volume: {config.V_neck*1e6:.1f} μm³")
    
    print(f"\nCasimir Performance (1 kg payload):")
    print(f"  Energy density: {analysis['energy_density']:.3e} J/m³")
    print(f"  Force per area: {analysis['force_per_area']:.3e} N/m²")
    print(f"  Base reduction: {analysis['base_reduction_factor']:.3e}")
    print(f"  Multi-plate enhancement: {analysis['multi_plate_enhancement']:.1f}×")
    print(f"  Total reduction factor: {analysis['total_reduction_factor']:.3e}")
    
    # Optimization example
    target = 1e-10
    opt_result = casimir.optimize_plate_separation(target, m_test)
    print(f"\nOptimization for R = {target:.1e}:")
    print(f"  Optimal separation: {opt_result['optimal_separation']*1e9:.1f} nm")
    print(f"  Achieved reduction: {opt_result['achieved_reduction']:.3e}")
