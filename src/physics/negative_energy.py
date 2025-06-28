"""
Negative Energy Generation Module for Enhanced Stargate Transporter

This module implements active exotic-matter generators based on:
- Casimir effect between parallel plates
- Dynamic Casimir effect with moving boundaries  
- Squeezed vacuum states
- Integration with Van den Broeck geometry

Mathematical Framework:
    Casimir energy density: ρ_C(a) = -π²ℏc/(720a⁴)
    Reduction factor: R_casimir = ∫ρ_C(a(ρ,z))dV / (|min ρ_C| × V_neck)

Author: Enhanced Implementation  
Created: June 27, 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import scipy.special
import scipy.integrate

@dataclass
class CasimirConfig:
    """Configuration for Casimir exotic energy generator."""
    
    # Parallel plate setup
    plate_separation: float = 1e-6     # m (1 micrometer)
    plate_area: float = 1e-2           # m² (1 cm²)
    num_plates: int = 100              # Number of plate pairs
    
    # Dynamic parameters
    enable_dynamic_casimir: bool = True
    oscillation_frequency: float = 1e9  # Hz (GHz oscillations)
    oscillation_amplitude: float = 1e-8 # m (10 nm amplitude)
    
    # Vacuum squeezing
    enable_vacuum_squeezing: bool = True
    squeezing_parameter: float = 2.0    # Squeezing strength
    
    # Integration parameters
    spatial_resolution: int = 50
    temporal_resolution: int = 100

@dataclass 
class SqueezeVacuumConfig:
    """Configuration for squeezed vacuum state generation."""
    
    # Squeezing parameters
    squeezing_strength: float = 20.0    # dB
    squeezing_angle: float = 0.0        # radians
    
    # Cavity parameters
    cavity_length: float = 1e-3         # m (1 mm)
    finesse: float = 1000               # Cavity finesse
    
    # Pump parameters
    pump_power: float = 1.0             # W
    pump_frequency: float = 1e14        # Hz (optical)

class CasimirGenerator:
    """
    Active exotic matter generator using Casimir effects.
    
    Implements:
    - Static parallel-plate Casimir effect
    - Dynamic Casimir effect with moving boundaries
    - Integration with transporter geometry
    - Multi-plate enhancement arrays
    """
    
    def __init__(self, config: CasimirConfig):
        """Initialize Casimir generator with configuration."""
        self.config = config
        
        # Physical constants
        self.hbar = 1.055e-34          # J⋅s (reduced Planck constant)
        self.c = 299792458.0           # m/s (speed of light)
        self.epsilon_0 = 8.854e-12     # F/m (vacuum permittivity)
        
        # Pre-compute base parameters
        self.base_casimir_coefficient = -np.pi**2 * self.hbar * self.c / 720
        
        print(f"CasimirGenerator initialized:")
        print(f"  Plate separation: {config.plate_separation*1e6:.1f} μm")
        print(f"  Plate area: {config.plate_area*1e4:.1f} cm²")
        print(f"  Number of plates: {config.num_plates}")
        print(f"  Dynamic Casimir: {'Enabled' if config.enable_dynamic_casimir else 'Disabled'}")
        print(f"  Vacuum squeezing: {'Enabled' if config.enable_vacuum_squeezing else 'Disabled'}")
    
    def static_casimir_density(self, plate_separation: float) -> float:
        """
        Calculate static Casimir energy density between parallel plates.
        
        Formula: ρ_Casimir(a) = -π²ℏc/(720a⁴)
        
        Args:
            plate_separation: Distance between plates (m)
            
        Returns:
            Casimir energy density (J/m³, negative)
        """
        # Use standalone function for better numerical stability
        def casimir_formula(a, hbar_c_coeff):
            a = jnp.maximum(a, 1e-12)  # Ensure non-zero separation
            return hbar_c_coeff / (a**4)
        
        # Calculate with JAX for precision
        a_jax = jnp.array(plate_separation)
        coeff = -np.pi**2 * self.hbar * self.c / 720
        
        density = float(casimir_formula(a_jax, coeff))
        
        return density
    
    def dynamic_casimir_density(self, plate_separation: float, 
                              velocity: float, time: float) -> float:
        """
        Calculate dynamic Casimir energy density with moving boundaries.
        
        Args:
            plate_separation: Base plate separation (m)
            velocity: Boundary velocity (m/s)
            time: Time coordinate (s)
            
        Returns:
            Enhanced Casimir energy density including dynamic effects
        """
        if not self.config.enable_dynamic_casimir:
            return self.static_casimir_density(plate_separation)
        
        # Time-dependent separation with oscillation
        omega = 2 * np.pi * self.config.oscillation_frequency
        a_t = plate_separation + self.config.oscillation_amplitude * np.sin(omega * time)
        
        # Base static density
        static_density = self.static_casimir_density(a_t)
        
        # Dynamic enhancement factor
        # Velocity-dependent correction (simplified model)
        v_factor = 1.0 + (velocity / self.c)**2
        
        # Oscillation enhancement (adiabatic approximation)
        oscillation_factor = 1.0 + 0.1 * np.sin(omega * time)  # 10% modulation
        
        return static_density * v_factor * oscillation_factor
    
    def multi_plate_array_density(self, position: jnp.ndarray, 
                                time: float = 0.0) -> float:
        """
        Calculate energy density from multi-plate Casimir array.
        
        Args:
            position: 3D position vector [x, y, z]
            time: Time coordinate
            
        Returns:
            Total Casimir energy density from array
        """
        x, y, z = position
        
        # Plate spacing in array
        plate_spacing = 2 * self.config.plate_separation
        
        # Find nearest plate pair
        plate_index = int(z / plate_spacing) % self.config.num_plates
        z_local = z - plate_index * plate_spacing
        
        # Local plate separation (can vary with position)
        local_separation = self.config.plate_separation * (1.0 + 0.1 * np.sin(2 * np.pi * x / 1e-3))
        
        # Calculate local density
        if self.config.enable_dynamic_casimir:
            # Dynamic velocity (simple model)
            velocity = self.config.oscillation_amplitude * 2 * np.pi * self.config.oscillation_frequency * np.cos(2 * np.pi * self.config.oscillation_frequency * time)
            density = self.dynamic_casimir_density(local_separation, velocity, time)
        else:
            density = self.static_casimir_density(local_separation)
        
        # Array enhancement factor (coherent addition)
        array_factor = np.sqrt(self.config.num_plates)  # Square root scaling
        
        return density * array_factor
    
    def spatial_casimir_profile(self, transporter_geometry: Dict) -> Dict:
        """
        Calculate spatial profile of Casimir energy in transporter geometry.
        
        Args:
            transporter_geometry: Transporter geometric parameters
            
        Returns:
            Spatial Casimir energy profile
        """
        R_neck = transporter_geometry.get('R_neck', 0.1)
        R_payload = transporter_geometry.get('R_payload', 2.0)
        L_corridor = transporter_geometry.get('L_corridor', 10.0)
        
        # Create spatial grid
        resolution = self.config.spatial_resolution
        rho_grid = np.linspace(0, R_payload, resolution)
        z_grid = np.linspace(0, L_corridor, resolution)
        
        RHO, Z = np.meshgrid(rho_grid, z_grid)
        
        # Calculate Casimir density at each point
        casimir_density = np.zeros_like(RHO)
        
        for i in range(resolution):
            for j in range(resolution):
                rho_ij = RHO[i, j]
                z_ij = Z[i, j]
                
                # Position in Cartesian coordinates
                position = jnp.array([rho_ij, 0.0, z_ij])
                
                # Local plate separation varies with geometry
                # Smaller separation in neck region for stronger effect
                if rho_ij <= R_neck:
                    # In neck region - small separation
                    local_separation = self.config.plate_separation
                elif rho_ij >= R_payload:
                    # Outside payload region - large separation (weak effect)
                    local_separation = 10 * self.config.plate_separation
                else:
                    # Transition region - linear interpolation
                    fraction = (rho_ij - R_neck) / (R_payload - R_neck)
                    local_separation = self.config.plate_separation * (1 + 9 * fraction)
                
                # Calculate density
                casimir_density[i, j] = self.static_casimir_density(local_separation)
        
        # Calculate integrated quantities
        total_volume = np.pi * R_neck**2 * L_corridor  # Neck volume
        total_energy = np.trapz(np.trapz(casimir_density * 2 * np.pi * RHO, rho_grid), z_grid)
        
        average_density = total_energy / total_volume if total_volume > 0 else 0.0
        peak_density = np.min(casimir_density)  # Most negative
        
        return {
            'spatial_grid': {'rho': rho_grid, 'z': z_grid, 'RHO': RHO, 'Z': Z},
            'casimir_density': casimir_density,
            'total_energy': total_energy,
            'average_density': average_density,
            'peak_density': peak_density,
            'total_volume': total_volume
        }
    
    def casimir_reduction_factor(self, neck_volume: float, 
                               reference_energy: float = None) -> float:
        """
        Calculate energy reduction factor from Casimir generation.
        
        Args:
            neck_volume: Volume of transporter neck region (m³)
            reference_energy: Reference energy scale (J)
            
        Returns:
            Casimir energy reduction factor
        """
        # Calculate Casimir energy in neck volume
        casimir_density = abs(self.static_casimir_density(self.config.plate_separation))
        casimir_energy = casimir_density * neck_volume
        
        # Array enhancement
        array_enhancement = np.sqrt(self.config.num_plates)
        total_casimir_energy = casimir_energy * array_enhancement
        
        # Reference energy (default to 75 kg rest mass)
        if reference_energy is None:
            reference_energy = 75.0 * self.c**2  # mc²
        
        # Reduction factor
        reduction_factor = total_casimir_energy / reference_energy
        
        print(f"Casimir Energy Analysis:")
        print(f"  Single plate energy: {casimir_energy:.2e} J")
        print(f"  Array enhancement: {array_enhancement:.1f}×")
        print(f"  Total Casimir energy: {total_casimir_energy:.2e} J")
        print(f"  Reference energy: {reference_energy:.2e} J")
        print(f"  Reduction factor: {reduction_factor:.2e}")
        
        return reduction_factor
    
    def temporal_casimir_evolution(self, duration: float = 1e-3,
                                 transporter_velocity: float = 0.0) -> Dict:
        """
        Calculate temporal evolution of Casimir energy production.
        
        Args:
            duration: Time duration for analysis (s)
            transporter_velocity: Transporter conveyor velocity (m/s)
            
        Returns:
            Time evolution of Casimir energy
        """
        if not self.config.enable_dynamic_casimir:
            print("Dynamic Casimir effect disabled - returning static analysis")
            return {'static_density': self.static_casimir_density(self.config.plate_separation)}
        
        # Time array
        time_steps = self.config.temporal_resolution
        times = np.linspace(0, duration, time_steps)
        
        # Evolution arrays
        densities = []
        energies = []
        enhancement_factors = []
        
        for t in times:
            # Dynamic density
            density = self.dynamic_casimir_density(
                self.config.plate_separation,
                transporter_velocity,
                t
            )
            densities.append(density)
            
            # Energy in reference volume (1 μm³)
            ref_volume = (1e-6)**3
            energy = abs(density) * ref_volume
            energies.append(energy)
            
            # Enhancement over static case
            static_density = self.static_casimir_density(self.config.plate_separation)
            enhancement = abs(density / static_density) if static_density != 0 else 1.0
            enhancement_factors.append(enhancement)
        
        # Statistics
        average_density = np.mean(densities)
        peak_density = np.min(densities)  # Most negative
        enhancement_mean = np.mean(enhancement_factors)
        enhancement_std = np.std(enhancement_factors)
        
        return {
            'times': times,
            'densities': densities,
            'energies': energies,
            'enhancement_factors': enhancement_factors,
            'statistics': {
                'average_density': average_density,
                'peak_density': peak_density,
                'enhancement_mean': enhancement_mean,
                'enhancement_std': enhancement_std,
                'duration': duration
            }
        }

class SqueezeVacuumGenerator:
    """
    Squeezed vacuum state generator for enhanced negative energy production.
    
    Implements squeezed coherent states for enhanced Casimir effect:
    |ψ⟩ = S(ξ)|α⟩ where S(ξ) = exp[½(ξ*a² - ξa†²)]
    """
    
    def __init__(self, config: SqueezeVacuumConfig):
        """Initialize squeezed vacuum generator."""
        self.config = config
        
        # Physical constants
        self.hbar = 1.055e-34
        self.c = 299792458.0
        
        # Derived parameters
        self.omega = 2 * np.pi * self.c / config.cavity_length  # Cavity frequency
        self.squeezing_factor = 10**(config.squeezing_strength / 20)  # dB to linear
        
        print(f"SqueezeVacuumGenerator initialized:")
        print(f"  Squeezing strength: {config.squeezing_strength:.1f} dB")
        print(f"  Cavity length: {config.cavity_length*1e3:.1f} mm")
        print(f"  Cavity frequency: {self.omega/(2*np.pi)*1e-12:.1f} THz")
    
    def squeezing_enhancement_factor(self, field_mode: int = 0) -> float:
        """
        Calculate enhancement factor from vacuum squeezing.
        
        Args:
            field_mode: Field mode number
            
        Returns:
            Enhancement factor for energy extraction
        """
        # Squeezing parameter
        r = np.log(self.squeezing_factor)  # Squeezing parameter
        theta = self.config.squeezing_angle
        
        # Quadrature variance reduction
        var_x = np.exp(-2 * r)  # Squeezed quadrature
        var_p = np.exp(2 * r)   # Anti-squeezed quadrature
        
        # Enhancement comes from reduced vacuum fluctuations
        # in one quadrature allowing more efficient energy extraction
        enhancement = 1.0 / var_x  # Inverse of squeezed variance
        
        return enhancement
    
    def integrated_vacuum_energy(self, volume: float) -> float:
        """
        Calculate total vacuum energy in squeezed state.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total vacuum energy (J)
        """
        # Vacuum energy density (simplified)
        omega = self.omega
        energy_density = self.hbar * omega / (2 * volume)  # Zero-point energy
        
        # Squeezing enhancement
        enhancement = self.squeezing_enhancement_factor()
        
        # Total energy
        total_energy = energy_density * volume * enhancement
        
        return total_energy

class IntegratedNegativeEnergySystem:
    """
    Integrated system combining Casimir generation with transporter geometry.
    """
    
    def __init__(self, casimir_config: CasimirConfig = None,
                 squeeze_config: SqueezeVacuumConfig = None):
        """Initialize integrated negative energy system."""
        
        # Default configurations
        self.casimir_config = casimir_config or CasimirConfig()
        self.squeeze_config = squeeze_config or SqueezeVacuumConfig()
        
        # Initialize generators
        self.casimir_gen = CasimirGenerator(self.casimir_config)
        self.squeeze_gen = SqueezeVacuumGenerator(self.squeeze_config)
        
        print(f"\nIntegratedNegativeEnergySystem initialized")
        print(f"  Casimir plates: {self.casimir_config.num_plates}")
        print(f"  Vacuum squeezing: {self.squeeze_config.squeezing_strength:.1f} dB")
    
    def total_reduction_factor(self, transporter_geometry: Dict) -> float:
        """
        Calculate total energy reduction from all negative energy sources.
        
        Args:
            transporter_geometry: Transporter geometric parameters
            
        Returns:
            Combined reduction factor
        """
        # Neck volume
        R_neck = transporter_geometry.get('R_neck', 0.1)
        L_corridor = transporter_geometry.get('L_corridor', 10.0)
        neck_volume = np.pi * R_neck**2 * L_corridor
        
        # Casimir reduction
        casimir_factor = self.casimir_gen.casimir_reduction_factor(neck_volume)
        
        # Vacuum squeezing enhancement
        squeeze_factor = self.squeeze_gen.squeezing_enhancement_factor()
        
        # Combined effect (multiplicative for independent sources)
        total_factor = casimir_factor * squeeze_factor
        
        print(f"\nTotal Negative Energy Analysis:")
        print(f"  Casimir reduction: {casimir_factor:.2e}")
        print(f"  Vacuum squeezing: {squeeze_factor:.2f}×")
        print(f"  Combined factor: {total_factor:.2e}")
        
        return total_factor
    
    def demonstrate_negative_energy_integration(self) -> Dict:
        """Demonstrate complete negative energy integration."""
        print("\n" + "="*70)
        print("INTEGRATED NEGATIVE ENERGY GENERATION DEMONSTRATION")
        print("="*70)
        
        # Test geometry (typical transporter)
        geometry = {
            'R_neck': 0.05,      # 5 cm
            'R_payload': 2.0,    # 2 m
            'L_corridor': 50.0   # 50 m
        }
        
        print(f"\nTest Geometry:")
        print(f"  Neck radius: {geometry['R_neck']:.2f} m")
        print(f"  Payload radius: {geometry['R_payload']:.1f} m")
        print(f"  Corridor length: {geometry['L_corridor']:.1f} m")
        
        # Casimir spatial analysis
        casimir_profile = self.casimir_gen.spatial_casimir_profile(geometry)
        
        print(f"\n📊 Casimir Spatial Analysis:")
        print(f"  Peak density: {casimir_profile['peak_density']:.2e} J/m³")
        print(f"  Average density: {casimir_profile['average_density']:.2e} J/m³")
        print(f"  Total energy: {casimir_profile['total_energy']:.2e} J")
        
        # Dynamic evolution
        dynamic_evolution = self.casimir_gen.temporal_casimir_evolution(
            duration=1e-6,  # 1 microsecond
            transporter_velocity=1e5  # 100 km/s conveyor
        )
        
        if 'statistics' in dynamic_evolution:
            stats = dynamic_evolution['statistics']
            print(f"\n⚡ Dynamic Casimir Analysis:")
            print(f"  Enhancement mean: {stats['enhancement_mean']:.2f}×")
            print(f"  Enhancement std: {stats['enhancement_std']:.2f}")
            print(f"  Peak density: {stats['peak_density']:.2e} J/m³")
        
        # Total reduction factor
        total_reduction = self.total_reduction_factor(geometry)
        
        # Energy comparison
        reference_energy = 75.0 * (299792458.0)**2  # 75 kg rest mass
        effective_energy = reference_energy * total_reduction
        
        print(f"\n🌟 FINAL ENERGY ANALYSIS:")
        print(f"  Reference energy: {reference_energy:.2e} J")
        print(f"  Total reduction: {total_reduction:.2e}")
        print(f"  Effective energy: {effective_energy:.2e} J")
        print(f"  Energy ratio: {effective_energy/reference_energy:.2e}")
        
        return {
            'geometry': geometry,
            'casimir_profile': casimir_profile,
            'dynamic_evolution': dynamic_evolution,
            'total_reduction': total_reduction,
            'energy_analysis': {
                'reference_energy': reference_energy,
                'effective_energy': effective_energy,
                'reduction_factor': total_reduction
            }
        }

def main():
    """Main demonstration of negative energy generation."""
    
    # Create integrated system
    casimir_config = CasimirConfig(
        plate_separation=1e-6,
        num_plates=200,
        enable_dynamic_casimir=True,
        oscillation_frequency=1e9
    )
    
    squeeze_config = SqueezeVacuumConfig(
        squeezing_strength=25.0,  # 25 dB squeezing
        cavity_length=1e-3
    )
    
    integrated_system = IntegratedNegativeEnergySystem(casimir_config, squeeze_config)
    
    # Run demonstration
    results = integrated_system.demonstrate_negative_energy_integration()
    
    print(f"\n✅ Negative Energy Generation Ready")
    print(f"   Total reduction factor: {results['total_reduction']:.2e}")
    print(f"   System ready for transporter integration")
    
    return integrated_system, results

if __name__ == "__main__":
    system, demo_results = main()
