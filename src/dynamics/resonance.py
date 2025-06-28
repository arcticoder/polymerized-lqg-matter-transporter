#!/usr/bin/env python3
"""
Time-Dependent Resonance Module
==============================

Implements dynamic resonance effects for matter transport:
- v_s(t) = V_max * sin(π * t / T_period)
- Time-dependent field evolution
- Resonance optimization
- Dynamic stability analysis

This module provides time-dependent enhancements to matter transport
through resonant field coupling and dynamic optimization of transport
corridors with temporal variation.

Mathematical Foundation:
Time-dependent resonance arises from parametric coupling between
the transport field and oscillating boundary conditions. Optimal
resonance frequencies can enhance transport efficiency through
constructive interference effects in the field evolution.

Author: Enhanced Matter Transporter Framework
Date: June 28, 2025
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from functools import partial
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp

# Physical constants
c = 299792458.0     # Speed of light (m/s)
hbar = 1.0545718e-34  # Planck's constant (J⋅s)

@dataclass
class ResonanceConfig:
    """Configuration for time-dependent resonance."""
    v_max: float = 1e6                    # Maximum velocity (m/s)
    T_period: float = 3600.0              # Oscillation period (s)
    resonance_modes: List[int] = None     # Harmonic modes to include
    damping_coefficient: float = 0.01     # Damping parameter
    field_coupling: float = 1.0           # Field coupling strength
    
    def __post_init__(self):
        if self.resonance_modes is None:
            self.resonance_modes = [1, 2, 3]  # First three harmonics

class TimeDependentResonance:
    """
    Time-dependent resonance for enhanced matter transport.
    
    Implements sinusoidal velocity modulation:
    v_s(t) = V_max * sin(π * t / T_period)
    
    This creates resonant enhancement when the oscillation frequency
    matches natural field modes, leading to constructive interference
    and reduced energy requirements.
    
    Parameters:
    -----------
    config : ResonanceConfig
        Configuration for resonance parameters
    """
    
    def __init__(self, config: ResonanceConfig):
        """
        Initialize time-dependent resonance.
        
        Args:
            config: Resonance configuration
        """
        self.config = config
        self.v_max = config.v_max
        self.T_period = config.T_period
        self.omega = 2 * np.pi / config.T_period  # Angular frequency
        
        # Validate configuration
        self._validate_config()
        
        # Precompute resonance characteristics
        self._compute_resonance_properties()
    
    def _validate_config(self):
        """Validate resonance configuration."""
        if self.v_max >= c:
            raise ValueError(f"Maximum velocity {self.v_max} exceeds speed of light")
        if self.v_max > 0.1 * c:
            warnings.warn(f"High velocity {self.v_max/c:.1%}c may require relativistic treatment")
        if self.T_period <= 0:
            raise ValueError("Period must be positive")
        if self.config.damping_coefficient < 0:
            raise ValueError("Damping coefficient must be non-negative")
    
    def _compute_resonance_properties(self):
        """Compute derived resonance properties."""
        # Characteristic frequency
        self.f_characteristic = 1 / self.T_period
        
        # Velocity amplitude for each harmonic
        self.harmonic_amplitudes = {}
        for mode in self.config.resonance_modes:
            # Fourier coefficients for sinusoidal modulation
            if mode == 1:
                self.harmonic_amplitudes[mode] = self.v_max
            else:
                # Higher harmonics from nonlinear coupling
                self.harmonic_amplitudes[mode] = self.v_max / (mode**2)
    
    @partial(jit, static_argnums=(0,))
    def v_s(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute sinusoidal velocity profile.
        
        Formula: v_s(t) = V_max * sin(π * t / T_period)
        
        Args:
            t: Time array
            
        Returns:
            Velocity array
        """
        return self.v_max * jnp.sin(jnp.pi * t / self.T_period)
    
    @partial(jit, static_argnums=(0,))
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute acceleration profile.
        
        a(t) = dv/dt = (π V_max / T_period) * cos(π * t / T_period)
        
        Args:
            t: Time array
            
        Returns:
            Acceleration array
        """
        return (jnp.pi * self.v_max / self.T_period) * jnp.cos(jnp.pi * t / self.T_period)
    
    @partial(jit, static_argnums=(0,))
    def kinetic_energy(self, t: jnp.ndarray, mass: float) -> jnp.ndarray:
        """
        Compute time-dependent kinetic energy.
        
        KE(t) = (1/2) * m * v_s(t)²
        
        Args:
            t: Time array
            mass: Transport mass
            
        Returns:
            Kinetic energy array
        """
        v = self.v_s(t)
        return 0.5 * mass * v**2
    
    def resonance_enhancement_factor(self, field_frequency: float) -> float:
        """
        Compute resonance enhancement factor.
        
        Enhancement occurs when field frequency matches oscillation harmonics.
        
        Args:
            field_frequency: Natural field frequency (Hz)
            
        Returns:
            Enhancement factor (≥ 1)
        """
        enhancement = 1.0
        
        for mode in self.config.resonance_modes:
            mode_frequency = mode * self.f_characteristic
            
            # Lorentzian resonance profile
            gamma = self.config.damping_coefficient * mode_frequency
            resonance_strength = self.harmonic_amplitudes[mode] / self.v_max
            
            lorentzian = (gamma / (2 * np.pi)) / ((field_frequency - mode_frequency)**2 + (gamma/2)**2)
            enhancement += resonance_strength * lorentzian * self.config.field_coupling
        
        return enhancement
    
    def optimal_period_for_field(self, field_frequency: float) -> Dict[str, float]:
        """
        Find optimal oscillation period for given field frequency.
        
        Args:
            field_frequency: Target field frequency for resonance
            
        Returns:
            Optimization results
        """
        def objective(T_period):
            # Temporarily modify period for evaluation
            old_period = self.T_period
            old_omega = self.omega
            old_f = self.f_characteristic
            
            self.T_period = T_period
            self.omega = 2 * np.pi / T_period
            self.f_characteristic = 1 / T_period
            
            # Negative because we want to maximize enhancement
            enhancement = -self.resonance_enhancement_factor(field_frequency)
            
            # Restore original values
            self.T_period = old_period
            self.omega = old_omega
            self.f_characteristic = old_f
            
            return enhancement
        
        # Search reasonable period range
        period_bounds = (1.0, 86400.0)  # 1 second to 1 day
        
        result = minimize_scalar(objective, bounds=period_bounds, method='bounded')
        
        optimal_period = result.x
        max_enhancement = -result.fun
        
        return {
            'optimal_period': optimal_period,
            'max_enhancement': max_enhancement,
            'optimal_frequency': 1 / optimal_period,
            'optimization_success': result.success
        }
    
    def field_evolution_ode(self, t: float, y: np.ndarray, 
                           system_params: Dict) -> np.ndarray:
        """
        ODE for time-dependent field evolution.
        
        dy/dt = F(t, y) with resonant driving
        
        Args:
            t: Time
            y: Field state vector [amplitude, phase]
            system_params: System parameters
            
        Returns:
            Time derivative dy/dt
        """
        amplitude, phase = y
        
        # Current velocity and acceleration
        v_current = float(self.v_s(jnp.array([t])))
        a_current = float(self.acceleration(jnp.array([t])))
        
        # Field evolution equations with resonant coupling
        omega_field = system_params.get('field_frequency', 1.0)
        coupling = system_params.get('coupling_strength', 0.1)
        damping = system_params.get('field_damping', 0.01)
        
        # Amplitude evolution (with driving and damping)
        dA_dt = -damping * amplitude + coupling * v_current * np.cos(omega_field * t)
        
        # Phase evolution (with velocity-dependent frequency shift)
        dphi_dt = omega_field + coupling * a_current / amplitude if amplitude > 0 else omega_field
        
        return np.array([dA_dt, dphi_dt])
    
    def simulate_field_evolution(self, duration: float, system_params: Dict,
                                initial_conditions: Tuple[float, float] = (1.0, 0.0)) -> Dict:
        """
        Simulate time-dependent field evolution.
        
        Args:
            duration: Simulation duration (s)
            system_params: System parameters for ODE
            initial_conditions: Initial [amplitude, phase]
            
        Returns:
            Evolution results
        """
        # Time array
        n_points = max(100, int(duration / self.T_period * 50))  # ~50 points per period
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, n_points)
        
        # Solve ODE
        solution = solve_ivp(
            fun=self.field_evolution_ode,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            args=(system_params,),
            method='RK45',
            rtol=1e-8
        )
        
        if not solution.success:
            warnings.warn("Field evolution simulation failed")
            return {'success': False}
        
        # Extract results
        times = solution.t
        amplitudes = solution.y[0]
        phases = solution.y[1]
        
        # Compute derived quantities
        velocities = np.array([float(self.v_s(jnp.array([t]))) for t in times])
        accelerations = np.array([float(self.acceleration(jnp.array([t]))) for t in times])
        field_energy = 0.5 * amplitudes**2  # Simplified energy
        
        # Resonance analysis
        enhancement_factors = [
            self.resonance_enhancement_factor(system_params.get('field_frequency', 1.0))
            for _ in times
        ]
        
        return {
            'success': True,
            'times': times,
            'amplitudes': amplitudes,
            'phases': phases,
            'velocities': velocities,
            'accelerations': accelerations,
            'field_energy': field_energy,
            'enhancement_factors': enhancement_factors,
            'final_amplitude': amplitudes[-1],
            'mean_enhancement': np.mean(enhancement_factors),
            'energy_stability': np.std(field_energy) / np.mean(field_energy)
        }
    
    def resonance_sweep_analysis(self, frequency_range: np.ndarray) -> Dict:
        """
        Analyze resonance enhancement across frequency range.
        
        Args:
            frequency_range: Array of frequencies to analyze
            
        Returns:
            Resonance sweep results
        """
        enhancements = [self.resonance_enhancement_factor(f) for f in frequency_range]
        enhancements = np.array(enhancements)
        
        # Find resonance peaks
        peak_indices = []
        for i in range(1, len(enhancements) - 1):
            if (enhancements[i] > enhancements[i-1] and 
                enhancements[i] > enhancements[i+1] and 
                enhancements[i] > 1.1):  # Significant enhancement
                peak_indices.append(i)
        
        peak_frequencies = frequency_range[peak_indices]
        peak_enhancements = enhancements[peak_indices]
        
        return {
            'frequencies': frequency_range,
            'enhancements': enhancements,
            'peak_frequencies': peak_frequencies,
            'peak_enhancements': peak_enhancements,
            'max_enhancement': np.max(enhancements),
            'optimal_frequency': frequency_range[np.argmax(enhancements)],
            'resonance_bandwidth': self._compute_resonance_bandwidth(frequency_range, enhancements)
        }
    
    def _compute_resonance_bandwidth(self, frequencies: np.ndarray, 
                                   enhancements: np.ndarray) -> float:
        """Compute 3dB bandwidth of main resonance."""
        max_enhancement = np.max(enhancements)
        half_max = 1 + (max_enhancement - 1) / 2  # 3dB point
        
        # Find frequencies where enhancement > half_max
        above_half = frequencies[enhancements > half_max]
        
        if len(above_half) > 0:
            return np.max(above_half) - np.min(above_half)
        else:
            return 0.0
    
    def stability_analysis(self, perturbation_amplitude: float = 0.01) -> Dict:
        """
        Analyze stability to parameter perturbations.
        
        Args:
            perturbation_amplitude: Relative perturbation size
            
        Returns:
            Stability analysis results
        """
        base_enhancement = self.resonance_enhancement_factor(self.f_characteristic)
        
        # Perturb key parameters
        perturbations = {
            'v_max': self.v_max * (1 + perturbation_amplitude),
            'T_period': self.T_period * (1 + perturbation_amplitude),
            'damping': self.config.damping_coefficient * (1 + perturbation_amplitude)
        }
        
        sensitivities = {}
        
        for param, perturbed_value in perturbations.items():
            # Temporarily modify parameter
            if param == 'v_max':
                old_value = self.v_max
                self.v_max = perturbed_value
                self._compute_resonance_properties()
            elif param == 'T_period':
                old_value = self.T_period
                self.T_period = perturbed_value
                self.omega = 2 * np.pi / self.T_period
                self.f_characteristic = 1 / self.T_period
                self._compute_resonance_properties()
            
            # Compute perturbed enhancement
            perturbed_enhancement = self.resonance_enhancement_factor(self.f_characteristic)
            
            # Restore original value
            if param == 'v_max':
                self.v_max = old_value
            elif param == 'T_period':
                self.T_period = old_value
                self.omega = 2 * np.pi / self.T_period
                self.f_characteristic = 1 / self.T_period
            
            self._compute_resonance_properties()
            
            # Compute sensitivity
            relative_change = (perturbed_enhancement - base_enhancement) / base_enhancement
            sensitivity = relative_change / perturbation_amplitude
            sensitivities[param] = sensitivity
        
        return {
            'base_enhancement': base_enhancement,
            'parameter_sensitivities': sensitivities,
            'most_sensitive_parameter': max(sensitivities.keys(), key=lambda k: abs(sensitivities[k])),
            'stability_metric': 1 / max(abs(s) for s in sensitivities.values()),
            'perturbation_amplitude': perturbation_amplitude
        }
    
    def energy_efficiency_analysis(self, mass: float, transport_distance: float) -> Dict:
        """
        Analyze energy efficiency of time-dependent transport.
        
        Args:
            mass: Payload mass (kg)
            transport_distance: Transport distance (m)
            
        Returns:
            Energy efficiency analysis
        """
        # Time to travel distance at average velocity
        v_avg = self.v_max * (2 / np.pi)  # Average of |sin(x)|
        transport_time = transport_distance / v_avg
        
        # Energy components
        times = np.linspace(0, transport_time, 1000)
        kinetic_energies = self.kinetic_energy(jnp.array(times), mass)
        
        # Total energy (integral over time)
        total_kinetic_energy = np.trapz(kinetic_energies, times)
        
        # Compare to constant velocity transport
        constant_velocity_energy = 0.5 * mass * v_avg**2 * transport_time
        
        # Resonance enhancement
        field_freq_estimate = v_avg / transport_distance  # Rough estimate
        enhancement = self.resonance_enhancement_factor(field_freq_estimate)
        
        return {
            'transport_time': transport_time,
            'average_velocity': v_avg,
            'total_kinetic_energy': total_kinetic_energy,
            'constant_velocity_energy': constant_velocity_energy,
            'energy_efficiency_ratio': constant_velocity_energy / total_kinetic_energy,
            'resonance_enhancement': enhancement,
            'effective_energy_reduction': enhancement * constant_velocity_energy / total_kinetic_energy,
            'peak_velocity': self.v_max,
            'velocity_modulation_factor': self.v_max / v_avg
        }

# Utility functions
def fourier_analysis_velocity(resonance: TimeDependentResonance, 
                             duration: float, n_harmonics: int = 10) -> Dict:
    """
    Fourier analysis of velocity profile.
    
    Args:
        resonance: TimeDependentResonance instance
        duration: Analysis duration
        n_harmonics: Number of harmonics to analyze
        
    Returns:
        Fourier analysis results
    """
    # Sample velocity over multiple periods
    n_periods = max(1, int(duration / resonance.T_period))
    n_samples = n_periods * 100  # 100 samples per period
    
    times = np.linspace(0, duration, n_samples)
    velocities = np.array([float(resonance.v_s(jnp.array([t]))) for t in times])
    
    # FFT
    fft_result = np.fft.fft(velocities)
    frequencies = np.fft.fftfreq(n_samples, duration / n_samples)
    
    # Extract harmonics
    fundamental_freq = 1 / resonance.T_period
    harmonic_amplitudes = {}
    harmonic_phases = {}
    
    for h in range(1, n_harmonics + 1):
        # Find closest frequency bin
        target_freq = h * fundamental_freq
        idx = np.argmin(np.abs(frequencies - target_freq))
        
        amplitude = np.abs(fft_result[idx]) / n_samples * 2  # Normalize
        phase = np.angle(fft_result[idx])
        
        harmonic_amplitudes[h] = amplitude
        harmonic_phases[h] = phase
    
    return {
        'fundamental_frequency': fundamental_freq,
        'harmonic_amplitudes': harmonic_amplitudes,
        'harmonic_phases': harmonic_phases,
        'total_harmonic_distortion': np.sqrt(sum(a**2 for h, a in harmonic_amplitudes.items() if h > 1)) / harmonic_amplitudes[1],
        'frequency_spectrum': (frequencies[:n_samples//2], np.abs(fft_result[:n_samples//2]))
    }

if __name__ == "__main__":
    # Demonstration of time-dependent resonance
    print("Time-Dependent Resonance Demonstration")
    print("=" * 45)
    
    # Configuration
    config = ResonanceConfig(
        v_max=1e6,           # 1000 km/s peak velocity
        T_period=3600.0,     # 1 hour period
        resonance_modes=[1, 2, 3],
        damping_coefficient=0.05,
        field_coupling=1.0
    )
    
    resonance = TimeDependentResonance(config)
    
    print(f"Configuration:")
    print(f"  Peak velocity: {config.v_max/1000:.0f} km/s")
    print(f"  Period: {config.T_period/3600:.1f} hours")
    print(f"  Fundamental frequency: {resonance.f_characteristic*1000:.3f} mHz")
    print(f"  Angular frequency: {resonance.omega*1000:.3f} mrad/s")
    
    # Test velocity profile
    times = np.linspace(0, config.T_period, 100)
    velocities = np.array([float(resonance.v_s(jnp.array([t]))) for t in times])
    
    print(f"\nVelocity Profile:")
    print(f"  Average velocity: {np.mean(np.abs(velocities))/1000:.0f} km/s")
    print(f"  RMS velocity: {np.sqrt(np.mean(velocities**2))/1000:.0f} km/s")
    
    # Resonance analysis
    test_freq = resonance.f_characteristic
    enhancement = resonance.resonance_enhancement_factor(test_freq)
    print(f"\nResonance Analysis:")
    print(f"  Enhancement at fundamental: {enhancement:.3f}×")
    
    # Frequency sweep
    freq_range = np.linspace(0.1 * resonance.f_characteristic, 
                            5 * resonance.f_characteristic, 100)
    sweep = resonance.resonance_sweep_analysis(freq_range)
    print(f"  Maximum enhancement: {sweep['max_enhancement']:.3f}×")
    print(f"  Optimal frequency: {sweep['optimal_frequency']*1000:.3f} mHz")
    print(f"  Number of peaks: {len(sweep['peak_frequencies'])}")
    
    # Energy efficiency
    mass = 1000.0  # 1000 kg payload
    distance = 1e9  # 1000 km transport
    efficiency = resonance.energy_efficiency_analysis(mass, distance)
    print(f"\nEnergy Efficiency (1000 kg, 1000 km):")
    print(f"  Transport time: {efficiency['transport_time']/3600:.2f} hours")
    print(f"  Energy efficiency ratio: {efficiency['energy_efficiency_ratio']:.3f}")
    print(f"  Effective energy reduction: {efficiency['effective_energy_reduction']:.3f}×")
