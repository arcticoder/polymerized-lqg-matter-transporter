"""
Physics module for Enhanced Stargate Transporter

Provides negative energy generation and exotic matter modeling capabilities.
"""

from .negative_energy import (
    CasimirConfig, 
    CasimirGenerator,
    SqueezeVacuumConfig,
    SqueezeVacuumGenerator, 
    IntegratedNegativeEnergySystem
)

__all__ = [
    'CasimirConfig',
    'CasimirGenerator', 
    'SqueezeVacuumConfig',
    'SqueezeVacuumGenerator',
    'IntegratedNegativeEnergySystem'
]
