"""
Enhanced Control Systems Module

This module provides advanced control theory implementations for the
polymerized LQG matter transporter system.

Components:
- H∞ optimal controller
- Multi-variable PID controller  
- Quantum error correction injector
"""

from .hinfty_controller import HInfinityController
from .multivar_pid_controller import MultiVarPIDController
from .qec_injector import QECInjector

__all__ = [
    'HInfinityController',
    'MultiVarPIDController', 
    'QECInjector'
]
