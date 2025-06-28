"""
Optimization module for Enhanced Stargate Transporter

Provides automated parameter optimization capabilities using JAX autodiff
and SciPy optimization algorithms.
"""

from .parameter_opt import TransporterOptimizer

__all__ = ['TransporterOptimizer']
