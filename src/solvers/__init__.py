"""
Advanced Numerical Solvers Module

This module provides high-performance numerical solvers for the
polymerized LQG matter transporter system.

Components:
- JAX pmap domain decomposition solver
- Newton-Raphson iterative solver
- Optimized 3D Laplacian operator
"""

from .jax_pmap_domain_solver import JAXPmapDomainDecompositionSolver
from .newton_raphson_solver import NewtonRaphsonIterativeSolver
from .optimized_3d_laplacian import Optimized3DLaplacianOperator

__all__ = [
    'JAXPmapDomainDecompositionSolver',
    'NewtonRaphsonIterativeSolver',
    'Optimized3DLaplacianOperator'
]
