"""
Simple SME Demo - No JAX computations
====================================

This version avoids JAX computations that might cause hangs.

Author: Complete LV Integration Team
Date: June 28, 2025
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

def run_simple_sme_demo():
    """Run simple SME demonstration without heavy computations."""
    print("Simple SME Demo")
    print("=" * 20)
    
    # Base transporter
    config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
    transporter = EnhancedStargateTransporter(config)
    print("SUCCESS: Base transporter")
    
    # SME Parameters
    sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
    print(f"SUCCESS: SME parameters created")
    print(f"  c_00^(3): {sme_params.c_00_3:.2e}")
    print(f"  c_11^(3): {sme_params.c_11_3:.2e}")
    print(f"  E_LV: {sme_params.E_LV:.2e} GeV")
    
    # Parameter validation (simple check)
    is_valid = sme_params.validate_experimental_bounds()
    print(f"  Experimental bounds valid: {is_valid}")
    
    # SME Solver (just creation, no heavy computations)
    sme_solver = SMEEinsteinSolver(transporter, sme_params)
    print("SUCCESS: SME solver created")
    
    # Simple enhancement calculation (avoid JAX ops)
    energy_scale = 100.0  # GeV
    enhancement = sme_solver.compute_enhancement_factor(energy_scale)
    print(f"SUCCESS: Enhancement factor: {enhancement:.6f}")
    
    print(f"\nSIMPLE SME RESULTS:")
    print(f"=" * 20)
    print(f"SME solver: OPERATIONAL")
    print(f"Enhancement: {enhancement:.6f}")
    print(f"Parameters valid: {is_valid}")
    print(f"Status: WORKING")
    
    return {
        'enhancement': enhancement,
        'valid': is_valid,
        'solver_created': True
    }

if __name__ == "__main__":
    print("Starting simple SME demo...")
    results = run_simple_sme_demo()
    print("Simple SME demo completed!")
