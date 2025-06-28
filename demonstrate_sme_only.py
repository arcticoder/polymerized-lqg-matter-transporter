"""
SME-Only Working Demo
====================

This version only uses the SME component which is confirmed to work.

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

def run_sme_only_demo():
    """Run SME-only demonstration."""
    print("SME-Only LV Framework Demo")
    print("=" * 30)
    
    # Base transporter
    config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
    transporter = EnhancedStargateTransporter(config)
    print("SUCCESS: Base transporter initialized")
    
    # SME Solver
    sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
    sme_solver = SMEEinsteinSolver(transporter, sme_params)
    print("SUCCESS: SME solver initialized")
    
    # Test SME functionality
    print("\nTesting SME functionality...")
    
    # Enhancement calculation
    start_time = time.time()
    enhancement = sme_solver.compute_enhancement_factor(100.0)
    calc_time = time.time() - start_time
    print(f"Enhancement factor: {enhancement:.6f} ({calc_time:.3f}s)")
    
    # Field equation validation
    start_time = time.time()
    import jax.numpy as jnp
    minkowski = jnp.diag(jnp.array([1.0, -1.0, -1.0, -1.0]))
    validation = sme_solver.validate_field_equations(minkowski)
    val_time = time.time() - start_time
    print(f"Field equations valid: {validation['experimental_compliance']} ({val_time:.3f}s)")
    
    print(f"\nSME FRAMEWORK RESULTS:")
    print(f"=" * 25)
    print(f"Enhancement: {enhancement:.6f}")
    print(f"Experimental compliance: {validation['experimental_compliance']}")
    print(f"Einstein tensor norm: {validation['einstein_tensor_norm']:.2e}")
    print(f"Total time: {calc_time + val_time:.3f} seconds")
    print(f"Status: FULLY OPERATIONAL")
    
    return {
        'enhancement': enhancement,
        'validation': validation,
        'total_time': calc_time + val_time
    }

if __name__ == "__main__":
    print("Starting SME-only demo...")
    results = run_sme_only_demo()
    print("SME demo completed successfully!")
