#!/usr/bin/env python3
"""
Test just SME component functionality
"""

import sys
import os
sys.path.append('src')
import time

def test_sme_only():
    print("Testing SME component only...")
    
    try:
        # Import and test SME
        from src.lorentz_violation.modified_einstein_solver import SMEParameters
        
        print("Creating SME parameters...")
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        print("SUCCESS: SME parameters created")
        
        print("Testing parameter validation...")
        is_valid = sme_params.validate_experimental_bounds()
        print(f"SUCCESS: Parameters valid: {is_valid}")
        
        # Create transporter for SME solver
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        print("Creating transporter...")
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        transporter = EnhancedStargateTransporter(config)
        print("SUCCESS: Transporter created")
        
        # Now test SME solver creation
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver
        
        print("Creating SME solver...")
        sme_solver = SMEEinsteinSolver(transporter, sme_params)
        print("SUCCESS: SME solver created")
        
        # Test enhancement calculation
        print("Testing enhancement calculation...")
        start_time = time.time()
        enhancement = sme_solver.compute_enhancement_factor(100.0)
        calc_time = time.time() - start_time
        print(f"SUCCESS: Enhancement factor: {enhancement:.6f} (took {calc_time:.3f}s)")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sme_only()
