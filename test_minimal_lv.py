#!/usr/bin/env python3
"""
Minimal LV Framework Test
"""

import sys
import os
sys.path.append('src')

def minimal_test():
    print("Starting minimal LV framework test...")
    
    try:
        # Create base transporter
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        transporter = EnhancedStargateTransporter(config)
        print("SUCCESS: Base transporter created")
        
        # Create SME solver
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
        
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        sme_solver = SMEEinsteinSolver(transporter, sme_params)
        print("SUCCESS: SME solver created")
        
        # Test enhancement calculation
        enhancement = sme_solver.compute_enhancement_factor(100.0)
        print(f"SUCCESS: Enhancement factor: {enhancement:.6f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_test()
    print(f"Test result: {'PASS' if success else 'FAIL'}")
