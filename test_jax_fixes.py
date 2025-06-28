#!/usr/bin/env python3
"""
Test JAX fixes directly
"""

import sys
import os
sys.path.append('src')

def test_jax_fixes():
    print("Testing JAX fixes...")
    
    try:
        # Test SME solver
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        print("Creating SME components...")
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        transporter = EnhancedStargateTransporter(config)
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        sme_solver = SMEEinsteinSolver(transporter, sme_params)
        
        print("Testing SME enhancement calculation...")
        enhancement = sme_solver.compute_enhancement_factor(100.0)
        print(f"SME enhancement: {enhancement:.6f}")
        
        # Test dispersion relations
        from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
        
        print("Creating dispersion relations...")
        disp_params = DispersionParameters(alpha_1=1e-16, alpha_2=1e-12)
        dispersion = PolynomialDispersionRelations(disp_params)
        
        print("Testing dispersion enhancement...")
        import jax.numpy as jnp
        p_test = jnp.array([100.0])
        m_test = 1.0
        disp_enhancement = dispersion.enhancement_factor(p_test, m_test)[0]
        print(f"Dispersion enhancement: {disp_enhancement:.6f}")
        
        print("SUCCESS: All JAX fixes working!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_jax_fixes()
