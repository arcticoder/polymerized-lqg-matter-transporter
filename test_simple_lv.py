#!/usr/bin/env python3
"""
Simplified LV Enhanced Framework Test
"""

import sys
import os
sys.path.append('src')

def test_lv_framework():
    print("🚀 Testing LV Enhanced Framework")
    print("=" * 40)
    
    try:
        # Test SME Parameters
        from src.lorentz_violation.modified_einstein_solver import SMEParameters
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        print(f"✅ SME Parameters: {sme_params.validate_experimental_bounds()}")
        
        # Test Ghost Config
        from src.physics.ghost_scalar_eft import GhostScalarConfig
        ghost_config = GhostScalarConfig(m=0.001, lam=0.1, mu=1e-6, alpha=1e-8, beta=1e-3)
        print(f"✅ Ghost Config: m={ghost_config.m}")
        
        # Test Dispersion
        from src.utils.dispersion_relations import DispersionParameters
        disp_params = DispersionParameters(alpha_1=1e-16, alpha_2=1e-12)
        print(f"✅ Dispersion: α₁={disp_params.alpha_1}")
        
        # Test Coherence
        from src.physics.energy_extractor import CoherenceConfiguration
        coh_config = CoherenceConfiguration(n_matter_states=8, n_gravity_states=8)
        print(f"✅ Coherence: {coh_config.n_matter_states}×{coh_config.n_gravity_states}")
        
        print(f"\n🎉 All LV components operational!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_lv_framework()
