#!/usr/bin/env python3
"""
Test individual LV component creation
"""

import sys
import os
sys.path.append('src')

def test_lv_component_creation():
    print("Testing LV component creation...")
    
    try:
        # Create base transporter first
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        transporter = EnhancedStargateTransporter(config)
        print("SUCCESS: Base transporter created")
        
        # Test SME solver
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
        
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        print("SUCCESS: SME params created")
        
        sme_solver = SMEEinsteinSolver(transporter, sme_params)
        print("SUCCESS: SME solver created")
        
        # Test Ghost EFT
        from src.physics.ghost_scalar_eft import GhostScalarEFT, GhostScalarConfig
        
        ghost_config = GhostScalarConfig(m=0.001, lam=0.1, mu=1e-6, alpha=1e-8, beta=1e-3)
        print("SUCCESS: Ghost config created")
        
        ghost_eft = GhostScalarEFT(ghost_config)
        print("SUCCESS: Ghost EFT created")
        
        # Test Dispersion
        from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
        
        disp_params = DispersionParameters(alpha_1=1e-16, alpha_2=1e-12)
        print("SUCCESS: Dispersion params created")
        
        dispersion = PolynomialDispersionRelations(disp_params)
        print("SUCCESS: Dispersion relations created")
        
        # Test Coherence Extractor
        from src.physics.energy_extractor import MatterGravityCoherenceExtractor, CoherenceConfiguration
        
        coh_config = CoherenceConfiguration(n_matter_states=8, n_gravity_states=8)
        print("SUCCESS: Coherence config created")
        
        extractor = MatterGravityCoherenceExtractor(coh_config)
        print("SUCCESS: Coherence extractor created")
        
        print("All LV components created successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_lv_component_creation()
