#!/usr/bin/env python3
"""
Test LV component methods to identify where the hang occurs
"""

import sys
import os
sys.path.append('src')
import time

def test_component_methods():
    print("Testing LV component methods...")
    
    try:
        # Create all components first
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
        from src.physics.ghost_scalar_eft import GhostScalarEFT, GhostScalarConfig
        from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
        from src.physics.energy_extractor import MatterGravityCoherenceExtractor, CoherenceConfiguration
        
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        transporter = EnhancedStargateTransporter(config)
        
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        sme_solver = SMEEinsteinSolver(transporter, sme_params)
        
        ghost_config = GhostScalarConfig(m=0.001, lam=0.1, mu=1e-6, alpha=1e-8, beta=1e-3, L=5.0, N=16)  # Smaller grid
        ghost_eft = GhostScalarEFT(ghost_config)
        
        disp_params = DispersionParameters(alpha_1=1e-16, alpha_2=1e-12)
        dispersion = PolynomialDispersionRelations(disp_params)
        
        coh_config = CoherenceConfiguration(n_matter_states=4, n_gravity_states=4)  # Smaller system
        extractor = MatterGravityCoherenceExtractor(coh_config)
        
        print("All components created, testing methods...")
        
        # Test SME enhancement calculation
        print("Testing SME enhancement...")
        start_time = time.time()
        enhancement = sme_solver.compute_enhancement_factor(100.0)
        print(f"SUCCESS: SME enhancement: {enhancement:.6f} ({time.time() - start_time:.3f}s)")
        
        # Test ghost field initialization (skip evolution for now)
        print("Testing ghost field initialization...")
        start_time = time.time()
        psi_initial = ghost_eft.initialize_field("gaussian")
        print(f"SUCCESS: Ghost field init ({time.time() - start_time:.3f}s)")
        
        # Test dispersion enhancement
        print("Testing dispersion enhancement...")
        start_time = time.time()
        import jax.numpy as jnp
        p_test = jnp.array([1.0])
        m_test = 1.0
        disp_enhancement = dispersion.enhancement_factor(p_test, m_test)[0]
        print(f"SUCCESS: Dispersion enhancement: {disp_enhancement:.6f} ({time.time() - start_time:.3f}s)")
        
        # Test coherence energy extraction
        print("Testing coherence extraction...")
        start_time = time.time()
        extraction_result = extractor.extract_coherent_energy()
        print(f"SUCCESS: Coherence extraction: {extraction_result.extractable_energy:.2e} J ({time.time() - start_time:.3f}s)")
        
        print("All methods tested successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_component_methods()
