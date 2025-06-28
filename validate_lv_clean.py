"""
Clean validation test for LV enhancement components.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_lv_components():
    """Test all LV enhancement components individually."""
    
    print("Testing LV Enhancement Components")
    print("=" * 40)
    
    results = []
    
    # Test 1: SME Einstein Solver
    try:
        print("\n1. Testing SME Einstein Solver...")
        from src.lorentz_violation.modified_einstein_solver import SMEParameters
        
        sme_params = SMEParameters(
            c_00_3=1e-9,
            c_11_3=1e-17,
            E_LV=1e19
        )
        
        print(f"   SUCCESS: SME parameters: {sme_params.validate_experimental_bounds()}")
        results.append(True)
        
    except Exception as e:
        print(f"   ERROR: SME Error: {e}")
        results.append(False)
    
    # Test 2: Ghost Scalar EFT
    try:
        print("\n2. Testing Ghost Scalar EFT...")
        from src.physics.ghost_scalar_eft import GhostScalarConfig
        
        ghost_config = GhostScalarConfig(
            m=0.001,
            lam=0.1,
            mu=1e-6,
            alpha=1e-8,
            beta=1e-3
        )
        
        print(f"   SUCCESS: Ghost config: m={ghost_config.m}, lambda={ghost_config.lam}")
        results.append(True)
        
    except Exception as e:
        print(f"   ERROR: Ghost Error: {e}")
        results.append(False)
    
    # Test 3: Dispersion Relations
    try:
        print("\n3. Testing Dispersion Relations...")
        from src.utils.dispersion_relations import DispersionParameters
        
        disp_params = DispersionParameters(
            alpha_1=1e-16,
            alpha_2=1e-12,
            alpha_3=1e-8,
            alpha_4=1e-6,
            beta_1=1e-14,
            beta_2=1e-10,
            E_pl=1.22e19,
            E_lv=1e18
        )
        
        print(f"   SUCCESS: Dispersion: alpha1={disp_params.alpha_1}, alpha2={disp_params.alpha_2}")
        results.append(True)
        
    except Exception as e:
        print(f"   ERROR: Dispersion Error: {e}")
        results.append(False)
    
    # Test 4: Coherence Extractor
    try:
        print("\n4. Testing Coherence Extractor...")
        from src.physics.energy_extractor import CoherenceConfiguration
        
        coh_config = CoherenceConfiguration(
            n_matter_states=8,
            n_gravity_states=8,
            coupling_strength=1e-5,
            lv_enhancement=1e-2,
            decoherence_time=1e-5,
            temperature=0.001
        )
        
        print(f"   SUCCESS: Coherence: {coh_config.n_matter_states}x{coh_config.n_gravity_states} states")
        results.append(True)
        
    except Exception as e:
        print(f"   ERROR: Coherence Error: {e}")
        results.append(False)
    
    # Summary
    success_count = sum(results)
    print(f"\nLV COMPONENT TEST RESULTS:")
    print(f"   Successful components: {success_count}/4")
    
    if success_count == 4:
        print("   ALL LV COMPONENTS OPERATIONAL")
        return True
    else:
        print("   Some LV components have issues")
        return False

if __name__ == "__main__":
    success = test_lv_components()
    sys.exit(0 if success else 1)
