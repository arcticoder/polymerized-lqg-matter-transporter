#!/usr/bin/env python3
"""
Test imports for demonstrate_lv_enhanced_framework_clean.py
"""

import sys
import os
sys.path.append('src')

def test_imports():
    print("Testing imports for clean framework...")
    
    try:
        # Test core imports
        from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
        print("SUCCESS: SME imports")
        
        from src.physics.ghost_scalar_eft import GhostScalarEFT, GhostScalarConfig
        print("SUCCESS: Ghost EFT imports")
        
        from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
        print("SUCCESS: Dispersion imports")
        
        from src.physics.energy_extractor import MatterGravityCoherenceExtractor, CoherenceConfiguration
        print("SUCCESS: Energy extractor imports")
        
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        print("SUCCESS: Core transporter imports")
        
        print("All imports successful!")
        return True
        
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
