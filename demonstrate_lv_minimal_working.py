"""
Minimal Working LV Demo - Only SME and Dispersion
================================================

This version only uses components that are confirmed to work without hanging.

Author: Complete LV Integration Team
Date: June 28, 2025
"""

import sys
import os
import time
from typing import Dict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import only working components
from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

class MinimalWorkingLVFramework:
    """Minimal working LV framework with only confirmed working components."""
    
    def __init__(self):
        """Initialize minimal framework."""
        print("Minimal Working LV Framework")
        print("=" * 35)
        
        # Base transporter
        config = EnhancedTransporterConfig(R_payload=2.0, R_neck=0.08, L_corridor=2.0)
        self.transporter = EnhancedStargateTransporter(config)
        print("SUCCESS: Base transporter initialized")
        
    def initialize_components(self):
        """Initialize working components only."""
        print("\nInitializing working components...")
        
        # SME Solver
        sme_params = SMEParameters(c_00_3=1e-9, c_11_3=1e-17, E_LV=1e19)
        self.sme_solver = SMEEinsteinSolver(self.transporter, sme_params)
        print("SUCCESS: SME solver initialized")
        
        # Dispersion Relations
        disp_params = DispersionParameters(alpha_1=1e-16, alpha_2=1e-12, alpha_3=1e-8, alpha_4=1e-6)
        self.dispersion = PolynomialDispersionRelations(disp_params)
        print("SUCCESS: Dispersion relations initialized")
        
    def run_tests(self):
        """Run tests of working components."""
        print("\nRunning component tests...")
        
        # Test SME
        print("Testing SME enhancement...")
        start_time = time.time()
        sme_enhancement = self.sme_solver.compute_enhancement_factor(100.0)
        sme_time = time.time() - start_time
        print(f"SUCCESS: SME enhancement: {sme_enhancement:.6f} ({sme_time:.3f}s)")
        
        # Test Dispersion
        print("Testing dispersion enhancement...")
        start_time = time.time()
        import jax.numpy as jnp
        p_test = jnp.array([100.0])
        m_test = 75.0
        disp_enhancement = self.dispersion.enhancement_factor(p_test, m_test)[0]
        disp_time = time.time() - start_time
        print(f"SUCCESS: Dispersion enhancement: {disp_enhancement:.6f} ({disp_time:.3f}s)")
        
        # Summary
        total_enhancement = sme_enhancement * float(disp_enhancement)
        total_time = sme_time + disp_time
        
        print(f"\nMINIMAL FRAMEWORK RESULTS:")
        print(f"=" * 30)
        print(f"SME Enhancement: {sme_enhancement:.6f}")
        print(f"Dispersion Enhancement: {float(disp_enhancement):.6f}")
        print(f"Total Enhancement: {total_enhancement:.6f}")
        print(f"Total Time: {total_time:.3f} seconds")
        print(f"Status: WORKING")
        
        return {
            'sme_enhancement': sme_enhancement,
            'dispersion_enhancement': float(disp_enhancement),
            'total_enhancement': total_enhancement,
            'total_time': total_time
        }

def run_minimal_demo():
    """Run minimal working demo."""
    framework = MinimalWorkingLVFramework()
    framework.initialize_components()
    results = framework.run_tests()
    return framework, results

if __name__ == "__main__":
    print("Starting minimal working LV demo...")
    framework, results = run_minimal_demo()
    print("Minimal demo completed successfully!")
