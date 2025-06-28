"""
SME/Ghost-Scalar/Dispersion-Corrected Enhanced Transporter Demo (MINIMAL)
========================================================================

Minimal working version that demonstrates LV framework without heavy computations.

Author: Complete LV Integration Team
Date: June 28, 2025
"""

import sys
import os
import time
from typing import Dict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all LV enhancement components
from src.lorentz_violation.modified_einstein_solver import SMEEinsteinSolver, SMEParameters
from src.physics.ghost_scalar_eft import GhostScalarEFT, GhostScalarConfig
from src.utils.dispersion_relations import PolynomialDispersionRelations, DispersionParameters
from src.physics.energy_extractor import MatterGravityCoherenceExtractor, CoherenceConfiguration

# Import core transporter framework
from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

class MinimalLVFramework:
    """
    Minimal LV-Enhanced Framework
    
    Demonstrates LV components without heavy computational overhead.
    """
    
    def __init__(self):
        """Initialize minimal LV framework."""
        print("Minimal LV-Enhanced Transporter Framework")
        print("=" * 50)
        
        # Base transporter configuration (simplified)
        self.transporter_config = EnhancedTransporterConfig(
            R_payload=2.0,
            R_neck=0.08,
            L_corridor=2.0
        )
        
        # Create base transporter
        self.transporter = EnhancedStargateTransporter(self.transporter_config)
        
        # Results storage
        self.results = {}
        
        print("SUCCESS: Base transporter initialized")
    
    def initialize_lv_components(self):
        """Initialize LV components with minimal parameters."""
        print("\nInitializing LV components...")
        
        # 1. SME-Enhanced Einstein Solver
        print("1. SME-Enhanced Einstein Solver")
        sme_params = SMEParameters(
            c_00_3=1e-9,
            c_11_3=1e-17,
            E_LV=1e19
        )
        self.sme_solver = SMEEinsteinSolver(self.transporter, sme_params)
        print("   SUCCESS: SME solver initialized")
        
        # 2. Ghost-Scalar EFT (minimal grid)
        print("2. Ghost-Scalar EFT")
        ghost_config = GhostScalarConfig(
            m=0.001,
            lam=0.1,
            mu=1e-6,
            alpha=1e-8,
            beta=1e-3,
            L=2.0,    # Smaller domain
            N=8,      # Very small grid
            dt=0.1,   # Larger time step
            T_max=1.0 # Shorter evolution
        )
        self.ghost_eft = GhostScalarEFT(ghost_config)
        print("   SUCCESS: Ghost EFT initialized")
        
        # 3. Polynomial Dispersion Relations
        print("3. Polynomial Dispersion Relations")
        dispersion_params = DispersionParameters(
            alpha_1=1e-16,
            alpha_2=1e-12,
            alpha_3=1e-8,
            alpha_4=1e-6
        )
        self.dispersion = PolynomialDispersionRelations(dispersion_params)
        print("   SUCCESS: Dispersion relations initialized")
        
        # 4. Matter-Gravity Coherence Extractor (minimal system)
        print("4. Matter-Gravity Coherence Extractor")
        coherence_config = CoherenceConfiguration(
            n_matter_states=4,
            n_gravity_states=4,
            coupling_strength=1e-5,
            lv_enhancement=1e-2,
            decoherence_time=1e-5,
            temperature=0.001
        )
        self.coherence_extractor = MatterGravityCoherenceExtractor(coherence_config)
        print("   SUCCESS: Coherence extractor initialized")
        
        print("\nSUCCESS: All LV components initialized!")
    
    def run_minimal_tests(self) -> Dict:
        """Run minimal tests of each component."""
        print("\nRunning minimal LV component tests...")
        
        results = {}
        
        # Test SME enhancement
        print("Testing SME enhancement...")
        start_time = time.time()
        sme_enhancement = self.sme_solver.compute_enhancement_factor(100.0)
        sme_time = time.time() - start_time
        results['sme'] = {
            'enhancement_factor': sme_enhancement,
            'computation_time': sme_time
        }
        print(f"   SUCCESS: SME enhancement: {sme_enhancement:.6f} ({sme_time:.3f}s)")
        
        # Test dispersion enhancement
        print("Testing dispersion enhancement...")
        start_time = time.time()
        import jax.numpy as jnp
        p_test = jnp.array([1.0])
        m_test = 1.0
        disp_enhancement = self.dispersion.enhancement_factor(p_test, m_test)[0]
        disp_time = time.time() - start_time
        results['dispersion'] = {
            'enhancement_factor': float(disp_enhancement),
            'computation_time': disp_time
        }
        print(f"   SUCCESS: Dispersion enhancement: {disp_enhancement:.6f} ({disp_time:.3f}s)")
        
        # Test coherence extraction (simple case)
        print("Testing coherence extraction...")
        start_time = time.time()
        extraction_result = self.coherence_extractor.extract_coherent_energy()
        coh_time = time.time() - start_time
        results['coherence'] = {
            'extractable_energy': extraction_result.extractable_energy,
            'coherence_factor': extraction_result.coherence_factor,
            'computation_time': coh_time
        }
        print(f"   SUCCESS: Extractable energy: {extraction_result.extractable_energy:.2e} J ({coh_time:.3f}s)")
        
        # Skip ghost field evolution for now (too computationally heavy)
        print("Skipping ghost field evolution (computationally intensive)")
        results['ghost'] = {
            'status': 'initialized',
            'computation_time': 0.0
        }
        
        return results
    
    def compute_framework_summary(self, results: Dict) -> Dict:
        """Compute framework summary."""
        print("\nComputing framework summary...")
        
        # Calculate total enhancement from available components
        sme_factor = results['sme']['enhancement_factor']
        disp_factor = results['dispersion']['enhancement_factor']
        
        # Simple combined enhancement
        total_enhancement = sme_factor * disp_factor
        
        # Total computation time
        total_time = sum([
            results['sme']['computation_time'],
            results['dispersion']['computation_time'],
            results['coherence']['computation_time']
        ])
        
        summary = {
            'total_enhancement_factor': total_enhancement,
            'total_computation_time': total_time,
            'components_tested': 3,
            'components_initialized': 4
        }
        
        return summary
    
    def run_minimal_framework(self) -> Dict:
        """Run minimal LV framework demonstration."""
        
        # Initialize components
        self.initialize_lv_components()
        
        # Run tests
        self.results = self.run_minimal_tests()
        
        # Compute summary
        summary = self.compute_framework_summary(self.results)
        self.results['summary'] = summary
        
        # Display results
        print("\nMINIMAL LV FRAMEWORK RESULTS:")
        print("=" * 40)
        print(f"SUCCESS: SME-Enhanced Einstein: Operational")
        print(f"SUCCESS: Ghost-Scalar EFT: Initialized")
        print(f"SUCCESS: Polynomial Dispersion: Operational")
        print(f"SUCCESS: Coherence Extraction: Operational")
        print(f"")
        print(f"Enhancement Factor: {summary['total_enhancement_factor']:.2e}")
        print(f"Total Computation Time: {summary['total_computation_time']:.3f} seconds")
        print(f"Components Tested: {summary['components_tested']}/4")
        print(f"Framework Status: OPERATIONAL")
        
        return self.results

def run_minimal_lv_demo():
    """Run minimal LV-enhanced transporter demonstration."""
    
    # Create framework
    framework = MinimalLVFramework()
    
    # Run demonstration
    results = framework.run_minimal_framework()
    
    return framework, results

if __name__ == "__main__":
    print("Starting minimal LV framework demonstration...")
    framework, results = run_minimal_lv_demo()
    print("\nMinimal demonstration completed successfully!")
