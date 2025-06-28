"""
Quick test to verify the three workstreams are properly integrated
and can be run independently.
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_workstream_1():
    """Test parameter optimization workstream."""
    try:
        print("🔧 Testing Workstream 1: Parameter Optimization...")
        
        from optimization.parameter_opt import TransporterOptimizer
        from core.enhanced_stargate_transporter import EnhancedTransporterConfig
        
        # Quick test
        config = EnhancedTransporterConfig()
        optimizer = TransporterOptimizer(config)
        
        # Test objective function
        import numpy as np
        test_params = np.array([0.15, 1.8, 1800.0, 0.08])
        energy = optimizer.objective_scipy(test_params)
        
        print(f"   ✅ Parameter optimization functional")
        print(f"   Test energy: {energy:.2e} J")
        return True
        
    except Exception as e:
        print(f"   ❌ Parameter optimization failed: {e}")
        return False

def test_workstream_2():
    """Test dynamic corridor workstream."""
    try:
        print("🌊 Testing Workstream 2: Dynamic Corridors...")
        
        from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
        
        # Test dynamic configuration
        config = EnhancedTransporterConfig(
            corridor_mode="sinusoidal",
            v_conveyor_max=1e6,
            temporal_scale=1800.0
        )
        
        transporter = EnhancedStargateTransporter(config)
        
        # Test time-dependent velocity
        v_t0 = transporter.v_s(0.0)
        v_t1 = transporter.v_s(900.0)  # Quarter period
        
        print(f"   ✅ Dynamic corridors functional")
        print(f"   v(t=0): {v_t0:.2e} m/s, v(t=T/4): {v_t1:.2e} m/s")
        return True
        
    except Exception as e:
        print(f"   ❌ Dynamic corridors failed: {e}")
        return False

def test_workstream_3():
    """Test Casimir integration workstream."""
    try:
        print("⚡ Testing Workstream 3: Casimir Integration...")
        
        from physics.negative_energy import CasimirConfig, CasimirGenerator
        
        # Test Casimir generator
        config = CasimirConfig(plate_separation=1e-6, num_plates=10)
        generator = CasimirGenerator(config)
        
        # Test density calculation
        density = generator.static_casimir_density(1e-6)
        reduction_factor = generator.casimir_reduction_factor(1e-6)  # 1 μm³ volume
        
        print(f"   ✅ Casimir integration functional")
        print(f"   Density: {density:.2e} J/m³, Reduction: {reduction_factor:.2e}")
        return True
        
    except Exception as e:
        print(f"   ❌ Casimir integration failed: {e}")
        return False

def test_integrated_system():
    """Test integrated system.""" 
    try:
        print("🔗 Testing Integrated System...")
        
        from core.enhanced_stargate_transporter import EnhancedTransporterConfig, EnhancedStargateTransporter
        
        # Test with Casimir integration
        config = EnhancedTransporterConfig(
            corridor_mode="sinusoidal",
            v_conveyor_max=1e6,
            use_van_den_broeck=True,
            use_temporal_smearing=True,
            use_multi_bubble=True
        )
        
        transporter = EnhancedStargateTransporter(config)
        energy_analysis = transporter.compute_total_energy_requirement(3600.0, 75.0)
        
        print(f"   ✅ Integrated system functional")
        print(f"   Total reduction: {energy_analysis['total_reduction_factor']:.2e}×")
        print(f"   Casimir integrated: {'E_after_casimir' in energy_analysis}")
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated system failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("WORKSTREAM INTEGRATION TEST")
    print("="*60)
    
    tests = [
        test_workstream_1,
        test_workstream_2, 
        test_workstream_3,
        test_integrated_system
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    success_count = sum(results)
    total_count = len(results)
    
    print("="*60)
    print(f"TEST RESULTS: {success_count}/{total_count} workstreams functional")
    
    if success_count == total_count:
        print("🌟 ALL SYSTEMS READY - Prototype demonstration can proceed")
    else:
        print("⚠️ Some systems need attention before full demonstration")
    
    print("="*60)
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
