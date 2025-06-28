#!/usr/bin/env python3
"""
Quick test of all three workstreams to verify they're working properly.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_workstream_1():
    """Test Parameter Optimization Workstream"""
    print("🎯 Testing Workstream 1: Parameter Optimization...")
    try:
        from optimization.parameter_optimizer import TransporterOptimizer, OptimizationConfiguration
        from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        # Create basic config
        config = EnhancedTransporterConfig(
            payload_mass=75.0,
            R_neck=0.08,
            L_corridor=2.0,
            mu_polymer=0.15,
            alpha_polymer=2.0,
            bio_safety_threshold=1e-12
        )
        
        transporter = EnhancedStargateTransporter(config)
        
        opt_config = OptimizationConfiguration(
            optimize_polymer_params=True,
            optimize_geometry=False,  # Disable for quick test
            optimize_control_params=False,  # Disable for quick test
            max_iterations=5,  # Small number for testing
            tolerance=1e-6,
            safety_factor=10.0
        )
        
        optimizer = TransporterOptimizer(transporter, opt_config)
        print("   ✅ Parameter optimization framework ready")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_workstream_2():
    """Test Dynamic Corridor Simulation Workstream"""
    print("🌊 Testing Workstream 2: Dynamic Corridor Simulation...")
    try:
        from simulations.dynamic_corridor import DynamicCorridorSimulator, DynamicConfiguration
        from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        # Create basic config
        config = EnhancedTransporterConfig(
            payload_mass=75.0,
            R_neck=0.08,
            L_corridor=2.0,
            mu_polymer=0.15,
            alpha_polymer=2.0,
            bio_safety_threshold=1e-12
        )
        
        transporter = EnhancedStargateTransporter(config)
        
        sim_config = DynamicConfiguration(
            T_period=300.0,
            V_max=0.95,
            n_timesteps=10,  # Small number for testing
            resonance_analysis=False,  # Disable for quick test
            field_evolution_order=1
        )
        
        simulator = DynamicCorridorSimulator(transporter, sim_config)
        print("   ✅ Dynamic corridor simulation framework ready")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_workstream_3():
    """Test Casimir Negative Energy Integration Workstream"""
    print("⚛️ Testing Workstream 3: Casimir Integration...")
    try:
        from physics.casimir_integrator import CasimirNegativeEnergyIntegrator, CasimirConfiguration
        from core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        
        # Create basic config
        config = EnhancedTransporterConfig(
            payload_mass=75.0,
            R_neck=0.08,
            L_corridor=2.0,
            mu_polymer=0.15,
            alpha_polymer=2.0,
            bio_safety_threshold=1e-12
        )
        
        transporter = EnhancedStargateTransporter(config)
        
        casimir_config = CasimirConfiguration(
            plate_separation=1e-6,
            num_plates=10,  # Small number for testing
            plate_area=0.01,
            material_properties={'conductivity': 'perfect'},
            spatial_arrangement='parallel'
        )
        
        integrator = CasimirNegativeEnergyIntegrator(transporter, casimir_config)
        print("   ✅ Casimir negative energy integration framework ready")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all workstream tests"""
    print("🚀 Quick Workstream Integration Test")
    print("=" * 40)
    
    results = []
    results.append(test_workstream_1())
    results.append(test_workstream_2())
    results.append(test_workstream_3())
    
    success_count = sum(results)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Successful workstreams: {success_count}/3")
    
    if success_count == 3:
        print("   🎉 ALL WORKSTREAMS OPERATIONAL")
        return True
    else:
        print("   ⚠️ Some workstreams have issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
