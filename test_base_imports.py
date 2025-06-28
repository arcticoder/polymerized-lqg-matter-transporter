#!/usr/bin/env python3
"""
Test Base Transporter Import
"""

import sys
import os
sys.path.append('src')

def test_base_imports():
    print("Testing base transporter imports...")
    
    try:
        from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig
        print("✅ Base transporter imports successful")
        
        # Test config creation
        config = EnhancedTransporterConfig(
            R_payload=2.0,
            R_neck=0.08,
            L_corridor=2.0,
            delta_wall=0.05,
            v_conveyor=0.0
        )
        print("✅ Transporter config created")
        
        # Test transporter creation
        transporter = EnhancedStargateTransporter(config)
        print("✅ Transporter created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing base transporter: {e}")
        return False

if __name__ == "__main__":
    success = test_base_imports()
