#!/usr/bin/env python3
"""
Test just the framework initialization
"""

import sys
import os
sys.path.append('src')

from src.core.enhanced_stargate_transporter import EnhancedStargateTransporter, EnhancedTransporterConfig

def test_framework_init():
    print("Testing framework initialization...")
    
    try:
        # Base transporter configuration
        config = EnhancedTransporterConfig(
            R_payload=2.0,
            R_neck=0.08,
            L_corridor=2.0,
            delta_wall=0.05,
            v_conveyor=0.0,
            use_van_den_broeck=True,
            use_temporal_smearing=True,
            use_multi_bubble=True
        )
        print("SUCCESS: Config created")
        
        # Create base transporter
        transporter = EnhancedStargateTransporter(config)
        print("SUCCESS: Transporter created")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_framework_init()
