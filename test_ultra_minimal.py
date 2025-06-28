#!/usr/bin/env python3
"""
Ultra-minimal test of clean framework
"""

import sys
import os
sys.path.append('src')

def test_ultra_minimal():
    print("Ultra-minimal framework test...")
    
    try:
        # Import the framework class
        from demonstrate_lv_enhanced_framework_clean import LVEnhancedTransporterFramework
        
        print("Creating framework...")
        framework = LVEnhancedTransporterFramework()
        
        print("Initializing LV components...")
        framework.initialize_lv_components()
        
        print("SUCCESS: Framework and components initialized!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ultra_minimal()
