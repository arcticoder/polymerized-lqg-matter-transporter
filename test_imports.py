#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

def test_imports():
    print("Testing individual imports...")
    
    # Test 1: JAX
    try:
        import jax
        print("✅ JAX import successful")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    # Test 2: JAX numpy
    try:
        import jax.numpy as jnp
        print("✅ JAX numpy import successful")
    except ImportError as e:
        print(f"❌ JAX numpy import failed: {e}")
        return False
    
    # Test 3: SME Parameters
    try:
        from src.lorentz_violation.modified_einstein_solver import SMEParameters
        print("✅ SME Parameters import successful")
    except Exception as e:
        print(f"❌ SME Parameters import failed: {e}")
        return False
    
    # Test 4: Ghost Config
    try:
        from src.physics.ghost_scalar_eft import GhostScalarConfig
        print("✅ Ghost Config import successful")
    except Exception as e:
        print(f"❌ Ghost Config import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    print(f"\nOverall success: {success}")
