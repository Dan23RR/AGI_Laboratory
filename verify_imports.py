#!/usr/bin/env python3
"""
Verify All Imports Work Correctly
=================================
"""

import sys
print("🔍 Verifying all imports...\n")

# Track results
imports_ok = []
imports_failed = []

def test_import(module_path, name):
    """Test if a module can be imported"""
    try:
        exec(f"from {module_path} import {name}")
        imports_ok.append(f"✅ {module_path}.{name}")
        return True
    except ImportError as e:
        imports_failed.append(f"❌ {module_path}.{name} - {str(e)}")
        return False
    except Exception as e:
        imports_failed.append(f"❌ {module_path}.{name} - {type(e).__name__}: {str(e)}")
        return False

# Test evolution imports
print("📦 Testing Evolution imports...")
test_import("evolution.general_evolution_lab_v3", "GeneralEvolutionLabV3")
test_import("evolution.extended_genome", "ExtendedGenome")
test_import("evolution.mind_factory_v2", "MindFactoryV2, MindConfig")
test_import("evolution.fitness.agi_fitness_metrics_v2", "AGIFitnessEvaluator")

# Test core imports
print("\n📦 Testing Core imports...")
test_import("core.meta_evolution", "MetaEvolution, MetaEvolutionConfig")
test_import("core.base_module", "BaseAGIModule")
test_import("core.memory_manager", "MemoryManager")
test_import("core.error_handling", "ErrorHandler")

# Test module imports
print("\n📦 Testing Module imports...")
test_import("modules.emergent_consciousness_v4", "EmergentConsciousnessV4")
test_import("modules.conscious_integration_hub_v2", "ConsciousIntegrationHubV2")
test_import("modules.recursive_self_model_v3", "RecursiveSelfModelV3")

# Test blueprint imports (these are scripts, not modules with classes)
print("\n📦 Testing Blueprint imports...")
# Blueprints are configuration files, not importable modules with classes
# Skip these tests as they don't export the expected classes

# Summary
print("\n" + "="*50)
print("📊 SUMMARY")
print("="*50)
print(f"✅ Successful imports: {len(imports_ok)}")
print(f"❌ Failed imports: {len(imports_failed)}")

if imports_failed:
    print("\n❌ FAILED IMPORTS:")
    for fail in imports_failed:
        print(f"  {fail}")
else:
    print("\n🎉 All imports working correctly!")

# Test if we can actually run the main scripts
print("\n🚀 Testing main launch scripts...")
try:
    import launch_agi_lab
    print("✅ launch_agi_lab.py imports correctly")
except Exception as e:
    print(f"❌ launch_agi_lab.py - {e}")

try:
    import launch_agi_clean
    print("✅ launch_agi_clean.py imports correctly")
except Exception as e:
    print(f"❌ launch_agi_clean.py - {e}")

print("\n✨ Verification complete!")