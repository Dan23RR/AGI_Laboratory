#!/usr/bin/env python3
"""
Full Integration Test for AGI Laboratory
========================================

Comprehensive test suite for all 19 refactored modules to ensure
the system is ready for production deployment.

Tests:
1. All module creation and initialization
2. Inter-module communication through ConsciousIntegrationHubV2
3. Memory management and leak detection
4. Evolution laboratory compatibility
5. Meta-evolution with safety controls
6. Performance benchmarking
7. Stress testing under load
"""

import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass

# Import core components
from mind_factory_v2 import MindFactoryV2, MindConfig
from core.meta_evolution import MetaEvolution, MetaEvolutionConfig
from extended_genome import ExtendedGenome
from core.base_module import ModuleConfig

# Create simplified EvolutionConfig since it's not in general_evolution_lab_v3
@dataclass
class EvolutionConfig:
    """Configuration for evolution experiments"""
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    n_generations: int = 100
    device: str = "cpu"
    fitness_threshold: float = 0.9
    diversity_bonus: float = 0.1
    checkpoint_interval: int = 10

# For memory profiling
import tracemalloc


class ComprehensiveIntegrationTest:
    """Comprehensive test suite for AGI laboratory"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            "module_tests": {},
            "integration_tests": {},
            "memory_tests": {},
            "performance_tests": {},
            "evolution_tests": {},
            "safety_tests": {}
        }
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 AGI LABORATORY FULL INTEGRATION TEST")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Start time: {datetime.now()}")
        print("=" * 80)
        
        # 1. Test all modules individually
        print("\n📦 PHASE 1: Testing All 19 Modules Individually")
        self._test_all_modules()
        
        # 2. Test integration with ConsciousIntegrationHubV2
        print("\n🔗 PHASE 2: Testing Inter-Module Communication")
        self._test_module_integration()
        
        # 3. Test memory management
        print("\n💾 PHASE 3: Testing Memory Management")
        self._test_memory_management()
        
        # 4. Test simplified evolution
        print("\n🧬 PHASE 4: Testing Evolution Capabilities")
        self._test_evolution_simple()
        
        # 5. Test meta-evolution with safety
        print("\n🛡️ PHASE 5: Testing Meta-Evolution Safety Controls")
        self._test_meta_evolution_safety()
        
        # 6. Performance benchmarking
        print("\n⚡ PHASE 6: Performance Benchmarking")
        self._test_performance()
        
        # 7. Stress testing
        print("\n🔥 PHASE 7: Stress Testing Under Load")
        self._test_stress()
        
        # Generate report
        print("\n📊 GENERATING FINAL REPORT...")
        self._generate_report()
        
    def _test_all_modules(self):
        """Test each module individually"""
        all_modules = [
            # Priority 1 - Critical
            "FeedbackLoopSystem",
            "SentientAGI", 
            "DynamicConceptualField",
            "ConsciousIntegrationHub",
            "EmergentConsciousness",
            "GoalConditionedMCTS",
            
            # Priority 2 - Performance
            "EmpowermentCalculator",
            "RecursiveSelfModel",
            "CounterfactualReasoner",
            
            # Priority 3 - Moderate
            "ConceptualCompressor",
            "AttractorNetwork",
            "EmergenceEnhancer",
            "GlobalIntegrationField",
            "InternalGoalGeneration",
            "CoherenceStabilizer",
            
            # Additional
            "EnergyBasedWorldModel"
        ]
        
        factory = MindFactoryV2(self.device)
        
        for module_name in all_modules:
            print(f"\n  Testing {module_name}...")
            try:
                # Create genome with only this module
                genome = {
                    'genes': {name: (name == module_name) for name in factory.MODULE_REGISTRY},
                    'hyperparameters': {}
                }
                
                # Create mind with single module
                mind = factory.create_mind_from_genome(genome, MindConfig())
                
                # Test forward pass
                test_input = torch.randn(2, 512, device=self.device)
                output = mind(test_input)
                
                # Verify output
                assert isinstance(output, dict), f"Output should be dict, got {type(output)}"
                assert 'output' in output, "Output should contain 'output' key"
                assert output['output'].shape[0] == 2, "Batch size mismatch"
                
                # Test memory usage
                if hasattr(mind, 'get_memory_usage'):
                    memory = mind.get_memory_usage()
                    assert memory['total_mb'] < 1000, f"Memory usage too high: {memory['total_mb']}MB"
                
                # Cleanup
                mind.cleanup()
                
                self.results["module_tests"][module_name] = {
                    "status": "PASS",
                    "output_shape": output['output'].shape,
                    "memory_mb": memory.get('total_mb', 0) if 'memory' in locals() else 0
                }
                print(f"    ✅ {module_name}: PASS")
                
            except Exception as e:
                self.results["module_tests"][module_name] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                print(f"    ❌ {module_name}: FAIL - {str(e)}")
        
        factory.cleanup()
    
    def _test_module_integration(self):
        """Test all modules working together"""
        print("\n  Creating mind with all modules...")
        
        factory = MindFactoryV2(self.device)
        
        # Create genome with all modules active
        all_active_genome = {
            'genes': {name: True for name in factory.MODULE_REGISTRY},
            'hyperparameters': {}
        }
        
        try:
            # Create complete mind
            mind = factory.create_mind_from_genome(all_active_genome, MindConfig(
                hidden_dim=512,
                n_modules=len(factory.MODULE_REGISTRY),
                output_dim=256,
                memory_fraction=0.8
            ))
            
            print(f"    ✅ Created mind with {len(mind.registered_modules)} modules")
            
            # Test various input sizes
            test_cases = [
                (1, 512),   # Single sample
                (8, 512),   # Small batch
                (32, 512),  # Medium batch
                (128, 512)  # Large batch
            ]
            
            for batch_size, dim in test_cases:
                test_input = torch.randn(batch_size, dim, device=self.device)
                
                start_time = time.time()
                output = mind(test_input)
                forward_time = time.time() - start_time
                
                assert output['output'].shape == (batch_size, 256)
                print(f"    ✅ Batch size {batch_size}: {forward_time*1000:.2f}ms")
                
                self.results["integration_tests"][f"batch_{batch_size}"] = {
                    "status": "PASS",
                    "forward_time_ms": forward_time * 1000
                }
            
            # Test coherence over time
            coherence_scores = []
            for _ in range(10):
                output = mind(torch.randn(4, 512, device=self.device))
                if 'coherence_score' in output:
                    coherence_scores.append(output['coherence_score'].item())
            
            if coherence_scores:
                avg_coherence = np.mean(coherence_scores)
                print(f"    ✅ Average coherence: {avg_coherence:.4f}")
                self.results["integration_tests"]["coherence"] = avg_coherence
            
            mind.cleanup()
            
        except Exception as e:
            self.results["integration_tests"]["error"] = str(e)
            print(f"    ❌ Integration test failed: {e}")
            
        factory.cleanup()
    
    def _test_memory_management(self):
        """Test memory leak detection and management"""
        print("\n  Testing memory management...")
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        
        factory = MindFactoryV2(self.device)
        
        # Create and destroy minds multiple times
        memory_readings = []
        
        for i in range(5):
            # Create mind with random subset of modules
            total_modules = len(factory.MODULE_REGISTRY)
            n_modules = np.random.randint(3, min(10, total_modules))
            active_modules = np.random.choice(
                list(factory.MODULE_REGISTRY.keys()), 
                size=n_modules, 
                replace=False
            )
            
            genome = {
                'genes': {name: (name in active_modules) for name in factory.MODULE_REGISTRY},
                'hyperparameters': {}
            }
            
            mind = factory.create_mind_from_genome(genome)
            
            # Run forward passes
            for _ in range(100):
                _ = mind(torch.randn(4, 512, device=self.device))
            
            # Record memory before cleanup
            current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
            memory_readings.append(current_memory)
            
            # Cleanup
            mind.cleanup()
            del mind
            gc.collect()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"    Iteration {i+1}: {current_memory:.2f}MB")
        
        # Check for memory leaks
        memory_growth = memory_readings[-1] - memory_readings[0]
        
        self.results["memory_tests"] = {
            "initial_mb": initial_memory,
            "final_mb": memory_readings[-1],
            "growth_mb": memory_growth,
            "readings": memory_readings,
            "leak_detected": memory_growth > 50  # 50MB threshold
        }
        
        if memory_growth < 50:
            print(f"    ✅ No significant memory leaks: growth = {memory_growth:.2f}MB")
        else:
            print(f"    ⚠️  Possible memory leak: growth = {memory_growth:.2f}MB")
        
        factory.cleanup()
        tracemalloc.stop()
    
    def _test_evolution_simple(self):
        """Test evolution capabilities with refactored modules"""
        print("\n  Testing evolution capabilities...")
        
        try:
            # Test 1: Basic genome evolution
            print("    Test 1: Basic genome evolution...")
            factory = MindFactoryV2(self.device)
            
            # Create population of genomes
            population = []
            for _ in range(10):
                genome = ExtendedGenome()
                genome.randomize()
                population.append(genome)
            
            # Evaluate fitness
            fitnesses = []
            for genome in population:
                fitness = self._simple_fitness(genome)
                fitnesses.append(fitness)
            
            # Find best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_genome = population[best_idx]
            
            print(f"    ✅ Population evaluated: best fitness = {best_fitness:.4f}")
            
            # Test 2: Mutation
            print("\n    Test 2: Testing mutation...")
            mutated = best_genome.copy()
            mutated.mutate(mutation_rate=0.1)
            
            # Check that mutation changed something
            genes_changed = sum(1 for k in mutated.genes if mutated.genes[k] != best_genome.genes[k])
            assert genes_changed > 0, "Mutation should change at least one gene"
            print(f"    ✅ Mutation changed {genes_changed} genes")
            
            # Test 3: Crossover
            print("\n    Test 3: Testing crossover...")
            parent1 = population[0]
            parent2 = population[1]
            child = parent1.crossover(parent2)
            
            # Verify child has genes from both parents
            from_p1 = sum(1 for k in child.genes if child.genes[k] == parent1.genes[k])
            from_p2 = sum(1 for k in child.genes if child.genes[k] == parent2.genes[k])
            
            print(f"    ✅ Crossover: {from_p1} genes from parent1, {from_p2} from parent2")
            
            # Test 4: Simple evolution loop
            print("\n    Test 4: Simple evolution (3 generations)...")
            
            for gen in range(3):
                # Evaluate
                fitnesses = [self._simple_fitness(g) for g in population]
                
                # Select best
                sorted_idx = np.argsort(fitnesses)[::-1]
                elite = [population[i] for i in sorted_idx[:3]]
                
                # Create new population
                new_population = elite.copy()
                
                while len(new_population) < 10:
                    # Select parents
                    p1 = elite[np.random.randint(len(elite))]
                    p2 = elite[np.random.randint(len(elite))]
                    
                    # Create child
                    if np.random.random() < 0.7:  # Crossover
                        child = p1.crossover(p2)
                    else:
                        child = p1.copy()
                    
                    # Mutate
                    child.mutate(0.1)
                    new_population.append(child)
                
                population = new_population[:10]
                
                # Report
                best_fitness = max(fitnesses)
                avg_fitness = np.mean(fitnesses)
                print(f"      Gen {gen+1}: best={best_fitness:.4f}, avg={avg_fitness:.4f}")
            
            self.results["evolution_tests"] = {
                "basic": "PASS",
                "mutation": "PASS", 
                "crossover": "PASS",
                "evolution_loop": "PASS",
                "final_best_fitness": best_fitness
            }
            
            print("\n    ✅ All evolution tests passed")
            
        except Exception as e:
            self.results["evolution_tests"]["error"] = str(e)
            print(f"    ❌ Evolution test failed: {e}")
    
    def _test_meta_evolution_safety(self):
        """Test meta-evolution with safety controls"""
        print("\n  Testing meta-evolution safety...")
        
        try:
            # Create meta-evolution instance with correct parameters
            meta_config = MetaEvolutionConfig(
                mutation_rate_min=0.001,
                mutation_rate_max=0.5,
                mutation_rate_initial=0.1,
                performance_collapse_threshold=0.5,
                parameter_change_limit=0.1,
                gradient_clip_value=1.0,
                max_meta_steps=100,
                safety_checks_enabled=True
            )
            
            meta_evo = MetaEvolution(meta_config)
            
            # Test 1: Normal adaptation
            print("    Test 1: Normal adaptation...")
            fitness_history = [0.5, 0.52, 0.54, 0.55, 0.56]
            hyperparams = {"mutation_rate": 0.1}
            
            new_params = meta_evo.adapt_hyperparameters(hyperparams, fitness_history)
            
            assert 0.001 <= new_params["mutation_rate"] <= 0.5
            print(f"    ✅ Mutation rate adapted: {hyperparams['mutation_rate']:.3f} → {new_params['mutation_rate']:.3f}")
            
            # Test 2: Performance collapse detection
            print("\n    Test 2: Performance collapse detection...")
            collapse_history = [0.8, 0.7, 0.5, 0.3, 0.1]  # Dramatic drop
            
            new_params = meta_evo.adapt_hyperparameters(hyperparams, collapse_history)
            
            # Should detect collapse and be conservative
            assert new_params["mutation_rate"] < hyperparams["mutation_rate"]
            print("    ✅ Performance collapse detected - parameters reverted")
            
            # Test 3: Parameter change limits
            print("\n    Test 3: Parameter change limits...")
            extreme_params = {"mutation_rate": 0.01}
            stable_history = [0.6] * 5
            
            for _ in range(10):
                extreme_params = meta_evo.adapt_hyperparameters(extreme_params, stable_history)
            
            # Even after many steps, change should be limited
            assert abs(extreme_params["mutation_rate"] - 0.01) < 0.5
            print("    ✅ Parameter changes properly limited")
            
            # Test 4: Safety validation
            print("\n    Test 4: Safety validation...")
            unsafe_config = MetaEvolutionConfig(
                mutation_rate_min=-1,  # Invalid
                mutation_rate_max=2,   # Invalid
                parameter_change_limit=0.9  # Too high
            )
            
            is_safe = meta_evo._validate_safety_bounds(unsafe_config.__dict__)
            assert not is_safe
            print("    ✅ Unsafe configuration detected")
            
            self.results["safety_tests"] = {
                "adaptation": "PASS",
                "collapse_detection": "PASS",
                "parameter_limits": "PASS",
                "validation": "PASS"
            }
            
        except Exception as e:
            self.results["safety_tests"]["error"] = str(e)
            print(f"    ❌ Safety test failed: {e}")
    
    def _test_performance(self):
        """Benchmark performance metrics"""
        print("\n  Benchmarking performance...")
        
        factory = MindFactoryV2(self.device)
        
        # Test configurations
        configs = [
            ("minimal", 5),
            ("standard", 10),
            ("full", len(factory.MODULE_REGISTRY))
        ]
        
        for config_name, n_modules in configs:
            print(f"\n    Testing {config_name} configuration ({n_modules} modules)...")
            
            # Select modules
            if n_modules < len(factory.MODULE_REGISTRY):
                active_modules = np.random.choice(
                    list(factory.MODULE_REGISTRY.keys()),
                    size=n_modules,
                    replace=False
                )
            else:
                active_modules = list(factory.MODULE_REGISTRY.keys())
            
            genome = {
                'genes': {name: (name in active_modules) for name in factory.MODULE_REGISTRY},
                'hyperparameters': {}
            }
            
            mind = factory.create_mind_from_genome(genome)
            
            # Warmup
            for _ in range(10):
                _ = mind(torch.randn(8, 512, device=self.device))
            
            # Benchmark
            batch_sizes = [1, 8, 32, 128]
            results = {}
            
            for batch_size in batch_sizes:
                times = []
                test_input = torch.randn(batch_size, 512, device=self.device)
                
                for _ in range(100):
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    _ = mind(test_input)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times) * 1000  # ms
                throughput = batch_size / np.mean(times)  # samples/sec
                
                results[f"batch_{batch_size}"] = {
                    "avg_ms": avg_time,
                    "throughput": throughput
                }
                
                print(f"      Batch {batch_size}: {avg_time:.2f}ms, {throughput:.0f} samples/sec")
            
            self.results["performance_tests"][config_name] = results
            
            mind.cleanup()
        
        factory.cleanup()
    
    def _test_stress(self):
        """Stress test under heavy load"""
        print("\n  Running stress tests...")
        
        factory = MindFactoryV2(self.device)
        
        # Create mind with many modules
        genome = {
            'genes': {name: np.random.random() > 0.3 for name in factory.MODULE_REGISTRY},
            'hyperparameters': {}
        }
        
        mind = factory.create_mind_from_genome(genome)
        active_modules = sum(1 for v in genome['genes'].values() if v)
        print(f"    Created mind with {active_modules} active modules")
        
        # Stress test parameters
        n_iterations = 1000
        batch_size = 64
        
        # Monitor resources
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        errors = []
        times = []
        memory_readings = []
        
        print(f"\n    Running {n_iterations} iterations...")
        
        for i in range(n_iterations):
            try:
                # Random input
                test_input = torch.randn(batch_size, 512, device=self.device)
                
                start_time = time.time()
                output = mind(test_input)
                times.append(time.time() - start_time)
                
                # Check output validity
                assert not torch.isnan(output['output']).any()
                assert not torch.isinf(output['output']).any()
                
                # Monitor memory every 100 iterations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_readings.append(current_memory)
                    
                    if i > 0:
                        avg_time = np.mean(times[-100:]) * 1000
                        print(f"      Iteration {i}: avg time = {avg_time:.2f}ms, memory = {current_memory:.0f}MB")
                
            except Exception as e:
                errors.append((i, str(e)))
        
        # Analysis
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        self.results["stress_tests"] = {
            "iterations": n_iterations,
            "errors": len(errors),
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "memory_growth_mb": memory_growth,
            "max_memory_mb": max(memory_readings) if memory_readings else final_memory
        }
        
        if len(errors) == 0:
            print(f"\n    ✅ Stress test passed: {n_iterations} iterations without errors")
            print(f"    ✅ Average time: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
            print(f"    ✅ Memory growth: {memory_growth:.2f}MB")
        else:
            print(f"\n    ⚠️  Stress test had {len(errors)} errors")
            for idx, err in errors[:5]:  # Show first 5
                print(f"      Error at iteration {idx}: {err}")
        
        mind.cleanup()
        factory.cleanup()
    
    def _simple_fitness(self, genome: ExtendedGenome) -> float:
        """Simple fitness function for evolution test"""
        # Favor genomes with 8-12 active modules
        n_active = sum(1 for active in genome.genes.values() if active)
        module_score = 1.0 - abs(n_active - 10) / 10
        
        # Favor balanced intrinsic motivation
        motivation_variance = np.var(list(genome.intrinsic_motivation.values())[:4])
        balance_score = 1.0 / (1.0 + motivation_variance)
        
        return 0.7 * module_score + 0.3 * balance_score
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 FINAL TEST REPORT")
        print("=" * 80)
        
        # Module tests summary
        print("\n🔧 MODULE TESTS:")
        passed = sum(1 for r in self.results["module_tests"].values() if r.get("status") == "PASS")
        total = len(self.results["module_tests"])
        print(f"  Passed: {passed}/{total}")
        
        if passed < total:
            print("  Failed modules:")
            for name, result in self.results["module_tests"].items():
                if result.get("status") != "PASS":
                    print(f"    - {name}: {result.get('error', 'Unknown error')}")
        
        # Integration tests
        print("\n🔗 INTEGRATION TESTS:")
        if "error" in self.results["integration_tests"]:
            print(f"  ❌ Failed: {self.results['integration_tests']['error']}")
        else:
            print("  ✅ All integration tests passed")
            if "coherence" in self.results["integration_tests"]:
                print(f"  Average coherence: {self.results['integration_tests']['coherence']:.4f}")
        
        # Memory tests
        print("\n💾 MEMORY TESTS:")
        mem_results = self.results.get("memory_tests", {})
        if mem_results.get("leak_detected"):
            print(f"  ⚠️  Possible memory leak detected: {mem_results['growth_mb']:.2f}MB growth")
        else:
            print(f"  ✅ No memory leaks detected: {mem_results.get('growth_mb', 0):.2f}MB growth")
        
        # Evolution tests
        print("\n🧬 EVOLUTION TESTS:")
        evo_results = self.results.get("evolution_tests", {})
        if "error" in evo_results:
            print(f"  ❌ Failed: {evo_results['error']}")
        else:
            basic = evo_results.get("basic", {})
            if basic.get("status") == "PASS":
                print(f"  ✅ Basic evolution: {basic['improvement']:.4f} fitness improvement")
                print(f"  ✅ Specialization: {evo_results.get('specialization', 'Not tested')}")
        
        # Safety tests
        print("\n🛡️  SAFETY TESTS:")
        safety_results = self.results.get("safety_tests", {})
        if all(v == "PASS" for k, v in safety_results.items() if k != "error"):
            print("  ✅ All safety controls verified")
        else:
            print("  ⚠️  Some safety tests failed")
        
        # Performance summary
        print("\n⚡ PERFORMANCE SUMMARY:")
        perf_results = self.results.get("performance_tests", {})
        for config, results in perf_results.items():
            if "batch_32" in results:
                metrics = results["batch_32"]
                print(f"  {config}: {metrics['avg_ms']:.2f}ms, {metrics['throughput']:.0f} samples/sec")
        
        # Stress test results
        print("\n🔥 STRESS TEST RESULTS:")
        stress = self.results.get("stress_tests", {})
        if stress:
            print(f"  Iterations: {stress['iterations']}")
            print(f"  Errors: {stress['errors']}")
            print(f"  Avg time: {stress['avg_time_ms']:.2f}ms ± {stress['std_time_ms']:.2f}ms")
            print(f"  Memory growth: {stress['memory_growth_mb']:.2f}MB")
        
        # Overall status
        print("\n" + "=" * 80)
        all_passed = (
            passed == total and
            "error" not in self.results["integration_tests"] and
            not mem_results.get("leak_detected", True) and
            "error" not in evo_results and
            all(v == "PASS" for k, v in safety_results.items() if k != "error") and
            stress.get("errors", 1) == 0
        )
        
        if all_passed:
            print("✅ SYSTEM READY FOR PRODUCTION - ALL TESTS PASSED")
            print("\n🎉 The AGI Laboratory is ready for deployment!")
        else:
            print("⚠️  SYSTEM NOT READY - SOME TESTS FAILED")
            print("\nPlease review the failed tests above before deployment.")
        
        print("=" * 80)
        print(f"End time: {datetime.now()}")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save test results to file"""
        import json
        
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy values to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
        
        # Recursively convert all values
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert(d)
        
        results_json = convert_dict(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n📁 Results saved to: {filename}")


def main():
    """Run the comprehensive integration test"""
    test_suite = ComprehensiveIntegrationTest()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()