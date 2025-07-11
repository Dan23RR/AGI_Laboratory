#!/usr/bin/env python3
"""
Test Best Genome
===============

Load and test the best evolved genome on various tasks.
"""

import torch
import glob
import os
from datetime import datetime

from extended_genome import ExtendedGenome
from mind_factory_v2 import MindFactoryV2, MindConfig

def find_best_genome():
    """Find the best genome from all checkpoints"""
    dirs = glob.glob("checkpoints_*")
    if not dirs:
        return None, 0
    
    best_genome = None
    best_fitness = 0
    best_file = None
    
    for dir in dirs:
        checkpoints = glob.glob(os.path.join(dir, "gen_*.pt"))
        for cp in checkpoints:
            try:
                data = torch.load(cp)
                fitness = data.get('best_fitness', 0)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = data.get('best_genome')
                    best_file = cp
            except:
                pass
    
    return best_genome, best_fitness, best_file

def test_pattern_recognition(mind):
    """Test pattern recognition ability"""
    print("\n🔍 Testing Pattern Recognition...")
    
    # Create repeating pattern
    pattern = torch.randn(1, 512)
    results = []
    
    for i in range(5):
        # Show pattern
        output1 = mind(pattern)
        # Show noise
        noise = torch.randn(1, 512)
        output2 = mind(noise)
        # Show pattern again
        output3 = mind(pattern)
        
        # Check if it recognizes the pattern
        similarity = torch.cosine_similarity(
            output1['output'].flatten(),
            output3['output'].flatten(),
            dim=0
        ).item()
        
        results.append(similarity)
    
    avg_similarity = sum(results) / len(results)
    print(f"  Pattern recognition score: {avg_similarity:.3f}")
    return avg_similarity

def test_memory_persistence(mind):
    """Test memory persistence"""
    print("\n💾 Testing Memory Persistence...")
    
    # Store information
    info_vectors = [torch.randn(1, 512) for _ in range(3)]
    stored_outputs = []
    
    for vec in info_vectors:
        output = mind(vec)
        stored_outputs.append(output['output'])
    
    # Add noise
    for _ in range(5):
        mind(torch.randn(1, 512))
    
    # Try to recall
    recall_scores = []
    for i, vec in enumerate(info_vectors):
        output = mind(vec)
        similarity = torch.cosine_similarity(
            stored_outputs[i].flatten(),
            output['output'].flatten(),
            dim=0
        ).item()
        recall_scores.append(similarity)
    
    avg_recall = sum(recall_scores) / len(recall_scores)
    print(f"  Memory persistence score: {avg_recall:.3f}")
    return avg_recall

def test_adaptation(mind):
    """Test adaptation to new patterns"""
    print("\n🔄 Testing Adaptation...")
    
    # Initial random responses
    initial_responses = []
    test_inputs = [torch.randn(1, 512) for _ in range(5)]
    
    for inp in test_inputs:
        output = mind(inp)
        initial_responses.append(output['output'])
    
    # Create structured input (lower variance)
    structured_inputs = [torch.randn(1, 512) * 0.5 for _ in range(10)]
    for inp in structured_inputs:
        mind(inp)
    
    # Test if it adapted to lower variance
    final_responses = []
    for inp in test_inputs:
        output = mind(inp)
        final_responses.append(output['output'])
    
    # Check if responses became more structured
    initial_var = torch.stack(initial_responses).var().item()
    final_var = torch.stack(final_responses).var().item()
    
    adaptation_score = 1.0 - (final_var / (initial_var + 1e-6))
    print(f"  Adaptation score: {adaptation_score:.3f}")
    return adaptation_score

def test_coherence(mind):
    """Test coherence over time"""
    print("\n🎯 Testing Coherence...")
    
    coherence_scores = []
    prev_output = None
    
    for i in range(10):
        input_vec = torch.randn(1, 512) * (0.5 + i * 0.05)  # Gradually increasing variance
        output = mind(input_vec)
        
        if prev_output is not None:
            # Check coherence with previous output
            similarity = torch.cosine_similarity(
                prev_output['output'].flatten(),
                output['output'].flatten(),
                dim=0
            ).item()
            coherence_scores.append(similarity)
        
        prev_output = output
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    print(f"  Coherence score: {avg_coherence:.3f}")
    return avg_coherence

def main():
    print("\n" + "="*80)
    print("🧪 TESTING BEST EVOLVED GENOME")
    print("="*80)
    
    # Find best genome
    genome_dict, fitness, source_file = find_best_genome()
    
    if not genome_dict:
        print("❌ No genome found!")
        return
    
    print(f"📁 Source: {source_file}")
    print(f"🏆 Fitness: {fitness:.4f}")
    
    # Reconstruct genome
    genome = ExtendedGenome()
    if 'genes' in genome_dict:
        genome.genes = genome_dict['genes']
    if 'connections' in genome_dict:
        genome.connections = genome_dict['connections']
    if 'hyperparam_genes' in genome_dict:
        genome.hyperparam_genes = genome_dict['hyperparam_genes']
    
    # Show active modules
    active_modules = [k for k, v in genome.genes.items() if v]
    print(f"\n🧬 Active Modules ({len(active_modules)}):")
    for i, module in enumerate(active_modules):
        if i < 10:
            print(f"  - {module}")
        elif i == 10:
            print(f"  ... and {len(active_modules)-10} more")
            break
    
    # Create mind
    print("\n🧠 Creating mind from genome...")
    factory = MindFactoryV2()
    mind_config = MindConfig(
        hidden_dim=512,
        output_dim=256,
        memory_fraction=0.5
    )
    
    try:
        mind = factory.create_mind_from_genome(genome.to_dict(), mind_config)
        print("✅ Mind created successfully!")
        
        # Run tests
        print("\n" + "="*60)
        print("RUNNING CAPABILITY TESTS")
        print("="*60)
        
        scores = {}
        scores['pattern'] = test_pattern_recognition(mind)
        scores['memory'] = test_memory_persistence(mind)
        scores['adaptation'] = test_adaptation(mind)
        scores['coherence'] = test_coherence(mind)
        
        # Overall assessment
        print("\n" + "="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)
        
        overall_score = sum(scores.values()) / len(scores)
        print(f"\n🎯 Overall Score: {overall_score:.3f}")
        
        print("\n📊 Detailed Scores:")
        for test, score in scores.items():
            bar = "█" * int(score * 20)
            print(f"  {test.capitalize():12} [{bar:<20}] {score:.3f}")
        
        # Recommendations
        print("\n💡 ANALYSIS:")
        if overall_score > 0.7:
            print("✅ Excellent performance! This genome shows strong AGI capabilities.")
        elif overall_score > 0.5:
            print("🔶 Good performance with room for improvement.")
        else:
            print("⚠️  Performance needs improvement. Consider longer evolution.")
        
        # Specific feedback
        weakest = min(scores.items(), key=lambda x: x[1])
        strongest = max(scores.items(), key=lambda x: x[1])
        
        print(f"\n  Strongest ability: {strongest[0]} ({strongest[1]:.3f})")
        print(f"  Weakest ability: {weakest[0]} ({weakest[1]:.3f})")
        
        # Save test results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Best Genome Test Results\n")
            f.write(f"========================\n")
            f.write(f"Source: {source_file}\n")
            f.write(f"Fitness: {fitness:.4f}\n")
            f.write(f"Active Modules: {len(active_modules)}\n")
            f.write(f"\nScores:\n")
            for test, score in scores.items():
                f.write(f"  {test}: {score:.3f}\n")
            f.write(f"\nOverall: {overall_score:.3f}\n")
        
        print(f"\n📄 Results saved: {results_file}")
        
        # Cleanup
        mind.cleanup()
        
    except Exception as e:
        print(f"\n❌ Error testing genome: {e}")
        import traceback
        traceback.print_exc()
    finally:
        factory.cleanup()

if __name__ == "__main__":
    main()