#!/usr/bin/env python3
"""
Real-time Evolution Monitor
==========================

Monitor the ongoing evolution with live statistics and visualizations.
"""

import torch
import glob
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def find_latest_checkpoint_dir():
    """Find the most recent checkpoint directory"""
    dirs = glob.glob("checkpoints_*")
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def load_evolution_history(checkpoint_dir):
    """Load all checkpoints and extract history"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "gen_*.pt"))
    history = []
    
    for cp in checkpoints:
        try:
            gen = int(os.path.basename(cp).replace("gen_", "").replace(".pt", ""))
            data = torch.load(cp)
            history.append({
                'generation': gen,
                'best_fitness': data.get('best_fitness', 0),
                'avg_fitness': data.get('avg_fitness', 0),
                'timestamp': os.path.getmtime(cp)
            })
        except:
            pass
    
    return sorted(history, key=lambda x: x['generation'])

def analyze_best_genome(checkpoint_file):
    """Analyze the best genome from checkpoint"""
    data = torch.load(checkpoint_file)
    best_genome = data.get('best_genome', {})
    
    if not best_genome:
        return None
    
    genes = best_genome.get('genes', {})
    active_modules = [k for k, v in genes.items() if v]
    
    # Module categories
    categories = {
        'consciousness': ['ConsciousIntegrationHub', 'EmergentConsciousness', 'RecursiveSelfModel'],
        'reasoning': ['CounterfactualReasoner', 'InternalGoalGeneration', 'PredictiveWorldModel'],
        'learning': ['EmpowermentCalculator', 'QualityDiversitySearch', 'MetaEvolution'],
        'memory': ['WorkingMemory', 'SemanticRouter', 'ContextualUnderstanding'],
        'emergence': ['EmergenceEnhancer', 'DynamicConceptualField', 'ConceptualCompressor'],
        'other': []
    }
    
    # Categorize active modules
    module_stats = {cat: [] for cat in categories}
    for module in active_modules:
        categorized = False
        for cat, patterns in categories.items():
            if cat != 'other' and any(p in module for p in patterns):
                module_stats[cat].append(module)
                categorized = True
                break
        if not categorized:
            module_stats['other'].append(module)
    
    return {
        'total_active': len(active_modules),
        'categories': module_stats,
        'all_modules': active_modules
    }

def main():
    print("\n" + "="*80)
    print("🔬 EVOLUTION MONITOR")
    print("="*80)
    
    # Find checkpoint directory
    checkpoint_dir = find_latest_checkpoint_dir()
    if not checkpoint_dir:
        print("❌ No checkpoint directory found!")
        return
    
    print(f"📁 Monitoring: {checkpoint_dir}")
    
    # Get latest checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "gen_*.pt"))
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    latest = max(checkpoints, key=lambda x: int(os.path.basename(x).replace("gen_", "").replace(".pt", "")))
    
    # Load and analyze
    print("\n📊 LATEST STATUS:")
    print("-"*40)
    
    data = torch.load(latest)
    gen = data.get('generation', 0)
    best_fit = data.get('best_fitness', 0)
    avg_fit = data.get('avg_fitness', 0)
    
    print(f"Generation: {gen}")
    print(f"Best fitness: {best_fit:.4f}")
    print(f"Average fitness: {avg_fit:.4f}")
    
    # Analyze best genome
    genome_analysis = analyze_best_genome(latest)
    if genome_analysis:
        print(f"\n🧬 BEST GENOME ANALYSIS:")
        print(f"Active modules: {genome_analysis['total_active']}")
        print("\nModule distribution:")
        for cat, modules in genome_analysis['categories'].items():
            if modules:
                print(f"  {cat.capitalize()}: {len(modules)}")
                for m in modules[:3]:  # Show first 3
                    print(f"    - {m}")
                if len(modules) > 3:
                    print(f"    ... and {len(modules)-3} more")
    
    # Evolution history
    print("\n📈 EVOLUTION PROGRESS:")
    history = load_evolution_history(checkpoint_dir)
    
    if len(history) > 1:
        # Calculate rates
        time_diff = history[-1]['timestamp'] - history[0]['timestamp']
        gen_diff = history[-1]['generation'] - history[0]['generation']
        
        if time_diff > 0 and gen_diff > 0:
            gen_per_hour = (gen_diff / time_diff) * 3600
            print(f"Generations per hour: {gen_per_hour:.1f}")
            
            # Estimate time to reach certain milestones
            current_best = history[-1]['best_fitness']
            if current_best < 0.9:
                # Estimate based on recent improvement rate
                recent_history = history[-10:] if len(history) > 10 else history
                if len(recent_history) > 1:
                    fitness_gain = recent_history[-1]['best_fitness'] - recent_history[0]['best_fitness']
                    gens_for_gain = recent_history[-1]['generation'] - recent_history[0]['generation']
                    
                    if fitness_gain > 0 and gens_for_gain > 0:
                        rate = fitness_gain / gens_for_gain
                        gens_to_90 = (0.9 - current_best) / rate
                        hours_to_90 = gens_to_90 / gen_per_hour
                        print(f"Estimated time to 0.9 fitness: {hours_to_90:.1f} hours")
    
    # Plot progress
    if len(history) > 2:
        plt.figure(figsize=(10, 6))
        
        gens = [h['generation'] for h in history]
        best_fits = [h['best_fitness'] for h in history]
        avg_fits = [h['avg_fitness'] for h in history]
        
        plt.plot(gens, best_fits, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(gens, avg_fits, 'g--', label='Average Fitness', alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(checkpoint_dir, 'evolution_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: {plot_path}")
        plt.close()
    
    # Suggestions
    print("\n💡 SUGGESTIONS:")
    if best_fit < 0.7:
        print("- Consider increasing population size for more diversity")
        print("- Maybe enable meta-evolution for adaptive parameters")
    elif best_fit < 0.85:
        print("- Evolution is progressing well")
        print("- Consider saving best genomes for specialized tasks")
    else:
        print("- Excellent progress! Near optimal fitness")
        print("- Consider testing on specific tasks")
    
    # Next steps
    print("\n🎯 NEXT STEPS:")
    print("1. Test best genome: python test_best_genome.py")
    print("2. Analyze module interactions: python analyze_modules.py")
    print("3. Export for production: python export_best_mind.py")
    print("4. Continue evolution: python launch_agi_resume.py")

if __name__ == "__main__":
    main()