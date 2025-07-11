#!/usr/bin/env python3
"""
Module Interaction Analyzer
==========================

Analyze which modules work well together in evolved genomes.
"""

import torch
import glob
import os
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_genomes():
    """Load all genomes from all checkpoints"""
    all_genomes = []
    
    dirs = glob.glob("checkpoints_*")
    for dir in dirs:
        checkpoints = glob.glob(os.path.join(dir, "gen_*.pt"))
        for cp in checkpoints:
            try:
                data = torch.load(cp)
                if 'population' in data:
                    for genome_dict in data['population']:
                        if 'genes' in genome_dict:
                            all_genomes.append({
                                'genes': genome_dict['genes'],
                                'fitness': data.get('avg_fitness', 0),
                                'generation': data.get('generation', 0)
                            })
                elif 'best_genome' in data and data['best_genome']:
                    all_genomes.append({
                        'genes': data['best_genome']['genes'],
                        'fitness': data.get('best_fitness', 0),
                        'generation': data.get('generation', 0)
                    })
            except:
                pass
    
    return all_genomes

def analyze_module_frequency(genomes):
    """Analyze how often each module is active"""
    module_counts = Counter()
    module_fitness = defaultdict(list)
    
    for genome in genomes:
        active_modules = [k for k, v in genome['genes'].items() if v]
        for module in active_modules:
            module_counts[module] += 1
            module_fitness[module].append(genome['fitness'])
    
    # Calculate average fitness when module is active
    module_avg_fitness = {}
    for module, fitnesses in module_fitness.items():
        module_avg_fitness[module] = sum(fitnesses) / len(fitnesses)
    
    return module_counts, module_avg_fitness

def analyze_module_pairs(genomes):
    """Analyze which module pairs work well together"""
    pair_counts = Counter()
    pair_fitness = defaultdict(list)
    
    for genome in genomes:
        active_modules = sorted([k for k, v in genome['genes'].items() if v])
        
        # Count all pairs
        for i in range(len(active_modules)):
            for j in range(i + 1, len(active_modules)):
                pair = (active_modules[i], active_modules[j])
                pair_counts[pair] += 1
                pair_fitness[pair].append(genome['fitness'])
    
    # Calculate average fitness for pairs
    pair_avg_fitness = {}
    for pair, fitnesses in pair_fitness.items():
        if len(fitnesses) > 5:  # Only consider pairs that appear often
            pair_avg_fitness[pair] = sum(fitnesses) / len(fitnesses)
    
    return pair_counts, pair_avg_fitness

def analyze_evolution_trends(genomes):
    """Analyze how module usage changes over generations"""
    generation_modules = defaultdict(lambda: defaultdict(int))
    
    for genome in genomes:
        gen = genome['generation']
        active_modules = [k for k, v in genome['genes'].items() if v]
        for module in active_modules:
            generation_modules[gen][module] += 1
    
    return generation_modules

def main():
    print("\n" + "="*80)
    print("🔬 MODULE INTERACTION ANALYSIS")
    print("="*80)
    
    # Load all genomes
    print("\n📊 Loading genome data...")
    genomes = load_all_genomes()
    print(f"✅ Loaded {len(genomes)} genomes")
    
    if len(genomes) < 10:
        print("⚠️  Not enough data for meaningful analysis")
        return
    
    # Filter to only high-fitness genomes
    high_fitness_genomes = [g for g in genomes if g['fitness'] > 0.5]
    print(f"📈 High-fitness genomes: {len(high_fitness_genomes)}")
    
    # Analyze module frequency
    print("\n🧬 MODULE FREQUENCY ANALYSIS")
    print("-"*40)
    module_counts, module_avg_fitness = analyze_module_frequency(high_fitness_genomes)
    
    # Sort by effectiveness (count * avg_fitness)
    module_effectiveness = {}
    for module in module_counts:
        effectiveness = module_counts[module] * module_avg_fitness[module]
        module_effectiveness[module] = effectiveness
    
    top_modules = sorted(module_effectiveness.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("\nMost Effective Modules:")
    print(f"{'Module':<30} {'Count':>8} {'Avg Fitness':>12} {'Effectiveness':>12}")
    print("-"*65)
    for module, effectiveness in top_modules:
        count = module_counts[module]
        avg_fit = module_avg_fitness[module]
        print(f"{module:<30} {count:>8} {avg_fit:>12.3f} {effectiveness:>12.1f}")
    
    # Analyze module pairs
    print("\n🔗 MODULE SYNERGY ANALYSIS")
    print("-"*40)
    pair_counts, pair_avg_fitness = analyze_module_pairs(high_fitness_genomes)
    
    # Find best synergies
    top_pairs = sorted(pair_avg_fitness.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nBest Module Synergies:")
    print(f"{'Module 1':<25} {'Module 2':<25} {'Avg Fitness':>12} {'Count':>8}")
    print("-"*75)
    for (m1, m2), avg_fit in top_pairs:
        count = pair_counts[(m1, m2)]
        print(f"{m1:<25} {m2:<25} {avg_fit:>12.3f} {count:>8}")
    
    # Core modules analysis
    print("\n🎯 CORE MODULE IDENTIFICATION")
    print("-"*40)
    
    # Find modules that appear in >80% of high-fitness genomes
    total_high_fitness = len(high_fitness_genomes)
    core_threshold = 0.8
    
    core_modules = []
    for module, count in module_counts.items():
        if count / total_high_fitness > core_threshold:
            core_modules.append(module)
    
    if core_modules:
        print(f"\nCore modules (appear in >{core_threshold*100:.0f}% of successful genomes):")
        for module in core_modules:
            percentage = (module_counts[module] / total_high_fitness) * 100
            print(f"  - {module} ({percentage:.1f}%)")
    else:
        print("\nNo single module appears in >80% of successful genomes")
        print("This suggests diverse strategies are being evolved")
    
    # Module categories
    print("\n📊 MODULE CATEGORY DISTRIBUTION")
    print("-"*40)
    
    categories = {
        'Consciousness': ['ConsciousIntegrationHub', 'EmergentConsciousness', 'RecursiveSelfModel'],
        'Reasoning': ['CounterfactualReasoner', 'InternalGoalGeneration', 'PredictiveWorldModel'],
        'Learning': ['EmpowermentCalculator', 'QualityDiversitySearch', 'MetaEvolution'],
        'Memory': ['WorkingMemory', 'SemanticRouter', 'ContextualUnderstanding'],
        'Emergence': ['EmergenceEnhancer', 'DynamicConceptualField', 'ConceptualCompressor']
    }
    
    category_stats = {cat: {'count': 0, 'total_fitness': 0} for cat in categories}
    
    for genome in high_fitness_genomes:
        active_modules = [k for k, v in genome['genes'].items() if v]
        for module in active_modules:
            for cat, patterns in categories.items():
                if any(p in module for p in patterns):
                    category_stats[cat]['count'] += 1
                    category_stats[cat]['total_fitness'] += genome['fitness']
                    break
    
    print("\nCategory Usage in High-Fitness Genomes:")
    for cat, stats in category_stats.items():
        if stats['count'] > 0:
            avg_contribution = stats['total_fitness'] / stats['count']
            print(f"  {cat}: {stats['count']} uses, avg fitness contribution: {avg_contribution:.3f}")
    
    # Evolution trends
    print("\n📈 EVOLUTION TRENDS")
    print("-"*40)
    generation_modules = analyze_evolution_trends(genomes)
    
    if generation_modules:
        # Find modules that increase/decrease over time
        all_modules = set()
        for gen_data in generation_modules.values():
            all_modules.update(gen_data.keys())
        
        module_trends = {}
        generations = sorted(generation_modules.keys())
        
        if len(generations) > 5:
            for module in all_modules:
                early_count = sum(generation_modules[g].get(module, 0) for g in generations[:3])
                late_count = sum(generation_modules[g].get(module, 0) for g in generations[-3:])
                
                if early_count > 0 or late_count > 0:
                    trend = (late_count - early_count) / max(early_count, 1)
                    module_trends[module] = trend
            
            # Show increasing/decreasing modules
            increasing = sorted([(m, t) for m, t in module_trends.items() if t > 0.5], 
                              key=lambda x: x[1], reverse=True)[:5]
            decreasing = sorted([(m, t) for m, t in module_trends.items() if t < -0.5], 
                              key=lambda x: x[1])[:5]
            
            if increasing:
                print("\nModules increasing in usage:")
                for module, trend in increasing:
                    print(f"  ↗️ {module} (+{trend:.1f}x)")
            
            if decreasing:
                print("\nModules decreasing in usage:")
                for module, trend in decreasing:
                    print(f"  ↘️ {module} ({trend:.1f}x)")
    
    # Recommendations
    print("\n💡 INSIGHTS & RECOMMENDATIONS")
    print("-"*40)
    
    if core_modules:
        print(f"✅ Core modules identified: {', '.join(core_modules[:3])}")
        print("   These should be prioritized in future experiments")
    
    if top_pairs:
        best_pair = top_pairs[0]
        print(f"\n✅ Best synergy: {best_pair[0][0]} + {best_pair[0][1]}")
        print(f"   Average fitness when combined: {best_pair[1]:.3f}")
    
    # Module diversity score
    module_counts_list = [len([k for k, v in g['genes'].items() if v]) 
                         for g in high_fitness_genomes]
    avg_modules = sum(module_counts_list) / len(module_counts_list)
    print(f"\n📊 Average modules per genome: {avg_modules:.1f}")
    
    if avg_modules < 5:
        print("   ⚠️  Low diversity - consider increasing mutation rate")
    elif avg_modules > 15:
        print("   ⚠️  High complexity - may be overfitting")
    else:
        print("   ✅ Good balance of complexity")
    
    # Save analysis
    analysis_file = f"module_analysis_{len(genomes)}_genomes.txt"
    with open(analysis_file, 'w') as f:
        f.write("MODULE INTERACTION ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(f"Total genomes analyzed: {len(genomes)}\n")
        f.write(f"High-fitness genomes: {len(high_fitness_genomes)}\n\n")
        
        f.write("Top Modules:\n")
        for module, eff in top_modules[:10]:
            f.write(f"  {module}: {eff:.1f}\n")
        
        f.write("\nBest Synergies:\n")
        for (m1, m2), fit in top_pairs[:5]:
            f.write(f"  {m1} + {m2}: {fit:.3f}\n")
    
    print(f"\n📄 Full analysis saved: {analysis_file}")

if __name__ == "__main__":
    main()