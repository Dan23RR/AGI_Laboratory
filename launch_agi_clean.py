#!/usr/bin/env python3
"""
AGI Laboratory Launch Script - Clean Output Version
==================================================

Shows only essential evolution progress.
"""

import torch
import argparse
import logging
from datetime import datetime
import os
import sys

# Import core components
from evolution.extended_genome import ExtendedGenome
from evolution.mind_factory_v2 import MindFactoryV2, MindConfig

# Suppress verbose logging
logging.getLogger('core.memory_manager').setLevel(logging.ERROR)
logging.getLogger('core.conscious_integration_hub_v2').setLevel(logging.ERROR)
logging.getLogger('core.base_module').setLevel(logging.ERROR)
logging.getLogger('core.error_handling').setLevel(logging.ERROR)
logging.getLogger('general_evolution_lab_v3').setLevel(logging.ERROR)

# Setup clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f'evolution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_fitness_function(target_task="general"):
    """Create fitness function based on target task"""
    
    def general_fitness(genome: ExtendedGenome) -> float:
        """General AGI fitness: balance of capabilities"""
        factory = MindFactoryV2()
        
        # Suppress factory logging
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            mind_config = MindConfig(
                hidden_dim=512,
                output_dim=256,
                memory_fraction=0.5
            )
            mind = factory.create_mind_from_genome(genome.to_dict(), mind_config)
            
            fitness_scores = []
            
            # Test basic functionality
            test_input = torch.randn(8, 512)
            if torch.cuda.is_available() and test_input.device.type == 'cpu':
                test_input = test_input.cuda()
                
            output = mind(test_input)
            pattern_score = 1.0 - output['output'].std().item()
            fitness_scores.append(pattern_score)
            
            # Check coherence
            coherence_scores = []
            for _ in range(3):
                test_data = torch.randn(4, 512)
                if torch.cuda.is_available() and test_data.device.type == 'cpu':
                    test_data = test_data.cuda()
                output = mind(test_data)
                if 'coherence_score' in output:
                    coherence_scores.append(output['coherence_score'].item())
            
            if coherence_scores:
                fitness_scores.append(sum(coherence_scores) / len(coherence_scores))
            
            # Memory efficiency
            if hasattr(mind, 'get_memory_usage'):
                memory = mind.get_memory_usage()
                memory_score = 1.0 / (1.0 + memory['total_mb'] / 100)
                fitness_scores.append(memory_score)
            
            # Module diversity
            active_modules = sum(1 for v in genome.genes.values() if v)
            diversity_score = min(active_modules / 10, 1.0) * 0.5
            fitness_scores.append(diversity_score)
            
            mind.cleanup()
            
            return sum(fitness_scores) / len(fitness_scores)
            
        except Exception as e:
            return 0.0
        finally:
            logging.getLogger().setLevel(old_level)
            factory.cleanup()
    
    return general_fitness


def print_header():
    """Print clean header"""
    print("\n" + "="*80)
    print("🧬 AGI EVOLUTION LABORATORY - CLEAN OUTPUT MODE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def print_generation_summary(gen, best_fitness, avg_fitness, best_genome):
    """Print clean generation summary"""
    active_modules = [k for k, v in best_genome.genes.items() if v] if best_genome else []
    
    print(f"\n🧪 Generation {gen:4d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Modules: {len(active_modules)}")
    
    if gen % 10 == 0 and active_modules:
        print(f"   Active modules: {', '.join(active_modules[:5])}{'...' if len(active_modules) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(description='Launch AGI Evolution - Clean Output')
    
    parser.add_argument('--generations', type=int, default=999999)
    parser.add_argument('--population', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--summary-interval', type=int, default=5,
                       help='Show summary every N generations')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    print(f"\nConfiguration:")
    print(f"  Population: {args.population}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Summary interval: {args.summary_interval}")
    print("\nPress Ctrl+C to stop\n")
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (slower)")
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get fitness function
    fitness_fn = create_fitness_function()
    
    # Evolution parameters
    mutation_rate = 0.1
    crossover_rate = 0.7
    elite_size = max(2, args.population // 10)
    
    try:
        # Create initial population
        print("\n📊 Creating initial population...")
        population = []
        for i in range(args.population):
            genome = ExtendedGenome()
            if i < args.population // 4:
                # Ensure some key modules
                for module in ["ConsciousIntegrationHub", "EmergentConsciousness"]:
                    if module in genome.genes:
                        genome.genes[module] = True
            population.append(genome)
        
        print("✅ Population created")
        print("\n" + "-"*80)
        print("Gen | Best    | Average | Modules | Status")
        print("-"*80)
        
        # Evolution loop
        best_ever_fitness = 0.0
        best_ever_genome = None
        generation = 0
        
        while generation < args.generations:
            generation += 1
            
            # Evaluate population (suppress logging)
            logging.getLogger().setLevel(logging.ERROR)
            fitnesses = []
            for i, genome in enumerate(population):
                fitness = fitness_fn(genome)
                fitnesses.append(fitness)
                
                # Progress indicator every 20 genomes
                if i % 20 == 0:
                    sys.stdout.write(f'\r  Evaluating: {i+1}/{len(population)}')
                    sys.stdout.flush()
            
            sys.stdout.write('\r' + ' '*50 + '\r')  # Clear line
            logging.getLogger().setLevel(logging.INFO)
            
            # Find best
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_fitness = fitnesses[best_idx]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            # Update best ever
            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever_genome = population[best_idx].copy()
                status = "🏆 NEW BEST!"
            else:
                status = ""
            
            # Print summary
            if generation % args.summary_interval == 0 or status:
                active = sum(1 for v in population[best_idx].genes.values() if v)
                print(f"{generation:3d} | {best_fitness:.4f} | {avg_fitness:.4f} | {active:7d} | {status}")
            
            # Save checkpoint
            if generation % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"gen_{generation}.pt")
                torch.save({
                    'generation': generation,
                    'best_genome': best_ever_genome.to_dict() if best_ever_genome else None,
                    'best_fitness': best_ever_fitness,
                    'population_size': len(population),
                    'avg_fitness': avg_fitness
                }, checkpoint_path)
                print(f"    💾 Checkpoint saved (gen {generation})")
            
            # Create next generation
            if generation < args.generations:
                # Sort by fitness
                sorted_indices = sorted(range(len(fitnesses)), 
                                      key=lambda i: fitnesses[i], 
                                      reverse=True)
                
                # Elite
                new_population = [population[i].copy() for i in sorted_indices[:elite_size]]
                
                # Fill with offspring
                while len(new_population) < args.population:
                    # Tournament selection
                    p1_idx = max(torch.randint(0, len(population), (3,)).tolist(), 
                               key=lambda i: fitnesses[i])
                    p2_idx = max(torch.randint(0, len(population), (3,)).tolist(), 
                               key=lambda i: fitnesses[i])
                    
                    # Crossover
                    if torch.rand(1).item() < crossover_rate:
                        child = population[p1_idx].crossover(population[p2_idx])
                    else:
                        child = population[p1_idx].copy()
                    
                    # Mutation
                    child.mutate(mutation_rate)
                    new_population.append(child)
                
                population = new_population[:args.population]
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evolution stopped by user")
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 EVOLUTION SUMMARY")
    print("="*80)
    print(f"Generations completed: {generation}")
    print(f"Best fitness achieved: {best_ever_fitness:.4f}")
    
    if best_ever_genome:
        active_modules = [k for k, v in best_ever_genome.genes.items() if v]
        print(f"Best genome uses {len(active_modules)} modules:")
        for i, module in enumerate(active_modules):
            if i < 10:  # Show first 10
                print(f"  - {module}")
            elif i == 10:
                print(f"  ... and {len(active_modules)-10} more")
                break
    
    print(f"\n📁 Results saved in: {checkpoint_dir}")
    print("✅ Done!")


if __name__ == "__main__":
    main()