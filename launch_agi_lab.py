#!/usr/bin/env python3
"""
AGI Laboratory Launch Script
===========================

Main entry point for starting AGI evolution experiments.
"""

import torch
import argparse
import logging
from datetime import datetime
import os
import numpy as np

# Core imports
from evolution.general_evolution_lab_v3 import GeneralEvolutionLabV3
from core.meta_evolution import MetaEvolution, MetaEvolutionConfig
from evolution.extended_genome import ExtendedGenome
from evolution.mind_factory_v2 import MindFactoryV2, MindConfig
from evolution.fitness.agi_fitness_metrics_v2 import AGIFitnessEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evolution_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_fitness_function(target_task="general"):
    """Create fitness function based on target task"""
    
    evaluator = AGIFitnessEvaluator() # Instantiate evaluator once

    def agi_overall_fitness(genome: ExtendedGenome) -> float:
        """Comprehensive AGI fitness using AGIFitnessEvaluator"""
        factory = MindFactoryV2()
        
        try:
            mind_config = MindConfig(
                hidden_dim=512,
                output_dim=256,
                memory_fraction=0.5
            )
            mind = factory.create_mind_from_genome(genome.to_dict(), mind_config)
            
            if mind is None:
                logger.warning("Mind creation failed, returning 0.0 fitness.")
                return 0.0

            # Evaluate complete AGI fitness
            agi_score_obj = evaluator.evaluate_complete(mind)
            overall_score = agi_score_obj.overall_agi_score()

            # Add bonus for FeedbackLoopSystemV3
            bonus = 0.0
            if genome.genes.get("FeedbackLoopSystemV3", False):
                bonus += 0.1 # Small bonus for activating this key module

            # Add bonus for other key modules if desired
            if genome.genes.get("EmpowermentCalculatorV3", False):
                bonus += 0.05
            if genome.genes.get("EmergentConsciousnessV4", False):
                bonus += 0.05

            final_fitness = overall_score + bonus
            
            # Ensure fitness is within a reasonable range (e.g., 0 to 1.5)
            final_fitness = max(0.0, min(1.5, final_fitness))

            mind.cleanup()
            return final_fitness
            
        except Exception as e:
            logger.error(f"Error evaluating genome with AGIFitnessEvaluator: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            factory.cleanup()
    
    # For now, all tasks will use the comprehensive AGI fitness
    # In the future, specific task fitness functions could be implemented
    # by weighting different components of AGIFitnessScore.
    return agi_overall_fitness


def main():
    parser = argparse.ArgumentParser(description='Launch AGI Evolution Laboratory')
    
    # Evolution parameters
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations to evolve (default: 100)')
    parser.add_argument('--population', type=int, default=50,
                       help='Population size (default: 50)')
    parser.add_argument('--task', type=str, default='general',
                       choices=['general', 'trading', 'research'],
                       help='Target task for evolution (default: general)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N generations (default: 10)')
    parser.add_argument('--memory-limit', type=float, default=16.0,
                       help='Memory limit in GB (default: 16.0)')
    
    # Meta-evolution
    parser.add_argument('--meta-evolution', action='store_true',
                       help='Enable meta-evolution (self-adapting parameters)')
    parser.add_argument('--safety-checks', action='store_true', default=True,
                       help='Enable safety checks (default: True)')
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info("="*80)
    logger.info("🚀 AGI LABORATORY LAUNCH")
    logger.info("="*80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Population: {args.population}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Meta-evolution: {args.meta_evolution}")
    logger.info(f"Safety checks: {args.safety_checks}")
    logger.info("="*80)
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize evolution lab
    logger.info("\n🧬 Initializing Evolution Laboratory...")
    lab = GeneralEvolutionLabV3(
        checkpoint_dir=checkpoint_dir,
        enable_meta_evolution=args.meta_evolution
    )
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Get fitness function
    fitness_fn = create_fitness_function(args.task)
    
    # Configure evolution
    evolution_params = {
        'population_size': args.population,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'elite_size': max(2, args.population // 10),
        'diversity_bonus': 0.1,
        'checkpoint_interval': args.checkpoint_interval
    }
    
    logger.info(f"\n📊 Evolution Parameters: {evolution_params}")
    
    # Run evolution
    logger.info(f"\n🔬 Starting evolution for {args.generations} generations...")
    
    try:
        # Create initial population
        initial_population = []
        for i in range(args.population):
            genome = ExtendedGenome()
            # ExtendedGenome is already initialized with random values
            
            if i < args.population // 3:
                # Keep as random (default initialization)
                pass
            elif i < 2 * args.population // 3:
                # Ensure some key modules are active
                key_modules = ["ConsciousIntegrationHub", "EmergentConsciousness"]
                for module in key_modules:
                    if module in genome.genes:
                        genome.genes[module] = True
            else:
                # Some minimal genomes - turn off most modules
                for key in list(genome.genes.keys())[5:]:
                    genome.genes[key] = False
            
            initial_population.append(genome)
        
        # Evolution loop
        best_genome = None
        best_fitness = 0.0
        
        for generation in range(args.generations):
            logger.info(f"\n🧪 Generation {generation + 1}/{args.generations}")
            
            # Evaluate population
            fitnesses = []
            for genome in initial_population:
                fitness = fitness_fn(genome)
                fitnesses.append(fitness)
            
            # Find best
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            current_best_fitness = fitnesses[best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_genome = initial_population[best_idx].copy()
                logger.info(f"🏆 New best fitness: {best_fitness:.4f}")
            
            # Log statistics
            avg_fitness = sum(fitnesses) / len(fitnesses)
            logger.info(f"📈 Avg fitness: {avg_fitness:.4f}, Best: {current_best_fitness:.4f}")
            
            # Save checkpoint
            if (generation + 1) % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"gen_{generation+1}.pt")
                torch.save({
                    'generation': generation,
                    'best_genome': best_genome.to_dict() if best_genome else None,
                    'best_fitness': best_fitness,
                    'population': [g.to_dict() for g in initial_population],
                    'fitnesses': fitnesses
                }, checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Create next generation
            if generation < args.generations - 1:
                # Sort by fitness
                sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
                
                # Elite selection
                elite_size = evolution_params['elite_size']
                new_population = [initial_population[i].copy() for i in sorted_indices[:elite_size]]
                
                # Fill rest with offspring
                while len(new_population) < args.population:
                    # Tournament selection
                    parent1_idx = max(torch.randint(0, len(initial_population), (3,)).tolist(), 
                                    key=lambda i: fitnesses[i])
                    parent2_idx = max(torch.randint(0, len(initial_population), (3,)).tolist(), 
                                    key=lambda i: fitnesses[i])
                    
                    parent1 = initial_population[parent1_idx]
                    parent2 = initial_population[parent2_idx]
                    
                    # Crossover
                    if torch.rand(1).item() < evolution_params['crossover_rate']:
                        child = parent1.crossover(parent2)
                    else:
                        child = parent1.copy()
                    
                    # Mutation
                    child.mutate(evolution_params['mutation_rate'])
                    
                    new_population.append(child)
                
                initial_population = new_population[:args.population]
            
            # Meta-evolution update
            if args.meta_evolution and hasattr(lab, 'meta_evolution'):
                # MetaEvolution.update expects fitness scores and current parameters
                fitness_array = np.array(fitnesses)
                current_params = {
                    'mutation_rate': evolution_params['mutation_rate'],
                    'crossover_rate': evolution_params['crossover_rate'],
                    'selection_pressure': evolution_params.get('selection_pressure', 2.0),
                    'diversity_weight': evolution_params.get('diversity_weight', 0.1)
                }
                
                try:
                    # Update meta-evolution
                    lab.meta_evolution.update(fitness_array, current_params)
                    
                    # Get current parameters after adaptation
                    new_mutation_rate = lab.meta_evolution.current_params.get('mutation_rate', evolution_params['mutation_rate'])
                    if new_mutation_rate != evolution_params['mutation_rate']:
                        evolution_params['mutation_rate'] = new_mutation_rate
                        logger.info(f"🔧 Updated mutation_rate: {new_mutation_rate:.3f}")
                except Exception as e:
                    logger.warning(f"Meta-evolution update failed: {e}")
        
        # Final results
        logger.info("\n" + "="*80)
        logger.info("🏁 EVOLUTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Best fitness achieved: {best_fitness:.4f}")
        
        if best_genome:
            active_modules = [k for k, v in best_genome.genes.items() if v]
            logger.info(f"Best genome uses {len(active_modules)} modules:")
            for module in active_modules:
                logger.info(f"  - {module}")
        
        # Save final results
        final_path = os.path.join(checkpoint_dir, "final_results.pt")
        torch.save({
            'best_genome': best_genome.to_dict() if best_genome else None,
            'best_fitness': best_fitness,
            'total_generations': args.generations,
            'task': args.task
        }, final_path)
        logger.info(f"\n💾 Final results saved: {final_path}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Evolution interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Evolution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("\n🧹 Cleaning up...")
        # GeneralEvolutionLabV3 doesn't have cleanup method, just let it be garbage collected
        pass
        
    logger.info("\n✅ Laboratory shutdown complete")


if __name__ == "__main__":
    main()