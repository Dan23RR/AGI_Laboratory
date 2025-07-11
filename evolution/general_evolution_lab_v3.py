#!/usr/bin/env python3
"""
General Evolution Laboratory V3 - With Refactored Modules and Meta-Evolution
===========================================================================

This version uses:
- All refactored modules (V2/V3/V4) with proper memory management
- Meta-evolution for self-adapting evolutionary parameters
- Enhanced safety and stability controls
"""

import os
import torch
import torch.nn as nn
import json
import numpy as np
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import multiprocessing as mp
from collections import defaultdict
import logging

# Import core infrastructure
from core import (
    get_memory_manager, get_checkpoint_manager,
    handle_errors, validate_tensor, EvolutionError,
    error_aggregator, ModuleConfig
)

# Import meta-evolution
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meta_evolution import MetaEvolution, MetaEvolutionConfig

# Import AGI components  
from evolution.extended_genome import ExtendedGenome
from evolution.mind_factory_v2 import MindFactoryV2 as ExtendedMindFactoryV2
from evolution.fitness.agi_fitness_metrics_v2 import AGIFitnessEvaluator, AGIFitnessScore
from environments.general_environments import EnvironmentSuite, MetaEnvironment
# from config_v2 import *  # Config file not found, using defaults below

logger = logging.getLogger(__name__)

# Default evolution parameters if not defined in config_v2
try:
    MUTATION_RATE
except NameError:
    MUTATION_RATE = 0.1
    
try:
    CROSSOVER_RATE
except NameError:
    CROSSOVER_RATE = 0.7
    
try:
    POPULATION_SIZE
except NameError:
    POPULATION_SIZE = 50


@dataclass
class EvolutionStatsV3:
    """Enhanced statistics with meta-evolution tracking"""
    generation: int
    best_fitness: float
    avg_fitness: float
    fitness_components: Dict[str, float]
    population_diversity: float
    emergence_events: int
    timestamp: str
    memory_usage_mb: float
    errors_count: int
    # Meta-evolution parameters
    mutation_rate: float
    crossover_rate: float
    selection_pressure: float
    diversity_weight: float
    meta_stability: str
    
    def to_dict(self):
        return asdict(self)


class GeneralEvolutionLabV3:
    """
    Evolution laboratory with refactored modules and meta-evolution
    """
    
    def __init__(self, checkpoint_dir: str = 'agi_evolution_checkpoints_v3',
                 enable_meta_evolution: bool = True):
        # Initialize core infrastructure
        self.memory_manager = get_memory_manager(total_budget_gb=32.0)
        self.checkpoint_manager = get_checkpoint_manager(checkpoint_dir)
        
        # Register components with memory manager
        self.memory_manager.register_module('evolution_lab', 0.25)
        self.memory_manager.register_module('fitness_evaluator', 0.20)
        self.memory_manager.register_module('environments', 0.15)
        self.memory_manager.register_module('population', 0.25)
        self.memory_manager.register_module('meta_evolution', 0.15)
        
        # Initialize components
        self.fitness_evaluator = AGIFitnessEvaluator()
        self.env_suite = EnvironmentSuite()
        self.mind_factory = ExtendedMindFactoryV2()
        
        # Initialize meta-evolution
        self.enable_meta_evolution = enable_meta_evolution
        if enable_meta_evolution:
            meta_config = MetaEvolutionConfig(
                meta_learning_rate=0.01,
                adaptation_window=10,
                performance_collapse_threshold=0.5,
                parameter_change_limit=0.1
            )
            self.meta_evolution = MetaEvolution(meta_config)
        else:
            self.meta_evolution = None
        
        # Evolution state
        self.population: List[ExtendedGenome] = []
        self.generation = 0
        self.evolution_history = []
        self.best_genome_ever = None
        self.best_fitness_ever = AGIFitnessScore()
        
        # Current evolution parameters (may be adapted by meta-evolution)
        self.current_params = {
            'mutation_rate': MUTATION_RATE,
            'crossover_rate': CROSSOVER_RATE,
            'selection_pressure': 2.0,
            'diversity_weight': 0.1
        }
        
        # Pre-allocate arrays
        self.fitness_scores = np.zeros(POPULATION_SIZE)
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        logger.info(f"GeneralEvolutionLabV3 initialized with meta-evolution: {enable_meta_evolution}")
    
    @handle_errors(error_types=EvolutionError, propagate=True)
    def initialize_population(self):
        """Create initial population with diversity"""
        logger.info(f"Initializing population of {POPULATION_SIZE} genomes...")
        
        self.population = []
        
        for i in range(POPULATION_SIZE):
            # Check memory allocation
            if not self.memory_manager.allocate('population', estimated_genome_size_bytes()):
                logger.warning(f"Memory limit reached at genome {i}")
                break
            
            genome = ExtendedGenome()
            
            # Ensure minimum complexity
            while sum(genome.genes.values()) < MIN_ACTIVE_MODULES:
                genome = ExtendedGenome()
            
            # Add diversity through initialization strategies
            if i < POPULATION_SIZE // 4:
                # Use refactored modules
                self._bias_genome_towards(genome, ['conscious_integration', 'emergent_consciousness', 'goal_mcts'])
            elif i < POPULATION_SIZE // 2:
                # Mix refactored and legacy
                self._bias_genome_towards(genome, ['feedback_loops', 'sentient_agi', 'dynamic_conceptual'])
            elif i < 3 * POPULATION_SIZE // 4:
                # Focus on planning and reasoning
                self._bias_genome_towards(genome, ['mcts', 'counterfactual', 'goal'])
            
            self.population.append(genome)
        
        actual_size = len(self.population)
        if actual_size < POPULATION_SIZE:
            logger.warning(f"Population limited to {actual_size} due to memory")
            self.fitness_scores = np.zeros(actual_size)
        
        logger.info(f"Population initialized with {actual_size} genomes")
    
    def _bias_genome_towards(self, genome: ExtendedGenome, preferred_keywords: List[str]):
        """Bias genome towards certain module types"""
        for gene_name in genome.genes:
            if any(keyword in gene_name.lower() for keyword in preferred_keywords):
                genome.genes[gene_name] = True
    
    @handle_errors(error_types=(RuntimeError, EvolutionError), default_return=None)
    def evaluate_population(self):
        """Evaluate population with refactored modules"""
        logger.info(f"Evaluating generation {self.generation}...")
        
        fitness_objects = []
        
        for i, genome in enumerate(self.population):
            # Periodic memory cleanup
            if i % 10 == 0:
                self.memory_manager.cleanup_module('fitness_evaluator')
            
            # Build and evaluate
            fitness = self._evaluate_genome_safe(genome, i)
            fitness_objects.append(fitness)
            
            # Update best
            if fitness.overall_agi_score() > self.best_fitness_ever.overall_agi_score():
                self.best_fitness_ever = fitness
                self.best_genome_ever = genome
                self._save_best_genome(fitness.overall_agi_score())
        
        # Extract scores
        self.fitness_scores = np.array([f.overall_agi_score() for f in fitness_objects])
        
        # Calculate population diversity
        diversity = self._calculate_population_diversity()
        
        # Update meta-evolution if enabled
        if self.meta_evolution:
            self.current_params = self.meta_evolution.update(
                self.fitness_scores,
                diversity
            )
            logger.info(f"Meta-evolution updated parameters: {self.current_params}")
        
        # Log statistics
        self._log_generation_stats(fitness_objects, diversity)
    
    def _evaluate_genome_safe(self, genome: ExtendedGenome, index: int) -> AGIFitnessScore:
        """Safely evaluate genome with error recovery"""
        try:
            # Build mind using refactored factory
            mind = self.mind_factory.build_from_genome(genome)
            
            if mind is None:
                logger.warning(f"Failed to build mind for genome {index}")
                return AGIFitnessScore()
            
            # Evaluate fitness
            component_models = list(mind.modules.values()) if hasattr(mind, 'modules') else None
            fitness = self.fitness_evaluator.evaluate_complete(mind, component_models)
            
            # Cleanup to free memory
            if hasattr(mind, 'cleanup'):
                mind.cleanup()
            del mind
            
            return fitness
            
        except Exception as e:
            error_aggregator.add_error(f"genome_{index}", e, {'generation': self.generation})
            logger.error(f"Error evaluating genome {index}: {e}")
            return AGIFitnessScore()
    
    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of population"""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Hamming distance between gene activations
                g1 = self.population[i].genes
                g2 = self.population[j].genes
                
                distance = sum(g1[k] != g2[k] for k in g1.keys()) / len(g1)
                distances.append(distance)
        
        # Average pairwise distance
        return np.mean(distances) if distances else 0.0
    
    def select_parents(self) -> List[ExtendedGenome]:
        """Selection with adaptive pressure from meta-evolution"""
        n_parents = len(self.population) // 2
        parents = []
        
        # Get selection pressure from meta-evolution
        selection_pressure = self.current_params.get('selection_pressure', 2.0)
        
        # Elitism
        elite_indices = np.argsort(self.fitness_scores)[-ELITE_SIZE:]
        for idx in elite_indices:
            parents.append(self.population[idx])
        
        # Tournament selection with adaptive pressure
        while len(parents) < n_parents:
            tournament_size = max(2, int(len(self.population) * 0.05 * selection_pressure))
            tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_scores = self.fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def create_next_generation(self, parents: List[ExtendedGenome]) -> List[ExtendedGenome]:
        """Create offspring with adaptive mutation/crossover rates"""
        next_generation = []
        
        # Get evolution parameters
        mutation_rate = self.current_params.get('mutation_rate', MUTATION_RATE)
        crossover_rate = self.current_params.get('crossover_rate', CROSSOVER_RATE)
        diversity_weight = self.current_params.get('diversity_weight', 0.1)
        
        # Keep elites
        elite_indices = np.argsort(self.fitness_scores)[-ELITE_SIZE:]
        for idx in elite_indices:
            next_generation.append(self.population[idx].copy())
        
        # Create offspring
        while len(next_generation) < len(self.population):
            # Select parents with diversity consideration
            if np.random.random() < diversity_weight:
                # Diversity selection - pick dissimilar parents
                p1 = np.random.choice(parents)
                p2 = max(parents, key=lambda p: self._genome_distance(p1, p))
            else:
                # Standard selection
                p1 = np.random.choice(parents)
                p2 = np.random.choice(parents)
            
            # Crossover with adaptive rate
            if np.random.random() < crossover_rate:
                child = self._crossover(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation with adaptive rate
            if np.random.random() < mutation_rate:
                child = self._mutate(child, mutation_rate)
            
            # Ensure minimum complexity
            if sum(child.genes.values()) >= MIN_ACTIVE_MODULES:
                next_generation.append(child)
        
        return next_generation[:len(self.population)]
    
    def _genome_distance(self, g1: ExtendedGenome, g2: ExtendedGenome) -> float:
        """Calculate distance between two genomes"""
        gene_dist = sum(g1.genes[k] != g2.genes[k] for k in g1.genes.keys()) / len(g1.genes)
        
        # Also consider hyperparameter differences
        hyper_dist = 0
        for k in g1.hyperparameters:
            if k in g2.hyperparameters:
                v1 = g1.hyperparameters[k]
                v2 = g2.hyperparameters[k]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    hyper_dist += abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-6)
        
        hyper_dist /= len(g1.hyperparameters)
        
        return 0.7 * gene_dist + 0.3 * hyper_dist
    
    def _crossover(self, parent1: ExtendedGenome, parent2: ExtendedGenome) -> ExtendedGenome:
        """Crossover with module awareness"""
        child = ExtendedGenome()
        
        # Gene crossover - keep module coherence
        module_groups = self._group_genes_by_module()
        
        for module, genes in module_groups.items():
            # Inherit entire module from one parent
            if np.random.random() < 0.5:
                for gene in genes:
                    child.genes[gene] = parent1.genes[gene]
            else:
                for gene in genes:
                    child.genes[gene] = parent2.genes[gene]
        
        # Hyperparameter crossover
        for key in parent1.hyperparameters:
            if np.random.random() < 0.5:
                child.hyperparameters[key] = parent1.hyperparameters[key]
            else:
                child.hyperparameters[key] = parent2.hyperparameters[key]
        
        return child
    
    def _group_genes_by_module(self) -> Dict[str, List[str]]:
        """Group genes by module for coherent crossover"""
        groups = defaultdict(list)
        
        # Define module groupings
        module_keywords = {
            'consciousness': ['conscious', 'emergent', 'awareness'],
            'planning': ['mcts', 'goal', 'plan'],
            'memory': ['feedback', 'memory', 'loop'],
            'reasoning': ['counterfactual', 'reason', 'logic'],
            'integration': ['integration', 'hub', 'global'],
            'motivation': ['empowerment', 'motivation', 'reward'],
            'world_model': ['world', 'model', 'prediction'],
            'other': []
        }
        
        # Assign genes to groups
        for gene in ExtendedGenome().genes.keys():
            assigned = False
            for module, keywords in module_keywords.items():
                if any(kw in gene.lower() for kw in keywords):
                    groups[module].append(gene)
                    assigned = True
                    break
            if not assigned:
                groups['other'].append(gene)
        
        return dict(groups)
    
    def _mutate(self, genome: ExtendedGenome, mutation_rate: float) -> ExtendedGenome:
        """Mutate with adaptive rates"""
        mutated = genome.copy()
        
        # Gene mutations
        for gene in mutated.genes:
            if np.random.random() < mutation_rate:
                mutated.genes[gene] = not mutated.genes[gene]
        
        # Hyperparameter mutations
        for key, value in mutated.hyperparameters.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, bool):
                    mutated.hyperparameters[key] = not value
                elif isinstance(value, int):
                    # Integer mutation
                    delta = np.random.randint(-2, 3)
                    mutated.hyperparameters[key] = max(1, value + delta)
                elif isinstance(value, float):
                    # Float mutation
                    delta = np.random.normal(0, 0.1 * abs(value))
                    mutated.hyperparameters[key] = max(0.001, value + delta)
        
        return mutated
    
    def evolve_one_generation(self):
        """Run one generation of evolution"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {self.generation}")
        logger.info(f"{'='*60}")
        
        # Evaluate current population
        self.evaluate_population()
        
        # Select parents
        parents = self.select_parents()
        
        # Create next generation
        self.population = self.create_next_generation(parents)
        
        # Increment generation counter
        self.generation += 1
        
        # Save checkpoint periodically
        if self.generation % CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint()
    
    def run_evolution(self, num_generations: int):
        """Run evolution for specified generations"""
        logger.info(f"Starting evolution for {num_generations} generations")
        
        start_gen = self.generation
        target_gen = start_gen + num_generations
        
        while self.generation < target_gen:
            try:
                self.evolve_one_generation()
                
                # Check for early stopping
                if self._should_stop_early():
                    logger.info("Early stopping triggered")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Evolution interrupted by user")
                self._save_checkpoint()
                break
            except Exception as e:
                logger.error(f"Error in generation {self.generation}: {e}")
                error_aggregator.add_error('evolution', e, {'generation': self.generation})
                
                # Try to recover
                if error_aggregator.get_error_count() > 10:
                    logger.error("Too many errors, stopping evolution")
                    break
        
        # Final checkpoint
        self._save_checkpoint()
        logger.info(f"Evolution completed. Total generations: {self.generation}")
    
    def _should_stop_early(self) -> bool:
        """Check if evolution should stop early"""
        # No improvement for many generations
        if len(self.evolution_history) > 50:
            recent_best = max(h.best_fitness for h in self.evolution_history[-50:])
            old_best = max(h.best_fitness for h in self.evolution_history[:-50])
            if recent_best <= old_best * 1.001:  # Less than 0.1% improvement
                return True
        
        return False
    
    def _log_generation_stats(self, fitness_objects: List[AGIFitnessScore], diversity: float):
        """Log detailed generation statistics"""
        best_fitness = max(f.overall_agi_score() for f in fitness_objects)
        avg_fitness = np.mean([f.overall_agi_score() for f in fitness_objects])
        
        # Get meta-evolution status
        meta_status = "stable"
        if self.meta_evolution:
            summary = self.meta_evolution.get_adaptation_summary()
            meta_status = summary['stability_status'][1]
        
        stats = EvolutionStatsV3(
            generation=self.generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            fitness_components=self._aggregate_fitness_components(fitness_objects),
            population_diversity=diversity,
            emergence_events=sum(f.emergence > 0.5 for f in fitness_objects),
            timestamp=datetime.now().isoformat(),
            memory_usage_mb=self.memory_manager.get_total_usage_mb(),
            errors_count=error_aggregator.get_error_count(),
            mutation_rate=self.current_params['mutation_rate'],
            crossover_rate=self.current_params['crossover_rate'],
            selection_pressure=self.current_params['selection_pressure'],
            diversity_weight=self.current_params['diversity_weight'],
            meta_stability=meta_status
        )
        
        self.evolution_history.append(stats)
        
        # Log summary
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Avg fitness: {avg_fitness:.4f}")
        logger.info(f"Population diversity: {diversity:.4f}")
        logger.info(f"Memory usage: {stats.memory_usage_mb:.1f} MB")
        logger.info(f"Evolution params: mut={stats.mutation_rate:.3f}, "
                   f"cross={stats.crossover_rate:.3f}, "
                   f"press={stats.selection_pressure:.2f}")
        
        # Log weakest capability for targeted evolution
        best_genome_idx = np.argmax(self.fitness_scores)
        weakest = fitness_objects[best_genome_idx].get_weakest_capability()
        logger.info(f"Weakest capability: {weakest[0]} ({weakest[1]:.4f})")
    
    def _aggregate_fitness_components(self, fitness_objects: List[AGIFitnessScore]) -> Dict[str, float]:
        """Aggregate fitness components across population"""
        components = defaultdict(list)
        
        for f in fitness_objects:
            components['generalization'].append(f.generalization)
            components['emergence'].append(f.emergence)
            components['adaptability'].append(f.adaptability)
            components['creativity'].append(f.creativity)
            components['reasoning'].append(f.reasoning)
            components['consciousness'].append(f.consciousness)
            components['efficiency'].append(f.efficiency)
            components['robustness'].append(f.robustness)
        
        return {k: float(np.mean(v)) for k, v in components.items()}
    
    def _save_checkpoint(self):
        """Save evolution state"""
        checkpoint = {
            'generation': self.generation,
            'population': [g.to_dict() for g in self.population],
            'best_genome': self.best_genome_ever.to_dict() if self.best_genome_ever else None,
            'best_fitness': self.best_fitness_ever.overall_agi_score(),
            'evolution_history': [s.to_dict() for s in self.evolution_history[-100:]],
            'current_params': self.current_params,
            'meta_evolution_state': self.meta_evolution.save_state('meta_evolution_temp.json') 
                                   if self.meta_evolution else None
        }
        
        filepath = self.checkpoint_manager.save_checkpoint(checkpoint, f"gen_{self.generation}")
        logger.info(f"Checkpoint saved: {filepath}")
    
    def _load_checkpoint(self):
        """Load evolution state from checkpoint"""
        # List available checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if checkpoints:
            # Get the latest checkpoint
            latest = max(checkpoints, key=lambda c: c['created'])
            # Load it - but we need to find how to load in this API
            logger.info(f"Found checkpoint: {latest['name']}, but load method not implemented")
            # For now, skip loading to allow the lab to start
        else:
            logger.info("No checkpoints found, starting fresh")
    
    def _save_best_genome(self, fitness: float):
        """Save the best genome found"""
        filepath = os.path.join(self.checkpoint_manager.checkpoint_dir, 
                               f"best_genome_f{fitness:.4f}_g{self.generation}.json")
        
        with open(filepath, 'w') as f:
            json.dump({
                'genome': self.best_genome_ever.to_dict(),
                'fitness': fitness,
                'generation': self.generation,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Best genome saved: {filepath}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create lab with meta-evolution
    lab = GeneralEvolutionLabV3(enable_meta_evolution=True)
    
    # Initialize population if needed
    if not lab.population:
        lab.initialize_population()
    
    # Run evolution
    lab.run_evolution(num_generations=100)