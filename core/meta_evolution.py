#!/usr/bin/env python3
"""
Meta-Evolution System with Robust Safety Controls
================================================

This module implements meta-evolution - the evolution of evolutionary parameters themselves.
Includes extensive safety mechanisms to prevent catastrophic parameter drift.

Key safety features:
- Hard bounds on all evolutionary parameters
- Gradual adaptation with momentum
- Automatic rollback on performance collapse
- Diversity preservation mechanisms
- Stability monitoring and intervention
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class MetaEvolutionConfig:
    """Configuration for meta-evolution with safety bounds"""
    # Mutation rate bounds
    mutation_rate_min: float = 0.001
    mutation_rate_max: float = 0.5
    mutation_rate_initial: float = 0.1
    
    # Crossover rate bounds
    crossover_rate_min: float = 0.1
    crossover_rate_max: float = 0.9
    crossover_rate_initial: float = 0.7
    
    # Selection pressure bounds
    selection_pressure_min: float = 1.0
    selection_pressure_max: float = 10.0
    selection_pressure_initial: float = 2.0
    
    # Population diversity bounds
    diversity_weight_min: float = 0.0
    diversity_weight_max: float = 0.5
    diversity_weight_initial: float = 0.1
    
    # Meta-learning parameters
    meta_learning_rate: float = 0.01
    meta_momentum: float = 0.9
    adaptation_window: int = 10  # generations
    
    # Safety parameters
    performance_collapse_threshold: float = 0.5  # 50% drop triggers rollback
    min_population_diversity: float = 0.1
    parameter_change_limit: float = 0.1  # Max 10% change per generation
    stability_check_interval: int = 5
    rollback_window: int = 20


@dataclass
class EvolutionState:
    """Current state of evolutionary parameters"""
    mutation_rate: float
    crossover_rate: float
    selection_pressure: float
    diversity_weight: float
    generation: int
    
    # Performance tracking
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    population_diversity: float = 1.0
    
    # Momentum for smooth adaptation
    mutation_rate_momentum: float = 0.0
    crossover_rate_momentum: float = 0.0
    selection_pressure_momentum: float = 0.0
    diversity_weight_momentum: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_pressure': self.selection_pressure,
            'diversity_weight': self.diversity_weight,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'population_diversity': self.population_diversity
        }


class SafetyMonitor:
    """Monitors evolution stability and triggers interventions"""
    
    def __init__(self, config: MetaEvolutionConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.rollback_window)
        self.diversity_history = deque(maxlen=config.rollback_window)
        self.parameter_history = deque(maxlen=config.rollback_window)
        
        # Tracking for anomaly detection
        self.consecutive_drops = 0
        self.stagnation_counter = 0
        self.last_intervention = -1
        
    def check_stability(self, state: EvolutionState) -> Tuple[bool, str]:
        """Check if evolution is stable"""
        # Track metrics
        self.performance_history.append(state.avg_fitness)
        self.diversity_history.append(state.population_diversity)
        self.parameter_history.append(state.to_dict())
        
        # Need minimum history
        if len(self.performance_history) < 5:
            return True, "Insufficient history"
        
        # Check for performance collapse
        recent_performance = list(self.performance_history)[-5:]
        baseline = np.mean(list(self.performance_history)[:-5])
        current = np.mean(recent_performance)
        
        if baseline > 0 and current < baseline * self.config.performance_collapse_threshold:
            self.consecutive_drops += 1
            if self.consecutive_drops >= 3:
                return False, f"Performance collapse: {current:.3f} < {baseline * self.config.performance_collapse_threshold:.3f}"
        else:
            self.consecutive_drops = 0
        
        # Check for diversity collapse
        if state.population_diversity < self.config.min_population_diversity:
            return False, f"Diversity too low: {state.population_diversity:.3f}"
        
        # Check for stagnation
        if len(set(recent_performance)) == 1:
            self.stagnation_counter += 1
            if self.stagnation_counter >= 10:
                return False, "Evolution stagnated"
        else:
            self.stagnation_counter = 0
        
        return True, "Stable"
    
    def get_rollback_state(self) -> Optional[Dict[str, float]]:
        """Get parameters from before the issue started"""
        if len(self.parameter_history) >= 10:
            # Return state from 10 generations ago
            return self.parameter_history[-10]
        return None


class MetaEvolution:
    """
    Meta-evolution system that evolves evolutionary parameters
    with robust safety controls
    """
    
    def __init__(self, config: Optional[MetaEvolutionConfig] = None):
        self.config = config or MetaEvolutionConfig()
        self.safety_monitor = SafetyMonitor(self.config)
        
        # Initialize state
        self.state = EvolutionState(
            mutation_rate=self.config.mutation_rate_initial,
            crossover_rate=self.config.crossover_rate_initial,
            selection_pressure=self.config.selection_pressure_initial,
            diversity_weight=self.config.diversity_weight_initial,
            generation=0
        )
        
        # Performance tracking
        self.fitness_history = deque(maxlen=self.config.adaptation_window)
        self.diversity_history = deque(maxlen=self.config.adaptation_window)
        
        # Gradient estimates for meta-learning
        self.param_gradients = {
            'mutation_rate': deque(maxlen=5),
            'crossover_rate': deque(maxlen=5),
            'selection_pressure': deque(maxlen=5),
            'diversity_weight': deque(maxlen=5)
        }
        
        logger.info("MetaEvolution initialized with safety controls")
    
    def update(self, fitness_scores: np.ndarray, 
               population_diversity: float) -> Dict[str, float]:
        """
        Update evolutionary parameters based on performance
        
        Args:
            fitness_scores: Array of fitness scores for current generation
            population_diversity: Measure of genetic diversity (0-1)
            
        Returns:
            Updated evolutionary parameters
        """
        # Update state
        self.state.generation += 1
        self.state.best_fitness = float(np.max(fitness_scores))
        self.state.avg_fitness = float(np.mean(fitness_scores))
        self.state.population_diversity = population_diversity
        
        # Track history
        self.fitness_history.append(self.state.avg_fitness)
        self.diversity_history.append(population_diversity)
        
        # Check stability
        stable, reason = self.safety_monitor.check_stability(self.state)
        
        if not stable:
            logger.warning(f"Stability check failed: {reason}")
            self._handle_instability(reason)
        else:
            # Adapt parameters if we have enough history
            if len(self.fitness_history) >= 5:
                self._adapt_parameters()
        
        # Apply safety bounds
        self._apply_safety_bounds()
        
        return self.get_current_parameters()
    
    def _adapt_parameters(self):
        """Adapt parameters using gradient-based meta-learning"""
        # Estimate gradients using finite differences
        if len(self.fitness_history) < 2:
            return
        
        # Performance trend
        performance_gradient = self.fitness_history[-1] - self.fitness_history[-2]
        diversity_gradient = self.diversity_history[-1] - self.diversity_history[-2]
        
        # Mutation rate adaptation
        if performance_gradient > 0 and diversity_gradient < 0:
            # Performance improving but diversity dropping - reduce mutation
            mutation_gradient = -0.1
        elif performance_gradient < 0 and diversity_gradient > 0:
            # Performance dropping but diversity increasing - reduce mutation 
            mutation_gradient = -0.05
        elif performance_gradient < 0 and diversity_gradient < 0:
            # Both dropping - increase mutation
            mutation_gradient = 0.1
        else:
            # Both improving - slight increase in mutation
            mutation_gradient = 0.02
            
        self.param_gradients['mutation_rate'].append(mutation_gradient)
        
        # Crossover rate adaptation
        if len(self.fitness_history) >= 3:
            # Look at fitness variance
            recent_variance = np.var(list(self.fitness_history)[-3:])
            if recent_variance < 0.001:  # Low variance suggests need for more exploration
                crossover_gradient = 0.05
            else:
                crossover_gradient = -0.02
            self.param_gradients['crossover_rate'].append(crossover_gradient)
        
        # Selection pressure adaptation
        if diversity_gradient < -0.05:  # Rapid diversity loss
            pressure_gradient = -0.1  # Reduce pressure
        elif diversity_gradient > 0.05 and performance_gradient < 0:
            pressure_gradient = 0.1  # Increase pressure
        else:
            pressure_gradient = 0.0
        self.param_gradients['selection_pressure'].append(pressure_gradient)
        
        # Diversity weight adaptation
        if self.state.population_diversity < 0.2:
            diversity_gradient = 0.1  # Urgently increase diversity weight
        elif self.state.population_diversity > 0.8:
            diversity_gradient = -0.05  # Can reduce diversity weight
        else:
            diversity_gradient = 0.0
        self.param_gradients['diversity_weight'].append(diversity_gradient)
        
        # Apply updates with momentum
        self._apply_parameter_updates()
    
    def _apply_parameter_updates(self):
        """Apply parameter updates with momentum and safety limits"""
        # Mutation rate
        if len(self.param_gradients['mutation_rate']) > 0:
            grad = np.mean(self.param_gradients['mutation_rate'])
            self.state.mutation_rate_momentum = (
                self.config.meta_momentum * self.state.mutation_rate_momentum +
                (1 - self.config.meta_momentum) * grad
            )
            delta = self.config.meta_learning_rate * self.state.mutation_rate_momentum
            delta = np.clip(delta, -self.config.parameter_change_limit, 
                          self.config.parameter_change_limit)
            self.state.mutation_rate *= (1 + delta)
        
        # Crossover rate
        if len(self.param_gradients['crossover_rate']) > 0:
            grad = np.mean(self.param_gradients['crossover_rate'])
            self.state.crossover_rate_momentum = (
                self.config.meta_momentum * self.state.crossover_rate_momentum +
                (1 - self.config.meta_momentum) * grad
            )
            delta = self.config.meta_learning_rate * self.state.crossover_rate_momentum
            delta = np.clip(delta, -self.config.parameter_change_limit,
                          self.config.parameter_change_limit)
            self.state.crossover_rate *= (1 + delta)
        
        # Selection pressure
        if len(self.param_gradients['selection_pressure']) > 0:
            grad = np.mean(self.param_gradients['selection_pressure'])
            self.state.selection_pressure_momentum = (
                self.config.meta_momentum * self.state.selection_pressure_momentum +
                (1 - self.config.meta_momentum) * grad
            )
            delta = self.config.meta_learning_rate * self.state.selection_pressure_momentum
            delta = np.clip(delta, -self.config.parameter_change_limit,
                          self.config.parameter_change_limit)
            self.state.selection_pressure *= (1 + delta)
        
        # Diversity weight
        if len(self.param_gradients['diversity_weight']) > 0:
            grad = np.mean(self.param_gradients['diversity_weight'])
            self.state.diversity_weight_momentum = (
                self.config.meta_momentum * self.state.diversity_weight_momentum +
                (1 - self.config.meta_momentum) * grad
            )
            delta = self.config.meta_learning_rate * self.state.diversity_weight_momentum
            delta = np.clip(delta, -self.config.parameter_change_limit,
                          self.config.parameter_change_limit)
            self.state.diversity_weight *= (1 + delta)
    
    def _apply_safety_bounds(self):
        """Enforce hard bounds on all parameters"""
        self.state.mutation_rate = np.clip(
            self.state.mutation_rate,
            self.config.mutation_rate_min,
            self.config.mutation_rate_max
        )
        
        self.state.crossover_rate = np.clip(
            self.state.crossover_rate,
            self.config.crossover_rate_min,
            self.config.crossover_rate_max
        )
        
        self.state.selection_pressure = np.clip(
            self.state.selection_pressure,
            self.config.selection_pressure_min,
            self.config.selection_pressure_max
        )
        
        self.state.diversity_weight = np.clip(
            self.state.diversity_weight,
            self.config.diversity_weight_min,
            self.config.diversity_weight_max
        )
    
    def _handle_instability(self, reason: str):
        """Handle detected instability"""
        logger.warning(f"Handling instability: {reason}")
        
        # Try to rollback to stable parameters
        rollback_state = self.safety_monitor.get_rollback_state()
        if rollback_state:
            logger.info("Rolling back to previous stable state")
            self.state.mutation_rate = rollback_state['mutation_rate']
            self.state.crossover_rate = rollback_state['crossover_rate']
            self.state.selection_pressure = rollback_state['selection_pressure']
            self.state.diversity_weight = rollback_state['diversity_weight']
            
            # Reset momentum
            self.state.mutation_rate_momentum = 0.0
            self.state.crossover_rate_momentum = 0.0
            self.state.selection_pressure_momentum = 0.0
            self.state.diversity_weight_momentum = 0.0
            
            # Clear gradient history
            for key in self.param_gradients:
                self.param_gradients[key].clear()
        else:
            # No rollback available - reset to safe defaults
            logger.warning("No rollback state available, resetting to defaults")
            self.state.mutation_rate = self.config.mutation_rate_initial
            self.state.crossover_rate = self.config.crossover_rate_initial
            self.state.selection_pressure = self.config.selection_pressure_initial
            self.state.diversity_weight = self.config.diversity_weight_initial
        
        # Mark intervention
        self.safety_monitor.last_intervention = self.state.generation
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current evolutionary parameters"""
        return {
            'mutation_rate': self.state.mutation_rate,
            'crossover_rate': self.state.crossover_rate,
            'selection_pressure': self.state.selection_pressure,
            'diversity_weight': self.state.diversity_weight,
            'generation': self.state.generation
        }
    
    def save_state(self, filepath: str):
        """Save meta-evolution state"""
        state_dict = {
            'state': self.state.to_dict(),
            'config': self.config.__dict__,
            'fitness_history': list(self.fitness_history),
            'diversity_history': list(self.diversity_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load meta-evolution state"""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Restore state
        self.state.mutation_rate = state_dict['state']['mutation_rate']
        self.state.crossover_rate = state_dict['state']['crossover_rate']
        self.state.selection_pressure = state_dict['state']['selection_pressure']
        self.state.diversity_weight = state_dict['state']['diversity_weight']
        self.state.generation = state_dict['state']['generation']
        
        # Restore history
        self.fitness_history.clear()
        self.fitness_history.extend(state_dict['fitness_history'])
        self.diversity_history.clear()
        self.diversity_history.extend(state_dict['diversity_history'])
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of meta-evolution adaptation"""
        return {
            'current_parameters': self.get_current_parameters(),
            'performance_trend': np.polyfit(range(len(self.fitness_history)), 
                                          list(self.fitness_history), 1)[0] 
                               if len(self.fitness_history) > 1 else 0.0,
            'diversity_trend': np.polyfit(range(len(self.diversity_history)),
                                        list(self.diversity_history), 1)[0]
                              if len(self.diversity_history) > 1 else 0.0,
            'stability_status': self.safety_monitor.check_stability(self.state),
            'generations_since_intervention': (
                self.state.generation - self.safety_monitor.last_intervention
                if self.safety_monitor.last_intervention >= 0 else self.state.generation
            )
        }