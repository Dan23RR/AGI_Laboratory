#!/usr/bin/env python3
"""
Internal Goal Generation V3 - Refactored with Memory Management
==============================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Fixed achievement history growing infinitely with circular buffer
- Optimized goal sampling and memory management
- Bounded goal hierarchies with efficient cleanup
- Proper resource cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from collections import deque
import uuid
import math

from core.base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from core.error_handling import RobustForward, handle_errors


@dataclass
class GoalV3:
    """Lightweight goal representation"""
    id: str
    embedding: torch.Tensor
    priority: float
    hierarchy_level: int
    creation_time: int
    expiry_time: int
    achievement_threshold: float = 0.8
    
    def __post_init__(self):
        # Detach embedding to prevent gradient accumulation
        self.embedding = self.embedding.detach()


class BoundedGoalMemory(nn.Module):
    """Memory-efficient goal memory with bounded storage"""
    
    def __init__(self, state_dim: int, max_goals: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.max_goals = max_goals
        
        # Pre-allocated goal storage
        self.register_buffer('goal_embeddings', torch.zeros(max_goals, state_dim))
        self.register_buffer('goal_priorities', torch.zeros(max_goals))
        self.register_buffer('goal_valid', torch.zeros(max_goals, dtype=torch.bool))
        self.register_buffer('access_counts', torch.zeros(max_goals))
        self.register_buffer('write_idx', torch.tensor(0, dtype=torch.long))
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )
        
    def add_goal(self, state: torch.Tensor, priority: float = 0.5) -> int:
        """Add goal to memory with circular buffer pattern"""
        idx = self.write_idx.item() % self.max_goals
        
        # Encode and store
        encoded = self.goal_encoder(state)
        self.goal_embeddings[idx] = encoded.detach()
        self.goal_priorities[idx] = priority
        self.goal_valid[idx] = True
        self.access_counts[idx] = 0
        
        self.write_idx += 1
        return idx
    
    def sample_goals(self, n_samples: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample goals based on priority and access patterns"""
        valid_indices = torch.where(self.goal_valid)[0]
        
        if len(valid_indices) == 0:
            # Return zeros if no valid goals
            return torch.zeros(n_samples, self.state_dim), torch.zeros(n_samples)
        
        n_samples = min(n_samples, len(valid_indices))
        
        # Compute sampling weights
        valid_priorities = self.goal_priorities[valid_indices]
        valid_access = self.access_counts[valid_indices]
        
        # Favor high priority, less accessed goals
        weights = valid_priorities * torch.exp(-valid_access / 100)
        probs = F.softmax(weights / temperature, dim=0)
        
        # Sample indices
        sampled_indices = torch.multinomial(probs, n_samples, replacement=False)
        goal_indices = valid_indices[sampled_indices]
        
        # Update access counts
        self.access_counts[goal_indices] += 1
        
        # Return sampled goals and priorities
        return self.goal_embeddings[goal_indices], self.goal_priorities[goal_indices]
    
    def remove_goal(self, idx: int):
        """Mark goal as invalid"""
        if 0 <= idx < self.max_goals:
            self.goal_valid[idx] = False
            self.access_counts[idx] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        n_valid = self.goal_valid.sum().item()
        return {
            'n_valid_goals': n_valid,
            'memory_utilization': n_valid / self.max_goals,
            'mean_priority': self.goal_priorities[self.goal_valid].mean().item() if n_valid > 0 else 0,
            'mean_access_count': self.access_counts[self.goal_valid].mean().item() if n_valid > 0 else 0
        }


class HierarchicalGoalGenerator(nn.Module):
    """Generate goals at multiple time horizons with bounded storage"""
    
    def __init__(self, state_dim: int, n_hierarchies: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.n_hierarchies = n_hierarchies
        
        # Hierarchy-specific generators
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.LayerNorm(state_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(state_dim, state_dim),
                nn.Tanh()  # Bounded output
            ) for _ in range(n_hierarchies)
        ])
        
        # Time horizon embeddings
        self.time_embeddings = nn.Parameter(torch.randn(n_hierarchies, state_dim) * 0.1)
        
    def generate(self, state: torch.Tensor, hierarchy_level: int) -> torch.Tensor:
        """Generate goal for specific hierarchy level"""
        if hierarchy_level >= self.n_hierarchies:
            hierarchy_level = self.n_hierarchies - 1
        
        # Add time horizon context
        time_context = state + self.time_embeddings[hierarchy_level]
        
        # Generate goal
        goal = self.generators[hierarchy_level](time_context)
        
        return goal


class InternalGoalGenerationV3(BaseAGIModule):
    """
    Refactored Internal Goal Generation with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.state_dim = self.config.hidden_size
        self.max_active_goals = 50  # Reduced from unbounded
        self.n_hierarchies = 3
        
        # Time tracking
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))
        
        # Bounded goal memory
        self.goal_memory = BoundedGoalMemory(self.state_dim, max_goals=200)
        
        # Hierarchical generator
        self.goal_generator = HierarchicalGoalGenerator(self.state_dim, self.n_hierarchies)
        
        # Goal evaluation network
        self.goal_evaluator = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, 1),
            nn.Sigmoid()
        )
        
        # Curiosity module
        self.curiosity_net = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim)
        )
        
        # Goal processor
        self.goal_processor = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_dim, self.config.output_dim)
        )
        
        # Active goals tracking (bounded)
        self.active_goals: Dict[str, GoalV3] = {}
        self.goal_hierarchies = {i: [] for i in range(self.n_hierarchies)}
        
        # Bounded history tracking
        self.goal_history = self.create_buffer(500)  # Already bounded with CircularBuffer
        self.achievement_history = self.create_buffer(1000)  # Fixed: now bounded!
        
        # Pre-allocated workspace
        self.register_buffer('_goal_distances', torch.zeros(self.max_active_goals))
        
    @RobustForward()
    def _forward_impl(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded goal generation and tracking
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        
        # Increment time
        self.time_step += 1
        current_time = self.time_step.item()
        
        # Clean up expired goals
        self._cleanup_expired_goals(current_time)
        
        # Generate new goals if needed
        if len(self.active_goals) < self.max_active_goals:
            self._generate_new_goals(state, current_time)
        
        # Evaluate progress on active goals
        achieved_goals = self._evaluate_goal_progress(state, current_time)
        
        # Sample goals for current action
        sampled_goals, priorities = self.goal_memory.sample_goals(
            min(5, len(self.active_goals)),
            temperature=kwargs.get('goal_temperature', 1.0)
        )
        
        # Compute curiosity bonus
        curiosity = self.curiosity_net(state)
        curiosity_bonus = torch.norm(curiosity, dim=-1, keepdim=True)
        
        # Process goals and state
        if sampled_goals.shape[0] > 0:
            # Combine state with sampled goals
            goal_context = sampled_goals.mean(dim=0, keepdim=True)
            combined = torch.cat([state, goal_context.expand(batch_size, -1)], dim=-1)
        else:
            # No goals, use curiosity
            combined = torch.cat([state, curiosity], dim=-1)
        
        # Generate output
        output = self.goal_processor(combined)
        
        # Track statistics (bounded)
        self.goal_history.append(torch.tensor(len(self.active_goals), dtype=torch.float))
        if achieved_goals:
            self.achievement_history.append(torch.tensor(len(achieved_goals), dtype=torch.float))
        
        return {
            'output': output,
            'n_active_goals': len(self.active_goals),
            'n_achieved': len(achieved_goals),
            'curiosity_bonus': curiosity_bonus.mean(),
            'goal_diversity': self._compute_goal_diversity(),
            'hierarchy_distribution': self._get_hierarchy_distribution(),
            'memory_stats': self.goal_memory.get_stats()
        }
    
    def _generate_new_goals(self, state: torch.Tensor, current_time: int):
        """Generate new goals with bounded storage"""
        # Determine how many goals to generate
        n_to_generate = min(
            self.max_active_goals - len(self.active_goals),
            3  # Max 3 new goals per step
        )
        
        for _ in range(n_to_generate):
            # Choose hierarchy level
            hierarchy_level = np.random.choice(self.n_hierarchies, p=[0.5, 0.3, 0.2])
            
            # Generate goal
            goal_embedding = self.goal_generator.generate(state[0], hierarchy_level)
            
            # Compute priority based on novelty
            novelty = self._compute_novelty(goal_embedding)
            priority = torch.sigmoid(novelty).item()
            
            # Create goal
            goal = GoalV3(
                id=str(uuid.uuid4())[:8],  # Shorter IDs
                embedding=goal_embedding,
                priority=priority,
                hierarchy_level=hierarchy_level,
                creation_time=current_time,
                expiry_time=current_time + (hierarchy_level + 1) * 100
            )
            
            # Add to active goals
            self.active_goals[goal.id] = goal
            self.goal_hierarchies[hierarchy_level].append(goal.id)
            
            # Add to memory
            self.goal_memory.add_goal(goal_embedding, priority)
    
    def _evaluate_goal_progress(self, state: torch.Tensor, current_time: int) -> List[str]:
        """Evaluate progress on active goals"""
        achieved_goals = []
        
        for goal_id, goal in list(self.active_goals.items()):
            # Compute achievement score
            combined = torch.cat([state[0], goal.embedding], dim=-1)
            achievement_score = self.goal_evaluator(combined).item()
            
            # Check if achieved
            if achievement_score > goal.achievement_threshold:
                achieved_goals.append(goal_id)
                
                # Remove from active goals
                del self.active_goals[goal_id]
                self.goal_hierarchies[goal.hierarchy_level].remove(goal_id)
        
        return achieved_goals
    
    def _cleanup_expired_goals(self, current_time: int):
        """Remove expired goals"""
        expired_goals = []
        
        for goal_id, goal in list(self.active_goals.items()):
            if current_time > goal.expiry_time:
                expired_goals.append(goal_id)
                
                # Remove from active goals
                del self.active_goals[goal_id]
                if goal_id in self.goal_hierarchies[goal.hierarchy_level]:
                    self.goal_hierarchies[goal.hierarchy_level].remove(goal_id)
        
        return expired_goals
    
    def _compute_novelty(self, goal_embedding: torch.Tensor) -> torch.Tensor:
        """Compute novelty score for goal"""
        if not self.active_goals:
            return torch.tensor(1.0)
        
        # Compare to existing goals
        existing_embeddings = torch.stack([g.embedding for g in self.active_goals.values()])
        distances = torch.norm(existing_embeddings - goal_embedding.unsqueeze(0), dim=-1)
        
        # Novelty is mean distance to existing goals
        return distances.mean()
    
    def _compute_goal_diversity(self) -> float:
        """Compute diversity of active goals"""
        if len(self.active_goals) < 2:
            return 0.0
        
        embeddings = torch.stack([g.embedding for g in self.active_goals.values()])
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)
        
        # Mean of upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        diversity = dists[mask].mean().item()
        
        return diversity
    
    def _get_hierarchy_distribution(self) -> Dict[int, float]:
        """Get distribution of goals across hierarchies"""
        total = len(self.active_goals)
        if total == 0:
            return {i: 0.0 for i in range(self.n_hierarchies)}
        
        return {
            i: len(self.goal_hierarchies[i]) / total
            for i in range(self.n_hierarchies)
        }
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear histories
        self.goal_history.clear()
        self.achievement_history.clear()
        
        # Clear active goals
        self.active_goals.clear()
        for h in self.goal_hierarchies.values():
            h.clear()
        
        # Reset time
        self.time_step.zero_()
    
    def get_goal_stats(self) -> Dict[str, Any]:
        """Get goal generation statistics"""
        goal_hist = self.goal_history.get_all()
        achieve_hist = self.achievement_history.get_all()
        
        return {
            'n_active_goals': len(self.active_goals),
            'mean_active_goals': np.mean(goal_hist) if goal_hist else 0.0,
            'total_achievements': sum(achieve_hist) if achieve_hist else 0,
            'achievement_rate': np.mean(achieve_hist) if achieve_hist else 0.0,
            'goal_diversity': self._compute_goal_diversity(),
            'hierarchy_distribution': self._get_hierarchy_distribution(),
            'memory_stats': self.goal_memory.get_stats(),
            'time_step': self.time_step.item(),
            'memory_usage': self.get_memory_usage()
        }