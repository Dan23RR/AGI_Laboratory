#!/usr/bin/env python3
"""
Goal-Conditioned MCTS V3 - Memory-Managed Refactoring
=====================================================

Complete refactoring with:
- Memory management via BaseAGIModule
- Bounded tree structure with automatic pruning
- Pre-allocated tensor pools for all operations
- Efficient sparse tree representation
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import math

from .base_module import BaseAGIModule, ModuleConfig
from .memory_manager import CircularBuffer
from .error_handling import (
    handle_errors, validate_tensor, DimensionError,
    safe_normalize, RobustForward
)

import logging
logger = logging.getLogger(__name__)


@dataclass
class MCTSConfigV3:
    """Simplified MCTS configuration"""
    action_dim: int = 4
    state_dim: int = 16
    discount: float = 0.99
    c_puct: float = 1.0
    n_simulations: int = 50
    horizon: int = 10
    batch_size: int = 16
    max_tree_size: int = 1000  # Hard limit on tree nodes
    max_children_per_node: int = 10  # Limit branching
    cache_size: int = 5000  # Reduced from 10000
    value_network_hidden: int = 128  # Reduced from 256


class CompactMCTSNode:
    """Memory-efficient node representation using indices instead of storing tensors"""
    
    def __init__(self, state_idx: int = -1, parent_idx: int = -1, action_idx: int = -1):
        self.state_idx = state_idx
        self.parent_idx = parent_idx
        self.action_idx = action_idx
        
        # Statistics (scalars only)
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 1.0
        
        # Children as indices only
        self.child_indices = []  # List of child node indices
        
    def reset(self, state_idx: int, parent_idx: int = -1, action_idx: int = -1):
        """Reset node for reuse"""
        self.state_idx = state_idx
        self.parent_idx = parent_idx
        self.action_idx = action_idx
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 1.0
        self.child_indices.clear()
        
    @property
    def value(self) -> float:
        """Average value"""
        return self.value_sum / max(1, self.visit_count)
    
    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """UCB score computation"""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value + exploration


class BoundedMCTSTree:
    """Memory-bounded MCTS tree with automatic pruning"""
    
    def __init__(self, config: MCTSConfigV3, device: torch.device):
        self.config = config
        self.device = device
        
        # Pre-allocate state and action pools
        self.state_pool = torch.zeros(
            config.max_tree_size, config.state_dim, 
            device=device, dtype=torch.float32
        )
        self.action_pool = torch.zeros(
            config.max_tree_size, config.action_dim,
            device=device, dtype=torch.float32
        )
        
        # Node storage - pre-allocate all nodes
        self.nodes = [CompactMCTSNode() for _ in range(config.max_tree_size)]  # Pre-create all nodes
        self.free_indices = list(range(config.max_tree_size))  # Available slots
        self.node_count = 0
        
        # Root tracking
        self.root_idx = -1
        
    def add_root(self, state: torch.Tensor) -> int:
        """Add root node"""
        if self.node_count >= self.config.max_tree_size:
            self._prune_tree()
        
        idx = self._allocate_index()
        self.state_pool[idx] = state
        
        # Reuse existing node
        self.nodes[idx].reset(state_idx=idx)
        
        self.root_idx = idx
        self.node_count += 1
        return idx
    
    def add_child(self, parent_idx: int, action: torch.Tensor, state: torch.Tensor, prior: float = 1.0) -> int:
        """Add child node with bounds checking"""
        # Validate parent index
        if parent_idx < 0 or parent_idx >= len(self.nodes) or self.nodes[parent_idx] is None:
            raise ValueError(f"Invalid parent index: {parent_idx}")
        
        parent = self.nodes[parent_idx]
        
        # Check child limit
        if len(parent.child_indices) >= self.config.max_children_per_node:
            # Replace least visited child
            min_visits = float('inf')
            min_idx = -1
            for child_idx in parent.child_indices:
                if child_idx < len(self.nodes) and self.nodes[child_idx] is not None:
                    child = self.nodes[child_idx]
                    if child.visit_count < min_visits:
                        min_visits = child.visit_count
                        min_idx = child_idx
            
            if min_idx >= 0:
                self._free_node(min_idx)
                parent.child_indices.remove(min_idx)
        
        # Allocate new node
        if self.node_count >= self.config.max_tree_size:
            self._prune_tree()
        
        idx = self._allocate_index()
        self.state_pool[idx] = state
        self.action_pool[idx] = action
        
        # Reuse existing node
        self.nodes[idx].reset(state_idx=idx, parent_idx=parent_idx, action_idx=idx)
        self.nodes[idx].prior = prior
        
        parent.child_indices.append(idx)
        self.node_count += 1
        
        return idx
    
    def select_best_child(self, node_idx: int) -> Tuple[int, torch.Tensor]:
        """Select best child using UCB"""
        if node_idx < 0 or node_idx >= len(self.nodes):
            return -1, None
        
        node = self.nodes[node_idx]
        
        if not node.child_indices:
            return -1, None
        
        best_score = -float('inf')
        best_idx = -1
        
        for child_idx in node.child_indices:
            if child_idx < len(self.nodes) and self.nodes[child_idx] is not None:
                child = self.nodes[child_idx]
                score = child.ucb_score(node.visit_count, self.config.c_puct)
                if score > best_score:
                    best_score = score
                    best_idx = child_idx
        
        if best_idx >= 0:
            return best_idx, self.action_pool[best_idx]
        return -1, None
    
    def backup(self, leaf_idx: int, value: float):
        """Backup value through tree"""
        idx = leaf_idx
        depth = 0
        
        while idx >= 0 and idx < len(self.nodes):
            node = self.nodes[idx]
            if node.state_idx == -1:  # Node not active
                break
            node.visit_count += 1
            node.value_sum += value * (self.config.discount ** depth)
            
            idx = node.parent_idx
            depth += 1
    
    def get_action_values(self, node_idx: int) -> Dict[int, float]:
        """Get action values for a node's children"""
        node = self.nodes[node_idx]
        values = {}
        
        for child_idx in node.child_indices:
            child = self.nodes[child_idx]
            values[child_idx] = child.value
        
        return values
    
    def _allocate_index(self) -> int:
        """Allocate index from free pool"""
        if self.free_indices:
            return self.free_indices.pop()
        else:
            # Tree full, force pruning
            self._prune_tree()
            if self.free_indices:
                return self.free_indices.pop()
            else:
                raise RuntimeError("Cannot allocate node: tree full after pruning")
    
    def _free_node(self, idx: int):
        """Free a node and its subtree"""
        if idx < 0 or idx >= len(self.nodes):
            return
            
        node = self.nodes[idx]
        if node.state_idx == -1:  # Already freed
            return
        
        # Recursively free children
        for child_idx in list(node.child_indices):  # Copy to avoid modification during iteration
            self._free_node(child_idx)
        
        # Reset node (don't create new object)
        node.reset(-1, -1, -1)
        self.free_indices.append(idx)
        self.node_count -= 1
    
    def _prune_tree(self):
        """Prune least visited branches"""
        if self.root_idx < 0:
            return
        
        # Find nodes to prune (least visited leaves)
        visit_counts = []
        for i, node in enumerate(self.nodes):
            if node.state_idx == -1 or i in self.free_indices:
                continue
            if not node.child_indices:  # Leaf node
                visit_counts.append((node.visit_count, i))
        
        # Sort by visit count
        visit_counts.sort()
        
        # Prune bottom 25%
        n_prune = len(visit_counts) // 4
        for _, idx in visit_counts[:n_prune]:
            if idx < len(self.nodes):
                parent_idx = self.nodes[idx].parent_idx
                if parent_idx >= 0 and parent_idx < len(self.nodes):
                    parent = self.nodes[parent_idx]
                    if idx in parent.child_indices:
                        parent.child_indices.remove(idx)
                self._free_node(idx)
    
    def clear(self):
        """Clear entire tree"""
        # Reset all nodes (don't create new objects)
        for node in self.nodes:
            node.reset(-1, -1, -1)
        
        self.free_indices = list(range(self.config.max_tree_size))
        self.node_count = 0
        self.root_idx = -1
        
        # Zero out pools
        self.state_pool.zero_()
        self.action_pool.zero_()


class EfficientWorldModelCache:
    """Memory-efficient world model cache with fixed size"""
    
    def __init__(self, config: MCTSConfigV3, device: torch.device):
        self.config = config
        self.device = device
        
        # Fixed-size cache arrays
        self.cache_states = torch.zeros(
            config.cache_size, config.state_dim,
            device=device, dtype=torch.float32
        )
        self.cache_actions = torch.zeros(
            config.cache_size, config.action_dim,
            device=device, dtype=torch.float32
        )
        self.cache_next_states = torch.zeros(
            config.cache_size, config.state_dim,
            device=device, dtype=torch.float32
        )
        
        # Cache metadata
        self.cache_valid = torch.zeros(config.cache_size, dtype=torch.bool, device=device)
        self.cache_index = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_or_compute(self, state: torch.Tensor, action: torch.Tensor, 
                      world_model: Callable) -> torch.Tensor:
        """Get cached prediction or compute new one"""
        # Check cache
        if self.cache_valid.any():
            # Compute distances to cached entries
            state_dists = torch.norm(self.cache_states - state.unsqueeze(0), dim=1)
            action_dists = torch.norm(self.cache_actions - action.unsqueeze(0), dim=1)
            
            # Combined distance
            distances = state_dists + action_dists
            distances[~self.cache_valid] = float('inf')
            
            min_dist, min_idx = distances.min(dim=0)
            
            if min_dist < 0.01:  # Cache hit threshold
                self.cache_hits += 1
                return self.cache_next_states[min_idx].clone()
        
        # Cache miss - compute
        self.cache_misses += 1
        with torch.no_grad():
            # Handle different world model signatures
            try:
                next_state = world_model(state.unsqueeze(0), action.unsqueeze(0))
            except TypeError:
                # Try with context parameter
                next_state = world_model(state.unsqueeze(0), action.unsqueeze(0), None)
            
            if isinstance(next_state, dict):
                next_state = next_state.get('next_state', next_state.get('state', state))
            if next_state.dim() > 1:
                next_state = next_state.squeeze(0)
        
        # Add to cache
        idx = self.cache_index
        self.cache_states[idx] = state
        self.cache_actions[idx] = action
        self.cache_next_states[idx] = next_state
        self.cache_valid[idx] = True
        
        self.cache_index = (self.cache_index + 1) % self.config.cache_size
        
        return next_state
    
    def clear(self):
        """Clear cache"""
        self.cache_valid.zero_()
        self.cache_hits = 0
        self.cache_misses = 0


class LightweightValueNetwork(nn.Module):
    """Lightweight value approximator"""
    
    def __init__(self, config: MCTSConfigV3):
        super().__init__()
        
        # Simplified network
        self.net = nn.Sequential(
            nn.Linear(config.state_dim * 2, config.value_network_hidden),
            nn.ReLU(),
            nn.Linear(config.value_network_hidden, 1)
        )
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Estimate value of state relative to goal"""
        # Ensure proper dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        
        # Debug dimensions
        if state.shape[-1] != goal.shape[-1]:
            raise ValueError(f"State and goal dimensions must match: {state.shape} vs {goal.shape}")
        
        # More debugging
        expected_dim = self.net[0].in_features
        combined = torch.cat([state, goal], dim=-1)
        if combined.shape[-1] != expected_dim:
            raise ValueError(f"Combined state+goal has dimension {combined.shape[-1]}, but network expects {expected_dim}. State: {state.shape}, Goal: {goal.shape}")
        return self.net(combined).squeeze(-1)


class GoalConditionedMCTSV3(BaseAGIModule):
    """
    Memory-managed MCTS planner.
    
    Key improvements:
    - Inherits from BaseAGIModule for memory management
    - Bounded tree structure with automatic pruning
    - Pre-allocated tensor pools
    - Efficient caching with fixed memory
    - Proper cleanup mechanisms
    """
    
    def __init__(self, config: ModuleConfig, mcts_config: MCTSConfigV3, world_model: Callable):
        # Reduce history tracking to prevent memory leak
        config.max_sequence_length = 10
        
        # Store as instance variables (not module attributes)
        self.__dict__['_temp_mcts_config'] = mcts_config
        self.__dict__['_temp_world_model'] = world_model
        
        super().__init__(config)
        
    def _build_module(self):
        """Build all components"""
        # Move temporary attributes to permanent ones
        self.mcts_config = self.__dict__['_temp_mcts_config']
        self.world_model = self.__dict__['_temp_world_model']
        del self.__dict__['_temp_mcts_config']
        del self.__dict__['_temp_world_model']
        
        # MCTS trees pool (pre-allocated)
        self.max_trees = 10
        self.trees = [
            BoundedMCTSTree(self.mcts_config, self.device) 
            for _ in range(self.max_trees)
        ]
        self.tree_index = 0
        
        # World model cache
        self.world_cache = EfficientWorldModelCache(self.mcts_config, self.device)
        
        # Value network
        self.value_network = LightweightValueNetwork(self.mcts_config)
        
        # Action sampling
        self.action_mean = nn.Parameter(torch.zeros(self.mcts_config.action_dim))
        self.action_std = nn.Parameter(torch.ones(self.mcts_config.action_dim))
        
        # Pre-allocate buffers to avoid creating new tensors
        self._action_buffer = torch.zeros(
            self.mcts_config.max_children_per_node,
            self.mcts_config.action_dim,
            device=self.device
        )
        self._eps_buffer = torch.zeros(
            self.mcts_config.action_dim,
            device=self.device
        )
        
        # Statistics (bounded)
        self.planning_stats = self.create_buffer(100)
        
        logger.info(f"GoalConditionedMCTSV3 initialized with max {self.max_trees} trees")
    
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Plan from current state to goal.
        
        Args:
            x: Combined [current_state, goal_state] tensor [batch, state_dim*2]
            
        Returns:
            Dict with planned actions and statistics
        """
        batch_size = x.shape[0]
        
        # Split input into current and goal states
        state_dim = self.mcts_config.state_dim
        if x.shape[-1] != state_dim * 2:
            raise ValueError(f"Expected input of dimension {state_dim * 2}, got {x.shape[-1]}")
        current_states = x[:, :state_dim]
        goal_states = x[:, state_dim:2*state_dim]
        
        # Plan for each example in batch
        all_actions = []
        all_values = []
        
        for i in range(batch_size):
            action, value, stats = self._plan_single(
                current_states[i], 
                goal_states[i],
                kwargs.get('context', {})
            )
            all_actions.append(action)
            all_values.append(value)
            
            # Track stats
            self.planning_stats.append(torch.tensor(stats['simulations'], dtype=torch.float32))
        
        # Stack results
        best_actions = torch.stack(all_actions)
        expected_values = torch.stack(all_values)
        
        # Goal achievement estimation
        distances = torch.norm(current_states - goal_states, dim=-1)
        goal_probs = torch.exp(-distances / 5.0)
        
        return {
            'output': best_actions,  # Best actions to take
            'best_actions': best_actions,
            'expected_values': expected_values,
            'goal_achievement_probs': goal_probs,
            'planning_confidence': torch.ones(batch_size, device=x.device) * 0.8,  # Simplified
            'cache_hit_rate': torch.tensor(
                self.world_cache.cache_hits / max(1, self.world_cache.cache_hits + self.world_cache.cache_misses),
                device=x.device
            )
        }
    
    def _plan_single(self, current_state: torch.Tensor, goal_state: torch.Tensor, 
                    context: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Plan for single state-goal pair"""
        # Get tree from pool
        tree = self.trees[self.tree_index]
        self.tree_index = (self.tree_index + 1) % self.max_trees
        
        # Clear tree for new planning
        tree.clear()
        
        # Initialize root
        root_idx = tree.add_root(current_state)
        
        # Run simulations
        for sim in range(self.mcts_config.n_simulations):
            self._simulate(tree, root_idx, goal_state, context)
        
        # Extract best action
        best_action, expected_value = self._extract_best_action(tree, root_idx)
        
        stats = {
            'simulations': self.mcts_config.n_simulations,
            'tree_size': tree.node_count
        }
        
        return best_action, expected_value, stats
    
    def _simulate(self, tree: BoundedMCTSTree, root_idx: int, 
                 goal_state: torch.Tensor, context: Dict):
        """Run single MCTS simulation"""
        # Selection phase
        path = []
        node_idx = root_idx
        depth = 0
        
        while depth < self.mcts_config.horizon:
            if node_idx < 0 or node_idx >= len(tree.nodes):
                break
                
            node = tree.nodes[node_idx]
            if node.state_idx == -1:  # Node not active
                break
            
            # Check if leaf node
            if not node.child_indices:
                # Expansion phase
                state = tree.state_pool[node.state_idx]
                
                # Sample actions directly into buffer
                n_actions = min(5, self.mcts_config.max_children_per_node)
                self._sample_actions_into_buffer(n_actions)
                
                # Add children using buffer
                for i in range(n_actions):
                    next_state = self.world_cache.get_or_compute(
                        state, self._action_buffer[i], self.world_model
                    )
                    tree.add_child(node_idx, self._action_buffer[i], next_state)
                
                # Select first child for rollout
                if node.child_indices:
                    node_idx = node.child_indices[0]
                    path.append(node_idx)
                break
            else:
                # Select best child
                child_idx, _ = tree.select_best_child(node_idx)
                if child_idx < 0:
                    break
                
                path.append(child_idx)
                node_idx = child_idx
                depth += 1
        
        # Evaluation phase
        if node_idx >= 0 and node_idx < len(tree.nodes):
            node = tree.nodes[node_idx]
            if node.state_idx >= 0:  # Valid node
                leaf_state = tree.state_pool[node.state_idx]
                value = self.value_network(leaf_state, goal_state).item()
                
                # Backup phase
                tree.backup(node_idx, value)
    
    def _sample_actions_into_buffer(self, n_samples: int):
        """Sample actions directly into pre-allocated buffer"""
        for i in range(n_samples):
            # Reuse epsilon buffer
            self._eps_buffer.normal_()
            self._action_buffer[i] = torch.tanh(self.action_mean + self.action_std * self._eps_buffer)
    
    def _extract_best_action(self, tree: BoundedMCTSTree, root_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract best action from tree"""
        if root_idx < 0 or root_idx >= len(tree.nodes):
            return torch.zeros(self.mcts_config.action_dim, device=self.device), torch.tensor(0.0)
            
        root = tree.nodes[root_idx]
        
        if not root.child_indices:
            # No children, return zero action
            return torch.zeros(self.mcts_config.action_dim, device=self.device), torch.tensor(0.0)
        
        # Find child with highest value
        best_value = -float('inf')
        best_action_idx = -1
        
        for child_idx in root.child_indices:
            if child_idx < len(tree.nodes) and tree.nodes[child_idx] is not None:
                child = tree.nodes[child_idx]
                if child.value > best_value:
                    best_value = child.value
                    best_action_idx = child.action_idx
        
        if best_action_idx >= 0:
            return tree.action_pool[best_action_idx].clone(), torch.tensor(best_value)
        
        return torch.zeros(self.mcts_config.action_dim, device=self.device), torch.tensor(0.0)
    
    def _cleanup_impl(self):
        """Clean up module-specific memory"""
        # Clear all trees
        for tree in self.trees:
            tree.clear()
        
        # Clear cache
        self.world_cache.clear()
        
        # Clear stats
        self.planning_stats.clear()
        
        logger.info("GoalConditionedMCTSV3 cleanup completed")
    
    def _reset_impl(self):
        """Reset module state"""
        self._cleanup_impl()
        
        # Reset parameters
        self.action_mean.data.zero_()
        self.action_std.data.fill_(1.0)
    
    def get_planning_summary(self) -> Dict[str, Any]:
        """Get summary of planning statistics"""
        stats_values = self.planning_stats.get_all()
        
        return {
            'avg_simulations': sum(stats_values) / len(stats_values) if stats_values else 0.0,
            'cache_hit_rate': self.world_cache.cache_hits / max(1, self.world_cache.cache_hits + self.world_cache.cache_misses),
            'total_plans': len(stats_values),
            'memory_usage': self.get_memory_usage()
        }


class GoalConditionedMCTSWrapperV3(GoalConditionedMCTSV3):
    """Wrapper for compatibility with existing code"""
    
    def __init__(self, genome: Dict[str, Any], world_model: Callable):
        # Extract configuration
        mcts_config = MCTSConfigV3(
            action_dim=genome.get('action_dim', 4),
            state_dim=genome.get('state_dim', 16),
            n_simulations=genome.get('n_simulations', 50),
            horizon=genome.get('horizon', 10)
        )
        
        config = ModuleConfig(
            name="goal_mcts",
            input_dim=mcts_config.state_dim * 2,  # current + goal
            output_dim=mcts_config.action_dim,
            hidden_dim=mcts_config.value_network_hidden,
            memory_fraction=0.1  # 10% of total memory budget
        )
        
        super().__init__(config, mcts_config, world_model)
        self.genome = genome
    
    def plan_batch(self, initial_states: torch.Tensor, goal_states: torch.Tensor,
                  context: Optional[Dict] = None, return_trees: bool = False) -> Dict[str, Any]:
        """Compatibility method for batch planning"""
        # Combine states for forward pass
        combined = torch.cat([initial_states, goal_states], dim=-1)
        
        # Call forward
        result = self.forward(combined, context=context)
        
        # Add compatibility fields
        result['best_actions'] = result['output']
        result['root_values'] = result['expected_values']
        result['final_distances'] = torch.norm(initial_states - goal_states, dim=-1)
        
        if return_trees:
            # Trees are not directly accessible in V3
            result['trees'] = None
        
        result['simulation_stats'] = self.get_planning_summary()
        
        return result


# Maintain compatibility
GoalConditionedMCTSWrapper = GoalConditionedMCTSWrapperV3

__all__ = ["GoalConditionedMCTSV3", "GoalConditionedMCTSWrapper", "MCTSConfigV3"]