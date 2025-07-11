#!/usr/bin/env python3
"""
Attractor Networks V3 - Refactored with Memory Management
=========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Fixed dynamic attractor memory leaks with pre-allocated buffers
- Bounded history tracking with circular buffers
- Cached sparse connectivity matrices
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

from core.base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from core.error_handling import RobustForward, handle_errors


@dataclass
class BoundedAttractorState:
    """Lightweight attractor state without metadata accumulation"""
    active_count: int
    mean_activity: float
    sparsity: float
    hierarchy_level: int
    

class MemoryEfficientSparsePool(nn.Module):
    """Sparse attractor pool with pre-allocated buffers"""
    
    def __init__(self, pool_size: int, input_dim: int, sparsity: float = 0.1):
        super().__init__()
        self.pool_size = pool_size
        self.input_dim = input_dim
        self.sparsity = sparsity
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, pool_size)
        
        # Sparse connectivity with fixed structure
        n_connections = int(pool_size * pool_size * sparsity)
        indices = torch.zeros(2, n_connections, dtype=torch.long)
        
        # Generate random sparse connections
        for i in range(n_connections):
            indices[0, i] = torch.randint(0, pool_size, (1,))
            indices[1, i] = torch.randint(0, pool_size, (1,))
        
        self.register_buffer('connectivity_indices', indices)
        self.connectivity_values = nn.Parameter(torch.randn(n_connections) * 0.1)
        
        # Pre-allocated buffers
        self.register_buffer('sparse_state', torch.zeros(pool_size))
        self.register_buffer('active_mask', torch.zeros(pool_size, dtype=torch.bool))
        self.register_buffer('dense_buffer', torch.zeros(pool_size))
        self.register_buffer('last_update', torch.tensor(0))
        
        # Cached sparse matrix (recreated only when values change)
        self._cached_sparse_matrix = None
        self._cache_valid = False
        
        # Non-linearity
        self.activation = nn.GELU()
        
    def get_sparse_connectivity(self) -> torch.sparse.FloatTensor:
        """Get cached sparse connectivity matrix"""
        if not self._cache_valid or self._cached_sparse_matrix is None:
            self._cached_sparse_matrix = torch.sparse_coo_tensor(
                self.connectivity_indices,
                self.connectivity_values,
                (self.pool_size, self.pool_size),
                dtype=torch.float32,
                device=self.connectivity_values.device
            )
            self._cache_valid = True
        return self._cached_sparse_matrix
    
    def forward(self, x: torch.Tensor) -> BoundedAttractorState:
        """Forward with pre-allocated buffers"""
        # Input transformation
        input_activation = self.input_projection(x)
        
        # Determine active neurons efficiently
        threshold = torch.quantile(torch.abs(input_activation), 1.0 - self.sparsity)
        self.active_mask = torch.abs(input_activation) > threshold
        
        # Update sparse state in-place
        self.sparse_state.mul_(0.9)  # Decay
        
        # Handle batch dimension properly
        if input_activation.dim() > 1:
            # Use first item in batch for pool state
            input_act = input_activation[0]
            active_mask = self.active_mask[0] if self.active_mask.dim() > 1 else self.active_mask
        else:
            input_act = input_activation
            active_mask = self.active_mask
            
        self.sparse_state[active_mask] = input_act[active_mask]
        
        # Sparse recurrent dynamics
        W_sparse = self.get_sparse_connectivity()
        recurrent = torch.sparse.mm(W_sparse, self.sparse_state.unsqueeze(1)).squeeze()
        
        # Update state with activation
        self.sparse_state = self.activation(self.sparse_state + 0.1 * recurrent)
        
        # Update timestamp
        self.last_update += 1
        
        # Return lightweight state info
        active_count = active_mask.sum().item()
        mean_activity = self.sparse_state[active_mask].mean().item() if active_count > 0 else 0.0
        
        return BoundedAttractorState(
            active_count=active_count,
            mean_activity=mean_activity,
            sparsity=active_count / self.pool_size,
            hierarchy_level=0
        )
    
    def get_dense_state(self) -> torch.Tensor:
        """Get dense representation using pre-allocated buffer"""
        self.dense_buffer.zero_()
        self.dense_buffer.copy_(self.sparse_state)
        return self.dense_buffer
    
    def reset(self):
        """Reset pool state"""
        self.sparse_state.zero_()
        self.active_mask.zero_()
        self.last_update.zero_()
        self._cache_valid = False


class HierarchicalAttractorNetworkV3(BaseAGIModule):
    """
    Refactored Hierarchical Attractor Network with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.n_hierarchies = min(3, getattr(self.config, 'n_hierarchies', 3))
        pool_sizes = [256, 128, 64][:self.n_hierarchies]
        
        # Create hierarchy of sparse pools
        self.pools = nn.ModuleList()
        
        input_dim = self.config.input_dim
        for i, pool_size in enumerate(pool_sizes):
            self.pools.append(MemoryEfficientSparsePool(
                pool_size=pool_size,
                input_dim=input_dim,
                sparsity=0.1 + i * 0.05  # Increasing sparsity
            ))
            input_dim = pool_size
        
        # Inter-level connections with pre-allocated buffers
        self.inter_level_projections = nn.ModuleList()
        # Track expected input dimensions for each pool
        expected_dims = [self.config.input_dim]  # First pool expects config.input_dim
        for i in range(1, self.n_hierarchies):
            expected_dims.append(pool_sizes[i-1])  # Each subsequent pool expects previous pool size
            
        for i in range(self.n_hierarchies - 1):
            # Project from current pool output to next pool input
            self.inter_level_projections.append(
                nn.Linear(pool_sizes[i], expected_dims[i+1])
            )
        
        # Decision network
        self.decision_net = nn.Sequential(
            nn.Linear(sum(pool_sizes), self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.output_dim)
        )
        
        # Bounded history tracking
        self.decision_history = self.create_buffer(100)
        self.hierarchy_stats = self.create_buffer(50)
        
        # Pre-allocated workspace
        self.register_buffer('_concat_buffer', torch.zeros(sum(pool_sizes)))
        self.register_buffer('_hierarchy_states', torch.zeros(self.n_hierarchies, max(pool_sizes)))
        
    @RobustForward()
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded memory usage
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Process through hierarchy
        current_input = x
        attractor_states = []
        dense_representations = []
        
        for level in range(self.n_hierarchies):
            # Process through pool (handle batch dimension)
            # Pools expect single input, so process first item in batch
            pool_input = current_input[0] if current_input.dim() > 1 else current_input
            attractor_state = self.pools[level](pool_input)
            attractor_states.append(attractor_state)
            
            # Get dense representation
            dense_state = self.pools[level].get_dense_state()
            dense_representations.append(dense_state)
            
            # Project to next level if not last
            if level < self.n_hierarchies - 1:
                # Keep the projected state for next level
                current_input = self.inter_level_projections[level](dense_state.unsqueeze(0))
        
        # Combine representations for decision
        combined = self._combine_hierarchical_states(dense_representations)
        
        # Make decision
        decision = self.decision_net(combined)
        
        # Store bounded history
        self.decision_history.append(decision.mean().detach())
        
        # Compute hierarchy statistics
        total_active = sum(state.active_count for state in attractor_states)
        mean_sparsity = np.mean([state.sparsity for state in attractor_states])
        
        self.hierarchy_stats.append(torch.tensor([total_active, mean_sparsity]))
        
        # Periodically reset sparse pools to prevent drift
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
            if self._forward_count % 1000 == 0:
                self._partial_reset()
        else:
            self._forward_count = 0
        
        return {
            'output': decision,
            'n_active_attractors': total_active,
            'mean_sparsity': mean_sparsity,
            'hierarchy_levels': self.n_hierarchies,
            'attractor_states': [
                {
                    'level': i,
                    'active_count': state.active_count,
                    'mean_activity': state.mean_activity,
                    'sparsity': state.sparsity
                }
                for i, state in enumerate(attractor_states)
            ]
        }
    
    def _combine_hierarchical_states(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Combine states using pre-allocated buffer"""
        offset = 0
        for state in states:
            size = state.shape[0]
            self._concat_buffer[offset:offset + size] = state
            offset += size
        
        return self._concat_buffer[:offset].unsqueeze(0)
    
    def _partial_reset(self):
        """Partial reset to prevent long-term drift"""
        for pool in self.pools:
            # Only reset inactive neurons
            inactive_mask = pool.sparse_state.abs() < 0.01
            pool.sparse_state.masked_fill_(inactive_mask, 0)
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear histories
        self.decision_history.clear()
        self.hierarchy_stats.clear()
        
        # Reset pools
        for pool in self.pools:
            pool.reset()
        
        # Reset counter
        self._forward_count = 0
    
    def get_attractor_stats(self) -> Dict[str, Any]:
        """Get attractor network statistics"""
        history = self.decision_history.get_all()
        stats = self.hierarchy_stats.get_all()
        
        if stats:
            stats_array = torch.stack(stats).cpu().numpy()
            mean_active = stats_array[:, 0].mean()
            mean_sparsity = stats_array[:, 1].mean()
        else:
            mean_active = 0
            mean_sparsity = 0
        
        return {
            'decision_history_size': len(history),
            'mean_active_attractors': mean_active,
            'mean_sparsity': mean_sparsity,
            'forward_count': getattr(self, '_forward_count', 0),
            'memory_usage': self.get_memory_usage()
        }


class MetaAttractorNetworkV3(BaseAGIModule):
    """
    Meta-learning attractor network with bounded task memory
    """
    
    def _build_module(self):
        """Build meta-network components"""
        # Base attractor network
        self.base_network = HierarchicalAttractorNetworkV3(self.config)
        
        # Meta-learning components
        meta_dim = self.config.hidden_size
        self.meta_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim + self.config.output_dim, meta_dim),
            nn.LayerNorm(meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, meta_dim)
        )
        
        # Bounded task memory (circular buffer pattern)
        self.max_tasks = 10
        self.register_buffer('task_memory', torch.zeros(self.max_tasks, meta_dim))
        self.register_buffer('task_valid', torch.zeros(self.max_tasks, dtype=torch.bool))
        self.task_idx = 0
        
        # Task adaptation network
        self.task_adapter = nn.Sequential(
            nn.Linear(meta_dim * 2, meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, self.config.output_dim)
        )
        
    @RobustForward()
    def _forward_impl(self, x: torch.Tensor, task_id: Optional[int] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward with task adaptation"""
        # Base network forward
        base_output = self.base_network(x, **kwargs)
        
        # Meta encoding
        meta_input = torch.cat([x, base_output['output']], dim=-1)
        meta_encoding = self.meta_encoder(meta_input)
        
        # Task-specific adaptation
        if task_id is not None and 0 <= task_id < self.max_tasks and self.task_valid[task_id]:
            task_encoding = self.task_memory[task_id]
            adapted_input = torch.cat([meta_encoding, task_encoding.unsqueeze(0)], dim=-1)
            adapted_output = self.task_adapter(adapted_input)
            output = base_output['output'] + 0.1 * adapted_output
        else:
            output = base_output['output']
        
        # Store task encoding (circular buffer)
        if kwargs.get('store_task', False):
            self.task_memory[self.task_idx] = meta_encoding.detach().mean(dim=0)
            self.task_valid[self.task_idx] = True
            self.task_idx = (self.task_idx + 1) % self.max_tasks
        
        return {
            'output': output,
            'base_output': base_output['output'],
            'meta_encoding': meta_encoding,
            'n_active_attractors': base_output['n_active_attractors'],
            'task_memory_usage': self.task_valid.sum().item()
        }
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clean base network
        self.base_network.cleanup()
        
        # Reset task memory
        self.task_memory.zero_()
        self.task_valid.zero_()
        self.task_idx = 0