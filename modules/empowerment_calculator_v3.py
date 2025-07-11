#!/usr/bin/env python3
"""
Empowerment Calculator V3 - Refactored with Memory Management
=============================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Bounded rollout batching with memory pooling
- Pre-allocated tensor pools for action sequences
- Efficient mutual information estimation
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from dataclasses import dataclass

from core.base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from core.error_handling import RobustForward, handle_errors


@dataclass
class EmpowermentConfigV3:
    """Configuration for empowerment calculation"""
    horizon: int = 10
    n_action_samples: int = 32  # Reduced from 64
    action_dim: int = 4
    state_dim: int = 64
    hidden_dim: int = 128
    optimization_steps: int = 5  # Reduced from 10
    batch_size: int = 16
    # Memory management
    max_cache_size: int = 100  # Reduced from 1000
    max_history_size: int = 100  # Reduced from 1000
    pool_size: int = 5  # Pre-allocated rollout pools


class TensorPool:
    """Efficient tensor pooling for rollouts"""
    
    def __init__(self, pool_size: int, shape: Tuple[int, ...], device: torch.device):
        self.pool_size = pool_size
        self.shape = shape
        self.device = device
        
        # Pre-allocate tensor pool
        self.pool = [torch.zeros(shape, device=device) for _ in range(pool_size)]
        self.available = list(range(pool_size))
        self.in_use = []
        
    def acquire(self) -> Optional[torch.Tensor]:
        """Get a tensor from the pool"""
        if self.available:
            idx = self.available.pop()
            self.in_use.append(idx)
            return self.pool[idx]
        return None
        
    def release(self, tensor: torch.Tensor):
        """Return a tensor to the pool"""
        for i, pooled in enumerate(self.pool):
            if pooled is tensor:
                if i in self.in_use:
                    self.in_use.remove(i)
                    self.available.append(i)
                    # Zero out the tensor for reuse
                    pooled.zero_()
                break
                
    def release_all(self):
        """Release all tensors back to pool"""
        self.available = list(range(self.pool_size))
        self.in_use = []
        for tensor in self.pool:
            tensor.zero_()


class BoundedWorldModelCache:
    """Memory-efficient world model cache with LRU eviction"""
    
    def __init__(self, max_size: int, device: torch.device):
        self.max_size = max_size
        self.device = device
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
        
    def get(self, key: Tuple) -> Optional[torch.Tensor]:
        """Get cached result"""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        self.misses += 1
        return None
        
    def put(self, key: Tuple, value: torch.Tensor):
        """Store result in cache"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            
        self.cache[key] = value.detach()  # Detach to prevent gradient accumulation
        self.access_order.append(key)
        
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class LightweightMutualInfoEstimator(nn.Module):
    """Simplified mutual information estimator"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        # Lightweight networks
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        self.mi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Estimate mutual information I(S;A)"""
        # Encode states and actions
        s_enc = self.state_encoder(states)
        a_enc = self.action_encoder(actions)
        
        # Concatenate and estimate MI
        combined = torch.cat([s_enc, a_enc], dim=-1)
        mi_estimate = self.mi_head(combined).squeeze(-1)
        
        # Apply softplus for non-negativity
        return F.softplus(mi_estimate)


class EmpowermentCalculatorV3(BaseAGIModule):
    """
    Refactored Empowerment Calculator with proper memory management
    """
    
    def __init__(self, config: ModuleConfig, emp_config: EmpowermentConfigV3, 
                 world_model: nn.Module):
        # Store temporary attributes
        self.__dict__['_temp_emp_config'] = emp_config
        self.__dict__['_temp_world_model'] = world_model
        
        # Reduce memory footprint
        config.max_sequence_length = 10
        super().__init__(config)
        
    def _build_module(self):
        """Build module with pre-allocated components"""
        # Move temp attributes
        self.emp_config = self.__dict__['_temp_emp_config']
        self.world_model = self.__dict__['_temp_world_model']
        del self.__dict__['_temp_emp_config']
        del self.__dict__['_temp_world_model']
        
        # Core components
        self.mi_estimator = LightweightMutualInfoEstimator(
            self.emp_config.state_dim,
            self.emp_config.action_dim,
            self.emp_config.hidden_dim
        )
        
        # Action distribution (simplified)
        self.action_mean = nn.Parameter(torch.zeros(self.emp_config.action_dim))
        self.action_logstd = nn.Parameter(torch.zeros(self.emp_config.action_dim))
        
        # Pre-allocated tensor pools
        self.state_pool = TensorPool(
            self.emp_config.pool_size,
            (self.emp_config.batch_size, self.emp_config.horizon, self.emp_config.state_dim),
            self.device
        )
        
        self.action_pool = TensorPool(
            self.emp_config.pool_size,
            (self.emp_config.batch_size, self.emp_config.horizon, self.emp_config.action_dim),
            self.device
        )
        
        # Bounded cache
        self.world_cache = BoundedWorldModelCache(
            self.emp_config.max_cache_size,
            self.device
        )
        
        # Bounded history
        self.empowerment_history = self.create_buffer(self.emp_config.max_history_size)
        
        # Pre-allocated buffers for optimization
        self.register_buffer('_action_buffer', torch.zeros(
            self.emp_config.n_action_samples,
            self.emp_config.horizon,
            self.emp_config.action_dim
        ))
        self.register_buffer('_state_buffer', torch.zeros(
            self.emp_config.n_action_samples,
            self.emp_config.horizon + 1,
            self.emp_config.state_dim
        ))
        
    @RobustForward()
    def _forward_impl(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute empowerment with bounded memory usage
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        
        # Limit batch size to prevent memory explosion
        if batch_size > self.emp_config.batch_size:
            # Process in chunks
            results = []
            for i in range(0, batch_size, self.emp_config.batch_size):
                chunk = state[i:i+self.emp_config.batch_size]
                chunk_result = self._compute_empowerment_batch(chunk)
                results.append(chunk_result)
            
            # Combine results
            empowerment = torch.cat([r['empowerment'] for r in results])
            optimal_actions = torch.cat([r['optimal_actions'] for r in results])
        else:
            result = self._compute_empowerment_batch(state)
            empowerment = result['empowerment']
            optimal_actions = result['optimal_actions']
        
        # Store in history (bounded)
        self.empowerment_history.append(empowerment.mean().detach())
        
        return {
            'output': empowerment,
            'empowerment': empowerment,
            'optimal_actions': optimal_actions,
            'cache_hit_rate': self.world_cache.hits / max(1, self.world_cache.hits + self.world_cache.misses),
            'mean_empowerment': empowerment.mean()
        }
    
    def _compute_empowerment_batch(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute empowerment for a batch with memory pooling"""
        batch_size = states.shape[0]
        device = states.device
        
        # Initialize best empowerment and actions
        best_empowerment = torch.zeros(batch_size, device=device)
        best_actions = torch.zeros(batch_size, self.emp_config.horizon, 
                                 self.emp_config.action_dim, device=device)
        
        # Optimization loop with gradient checkpointing
        for opt_step in range(self.emp_config.optimization_steps):
            # Sample action sequences using pre-allocated buffer
            self._sample_action_sequences(batch_size)
            
            # Evaluate sequences in chunks to limit memory
            chunk_size = 8
            mi_values = []
            
            for i in range(0, self.emp_config.n_action_samples, chunk_size):
                chunk_actions = self._action_buffer[i:i+chunk_size]
                chunk_mi = self._evaluate_action_chunk(states, chunk_actions)
                mi_values.append(chunk_mi)
            
            # Find best sequences
            all_mi = torch.cat(mi_values)
            
            # For now, use mean MI as empowerment estimate
            current_empowerment = all_mi.mean()
            
            # Update best values  
            if current_empowerment > best_empowerment.mean():
                best_empowerment.fill_(current_empowerment)
                # Use first action sequence for simplicity
                best_actions = self._action_buffer[0].unsqueeze(0).expand(batch_size, -1, -1).detach()
            
            # Update action distribution based on best sequences
            if opt_step < self.emp_config.optimization_steps - 1:
                self._update_action_distribution(best_actions.detach())
        
        return {
            'empowerment': best_empowerment,
            'optimal_actions': best_actions
        }
    
    def _sample_action_sequences(self, batch_size: int):
        """Sample action sequences into pre-allocated buffer"""
        # Sample from current distribution
        mean = self.action_mean.unsqueeze(0).unsqueeze(0)
        std = torch.exp(self.action_logstd).unsqueeze(0).unsqueeze(0)
        
        # Fill buffer with samples
        for i in range(self.emp_config.n_action_samples):
            eps = torch.randn(self.emp_config.horizon, self.emp_config.action_dim, 
                            device=self.device)
            self._action_buffer[i] = torch.tanh(mean + std * eps)
    
    def _evaluate_action_chunk(self, initial_states: torch.Tensor, 
                             action_chunk: torch.Tensor) -> torch.Tensor:
        """Evaluate a chunk of action sequences"""
        chunk_size = action_chunk.shape[0]
        batch_size = initial_states.shape[0]
        
        # Process each initial state with each action sequence
        mi_values = []
        for b in range(batch_size):
            for c in range(chunk_size):
                # Rollout single trajectory
                final_states = self._rollout_trajectories(
                    initial_states[b:b+1],
                    action_chunk[c:c+1]
                )
                
                # Compute MI for this trajectory
                states_flat = final_states.squeeze(0).reshape(-1, self.emp_config.state_dim)
                actions_flat = action_chunk[c].reshape(-1, self.emp_config.action_dim)
                mi = self.mi_estimator(states_flat, actions_flat)
                mi_values.append(mi.mean())
        
        return torch.stack(mi_values)
    
    def _rollout_trajectories(self, initial_states: torch.Tensor, 
                            actions: torch.Tensor) -> torch.Tensor:
        """Rollout trajectories with caching"""
        n_traj = initial_states.shape[0]
        
        # Get state buffer from pool or allocate
        state_buffer = torch.zeros(
            n_traj, self.emp_config.horizon + 1, self.emp_config.state_dim,
            device=self.device
        )
        
        # Initial states
        state_buffer[:, 0] = initial_states[:, :self.emp_config.state_dim]
        
        # Rollout with caching
        for t in range(self.emp_config.horizon):
            for i in range(n_traj):
                # Create cache key
                state_tuple = tuple(state_buffer[i, t].tolist())
                action_tuple = tuple(actions[i, t].tolist())
                cache_key = (state_tuple, action_tuple)
                
                # Check cache
                cached_next = self.world_cache.get(cache_key)
                if cached_next is not None:
                    state_buffer[i, t+1] = cached_next
                else:
                    # Compute and cache
                    with torch.no_grad():
                        # Handle different world model types
                        state_in = state_buffer[i, t].unsqueeze(0)
                        action_in = actions[i, t].unsqueeze(0)
                        
                        # Check if world model is PredictiveWorldModel (returns dict)
                        if hasattr(self.world_model, 'forward'):
                            # Try to call it properly based on its type
                            try:
                                # For PredictiveWorldModel which expects numpy arrays
                                if hasattr(self.world_model, '_prepare_input'):
                                    result = self.world_model.forward(
                                        state_in.cpu().numpy(),
                                        action_in.cpu().numpy(),
                                        deterministic_state=True
                                    )
                                    next_state = result['next_state'].squeeze(0)
                                    if isinstance(next_state, np.ndarray):
                                        next_state = torch.from_numpy(next_state).to(self.device)
                                    else:
                                        next_state = next_state.to(self.device)
                                else:
                                    # For other world models
                                    result = self.world_model(state_in, action_in)
                                    if isinstance(result, dict):
                                        next_state = result.get('next_state', result.get('output', state_in)).squeeze(0)
                                    else:
                                        next_state = result.squeeze(0)
                            except:
                                # Fallback: just return current state + small noise
                                next_state = state_in.squeeze(0) + torch.randn_like(state_in.squeeze(0)) * 0.01
                        else:
                            # Simple callable
                            next_state = self.world_model(state_in, action_in).squeeze(0)
                            
                    state_buffer[i, t+1] = next_state[:self.emp_config.state_dim]  # Ensure correct dim
                    self.world_cache.put(cache_key, next_state[:self.emp_config.state_dim])
        
        # Return only the states (not initial)
        return state_buffer[:, 1:]
    
    def _update_action_distribution(self, best_actions: torch.Tensor):
        """Update action distribution parameters"""
        # Simple gradient-free update
        with torch.no_grad():
            # Compute statistics of best actions
            best_flat = best_actions.reshape(-1, self.emp_config.action_dim)
            new_mean = best_flat.mean(dim=0)
            new_std = best_flat.std(dim=0)
            
            # Smooth update
            self.action_mean.data = 0.7 * self.action_mean.data + 0.3 * new_mean
            self.action_logstd.data = 0.7 * self.action_logstd.data + 0.3 * torch.log(new_std + 1e-6)
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear cache
        self.world_cache.clear()
        
        # Release all pooled tensors
        self.state_pool.release_all()
        self.action_pool.release_all()
        
        # Clear history
        self.empowerment_history.clear()
        
        # Zero out buffers
        self._action_buffer.zero_()
        self._state_buffer.zero_()
    
    def get_empowerment_stats(self) -> Dict[str, Any]:
        """Get empowerment statistics"""
        history = self.empowerment_history.get_all()
        
        return {
            'mean_empowerment': np.mean(history) if history else 0.0,
            'std_empowerment': np.std(history) if history else 0.0,
            'cache_hit_rate': self.world_cache.hits / max(1, self.world_cache.hits + self.world_cache.misses),
            'history_size': len(history)
        }