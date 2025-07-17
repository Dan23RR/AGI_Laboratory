#!/usr/bin/env python3
"""
Feedback Loop System V3 - Refactored with Memory Management
===========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Pre-allocated projection layers (no dynamic creation)
- Bounded memory with CircularBuffer
- Proper cleanup implementation
- Fixed-size architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import math

from core.base_module import BaseAGIModule, CircularBuffer, ModuleConfig
from core.error_handling import RobustForward, handle_errors


class NeuralMemoryBankV3(nn.Module):
    """Memory bank with pre-allocated buffers and bounded size"""
    
    def __init__(self, capacity: int, feature_dim: int):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        # Pre-allocated buffers - no dynamic growth
        self.register_buffer('memory_bank', torch.zeros(capacity, feature_dim))
        self.register_buffer('priorities', torch.ones(capacity))
        self.register_buffer('timestamps', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('write_position', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.long))
        self.register_buffer('global_timestamp', torch.tensor(0, dtype=torch.long))
        
        # Fixed-size memory compressor
        self.memory_compressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def write_batch(self, experiences: torch.Tensor, priorities: Optional[torch.Tensor] = None):
        """Write batch with circular buffer logic"""
        batch_size = min(experiences.shape[0], self.capacity)  # Prevent overflow
        
        if priorities is None:
            priorities = torch.ones(batch_size, device=experiences.device)
        
        # Compress experiences
        compressed = self.memory_compressor(experiences[:batch_size])
        
        # Calculate write indices
        write_indices = torch.arange(batch_size, device=experiences.device)
        write_indices = (self.write_position + write_indices) % self.capacity
        
        # Write to memory
        self.memory_bank[write_indices] = compressed.detach()  # Detach to prevent gradient accumulation
        self.priorities[write_indices] = priorities[:batch_size].detach()
        self.timestamps[write_indices] = self.global_timestamp
        
        # Update pointers
        self.write_position = (self.write_position + batch_size) % self.capacity
        self.memory_size = torch.clamp(self.memory_size + batch_size, max=self.capacity)
        self.global_timestamp += 1
        
    def sample_batch(self, batch_size: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch with prioritized replay"""
        if self.memory_size == 0:
            return None, None
            
        actual_size = min(batch_size, self.memory_size.item())
        
        # Calculate sampling probabilities
        valid_priorities = self.priorities[:self.memory_size]
        time_weights = torch.exp(-0.01 * (self.global_timestamp - self.timestamps[:self.memory_size]).float())
        
        # Combined priority with temporal decay
        combined_priorities = valid_priorities * time_weights
        probs = F.softmax(combined_priorities / temperature, dim=0)
        
        # Sample indices
        indices = torch.multinomial(probs, actual_size, replacement=False)
        
        return self.memory_bank[indices], indices
        
    def update_priorities(self, indices: torch.Tensor, new_priorities: torch.Tensor):
        """Update priorities for specific indices"""
        self.priorities[indices] = new_priorities.detach()
        
    def clear(self):
        """Clear all memory"""
        self.memory_size.zero_()
        self.write_position.zero_()
        self.global_timestamp.zero_()
        self.priorities.fill_(1.0)


class AdaptiveFeedbackProcessorV3(nn.Module):
    """Fixed-depth processor with pre-allocated layers"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        
        # Pre-allocate all layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # Fixed computation gates
        self.computation_gates = nn.ModuleList([
            nn.Linear(dims[i+1], 1) for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with fixed depth"""
        computation_costs = []
        
        for layer, gate in zip(self.layers, self.computation_gates):
            x = layer(x)
            
            # Compute gate probability
            gate_prob = torch.sigmoid(gate(x))
            computation_costs.append(gate_prob.mean())
            
            # Apply gating
            if self.training:
                mask = torch.bernoulli(gate_prob)
                x = x * mask
            else:
                x = x * gate_prob
        
        total_cost = torch.stack(computation_costs).mean()
        return x, total_cost


class FeedbackLoopSystemV3(BaseAGIModule):
    """
    Refactored feedback loop system with proper memory management
    """
    
    def _build_module(self):
        """Build module with pre-allocated components"""
        # Extract genome parameters
        self.hidden_size = self.config.hidden_size
        self.memory_capacity = min(self.config.memory_limit, 5000)  # Hard limit
        self.num_attention_heads = self.config.num_attention_heads
        
        # Neural memory bank with bounded capacity
        self.memory_bank = NeuralMemoryBankV3(
            capacity=self.memory_capacity,
            feature_dim=self.hidden_size
        )
        
        # Pre-allocated input projections for common dimensions
        self.input_projections = nn.ModuleDict({
            'proj_64': nn.Linear(64, self.hidden_size),
            'proj_128': nn.Linear(128, self.hidden_size),
            'proj_256': nn.Linear(256, self.hidden_size),
            'proj_512': nn.Linear(512, self.hidden_size),
            'proj_1024': nn.Linear(1024, self.hidden_size),
            'proj_default': nn.Linear(self.hidden_size, self.hidden_size)
        })
        
        # State and action projectors
        self.state_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        self.action_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Fixed-size feedback processor
        self.feedback_processor = AdaptiveFeedbackProcessorV3(
            input_dim=self.hidden_size * 3,
            hidden_dim=self.hidden_size * 2,
            num_layers=3  # Fixed depth
        )
        
        # Simplified attention (no hierarchical levels to save memory)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Quality estimator
        self.quality_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 3)  # [immediate, future, uncertainty]
        )
        
        # Output projector
        self.output_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Plasticity controller
        self.plasticity_controller = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Circular buffer for recent experiences
        self.recent_experiences = self.create_buffer(100)
        
    def _get_input_projection(self, dim: int) -> nn.Module:
        """Get appropriate pre-allocated projection"""
        if dim == 64:
            return self.input_projections['proj_64']
        elif dim == 128:
            return self.input_projections['proj_128']
        elif dim == 256:
            return self.input_projections['proj_256']
        elif dim == 512:
            return self.input_projections['proj_512']
        elif dim == 1024:
            return self.input_projections['proj_1024']
        else:
            # For other dimensions, use adaptive pooling + default projection
            return lambda x: self.input_projections['proj_default'](
                F.adaptive_avg_pool1d(x.unsqueeze(1), self.hidden_size).squeeze(1)
            )
    
    @RobustForward()
    def _forward_impl(self, state: torch.Tensor, action: torch.Tensor, 
                     reward: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded memory and pre-allocated components
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Project inputs to hidden size using pre-allocated projections
        if state.shape[-1] != self.hidden_size:
            proj = self._get_input_projection(state.shape[-1])
            state = proj(state)
        state = self.state_projector(state)
        
        if action.shape[-1] != self.hidden_size:
            proj = self._get_input_projection(action.shape[-1])
            action = proj(action)
        action = self.action_projector(action)
        
        # Normalize reward
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)  # [batch_size, 1]
        reward_expanded = reward.expand(batch_size, self.hidden_size)
        reward_encoded = torch.tanh(reward_expanded) * 0.1
        
        # Combine inputs
        experience = torch.cat([state, action, reward_encoded], dim=-1)
        
        # Process experience
        processed_exp, computation_cost = self.feedback_processor(experience)
        
        # Ensure correct dimension
        if processed_exp.shape[-1] != self.hidden_size:
            processed_exp = F.adaptive_avg_pool1d(
                processed_exp.unsqueeze(1), self.hidden_size
            ).squeeze(1)
        
        # Sample from memory
        memory_samples, memory_indices = self.memory_bank.sample_batch(
            min(32, self.memory_bank.memory_size.item()),
            temperature=0.5
        )
        
        if memory_samples is not None and memory_samples.shape[0] > 0:
            # Apply attention
            query = processed_exp.unsqueeze(1)
            keys = memory_samples.unsqueeze(0).expand(batch_size, -1, -1)
            values = keys
            
            attended_memory, _ = self.memory_attention(query, keys, values)
            attended_memory = attended_memory.squeeze(1)
            
            # Combine with residual connection
            integrated = processed_exp + 0.5 * attended_memory
        else:
            integrated = processed_exp
        
        # Estimate quality
        quality_metrics = self.quality_estimator(integrated)
        immediate_value = quality_metrics[:, 0]
        future_value = quality_metrics[:, 1]
        uncertainty = quality_metrics[:, 2]
        
        # Calculate priority
        priority = torch.abs(immediate_value) + 0.5 * torch.abs(future_value) + 0.1 * uncertainty
        
        # Store in memory (bounded)
        self.memory_bank.write_batch(processed_exp.detach(), priority.detach())
        
        # Store in recent buffer
        self.recent_experiences.append(integrated.detach())
        
        # Generate output
        output = self.output_projector(integrated)
        
        # Calculate plasticity
        plasticity = self.plasticity_controller(integrated).squeeze(-1)
        
        # Update memory priorities
        if memory_indices is not None:
            td_error = torch.abs(immediate_value.detach())
            # Expand td_error to match memory_indices size
            if td_error.dim() == 1:
                new_priorities = td_error.mean().expand(memory_indices.shape[0])
            else:
                new_priorities = td_error.mean().expand(memory_indices.shape[0])
            self.memory_bank.update_priorities(memory_indices, new_priorities)
        
        return {
            'output': output,
            'immediate_value': immediate_value,
            'future_value': future_value,
            'uncertainty': uncertainty,
            'plasticity': plasticity,
            'memory_size': self.memory_bank.memory_size.item(),
            'computation_cost': computation_cost,
            'integrated_features': integrated
        }
    
    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Main forward method with input validation"""
        # Handle dict input
        if isinstance(inputs, dict):
            state = inputs.get('state', inputs.get('observation', torch.zeros(1, self.hidden_size)))
            action = inputs.get('action', torch.zeros(1, self.hidden_size))
            reward = inputs.get('reward', torch.zeros(1))
        else:
            # Assume tensor input is state
            state = inputs
            action = torch.zeros_like(state)
            reward = torch.zeros(state.shape[0], device=state.device)
        
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        
        return self._forward_impl(state, action, reward)
    
    def _cleanup_impl(self):
        """Clean up memory and buffers"""
        self.memory_bank.clear()
        self.recent_experiences.clear()
        
        # Clear any cached computations
        if hasattr(self, '_cache'):
            self._cache.clear()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with torch.no_grad():
            valid_size = self.memory_bank.memory_size.item()
            
            if valid_size == 0:
                return {
                    'memory_size': 0,
                    'memory_utilization': 0.0,
                    'avg_priority': 0.0,
                    'avg_age': 0.0
                }
            
            valid_priorities = self.memory_bank.priorities[:valid_size]
            valid_timestamps = self.memory_bank.timestamps[:valid_size]
            age = (self.memory_bank.global_timestamp - valid_timestamps).float()
            
            return {
                'memory_size': valid_size,
                'memory_utilization': valid_size / self.memory_bank.capacity,
                'avg_priority': valid_priorities.mean().item(),
                'avg_age': age.mean().item(),
                'oldest_memory_age': age.max().item() if valid_size > 0 else 0.0,
                'priority_std': valid_priorities.std().item() if valid_size > 1 else 0.0
            }