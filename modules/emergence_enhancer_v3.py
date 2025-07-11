#!/usr/bin/env python3
"""
Emergence Enhancer V3 - Refactored with Memory Management
========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Fixed Lorenz dynamics history accumulation with circular buffers
- Pre-allocated tensors for chaos generation
- Bounded emergence metrics tracking
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

from .base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from .error_handling import RobustForward, handle_errors


@dataclass
class EmergenceConfigV3:
    """Configuration for emergence enhancement"""
    # Lorenz system parameters
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    
    # Emergence settings
    emergence_threshold: float = 0.3
    integration_steps: int = 5
    chaos_scale: float = 0.1
    
    # Memory management
    max_history_size: int = 10
    clear_history_interval: int = 100


class BoundedChaosGenerator(nn.Module):
    """Memory-efficient chaos generator with pre-allocated buffers"""
    
    def __init__(self, state_dim: int, config: EmergenceConfigV3):
        super().__init__()
        self.state_dim = state_dim
        self.config = config
        
        # Lorenz projections (fixed size)
        self.to_lorenz = nn.Linear(state_dim, 3)
        self.from_lorenz = nn.Linear(3, state_dim)
        
        # Pre-allocated buffers for Lorenz dynamics
        self.register_buffer('_lorenz_state', torch.zeros(1, 3))
        self.register_buffer('_dx', torch.zeros(1, 3))
        self.register_buffer('_dy', torch.zeros(1, 3))
        self.register_buffer('_dz', torch.zeros(1, 3))
        
    def generate_chaos(self, state: torch.Tensor) -> torch.Tensor:
        """Generate chaos using pre-allocated buffers"""
        batch_size = state.shape[0]
        
        # Project to Lorenz space
        lorenz_init = self.to_lorenz(state)
        lorenz_states = lorenz_init.chunk(3, dim=-1)
        x = lorenz_states[0].clone()  # Clone to avoid in-place modification of view
        y = lorenz_states[1].clone()
        z = lorenz_states[2].clone()
        
        # Pre-allocated workspace
        if self._lorenz_state.shape[0] != batch_size:
            # Create new tensors with correct batch size instead of expanding
            if self._lorenz_state.shape[0] < batch_size:
                # Repeat for larger batch size
                self._lorenz_state = self._lorenz_state[0:1].repeat(batch_size, 1)
                self._dx = self._dx[0:1].repeat(batch_size, 1)
                self._dy = self._dy[0:1].repeat(batch_size, 1)
                self._dz = self._dz[0:1].repeat(batch_size, 1)
            else:
                # Slice for smaller batch size
                self._lorenz_state = self._lorenz_state[:batch_size].contiguous()
                self._dx = self._dx[:batch_size].contiguous()
                self._dy = self._dy[:batch_size].contiguous()
                self._dz = self._dz[:batch_size].contiguous()
        
        # Lorenz dynamics with proper operations
        for _ in range(self.config.integration_steps):
            # Compute derivatives
            self._dx[:, 0] = self.config.sigma * (y.squeeze(-1) - x.squeeze(-1))
            self._dy[:, 0] = x.squeeze(-1) * (self.config.rho - z.squeeze(-1)) - y.squeeze(-1)
            self._dz[:, 0] = x.squeeze(-1) * y.squeeze(-1) - self.config.beta * z.squeeze(-1)
            
            # Integration step
            dt = 0.01
            x = x + self._dx[:, :1] * dt
            y = y + self._dy[:, :1] * dt
            z = z + self._dz[:, :1] * dt
        
        # Combine final state
        self._lorenz_state[:, 0] = x.squeeze(-1)
        self._lorenz_state[:, 1] = y.squeeze(-1)
        self._lorenz_state[:, 2] = z.squeeze(-1)
        
        # Project back to state space
        chaos_features = self.from_lorenz(self._lorenz_state)
        
        return chaos_features * self.config.chaos_scale


class BoundedEmergenceMetrics(nn.Module):
    """Compute emergence metrics with bounded history"""
    
    def __init__(self, state_dim: int, max_history: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.max_history = max_history
        
        # Pre-allocated circular buffer for state history
        self.register_buffer('state_history', torch.zeros(max_history, state_dim))
        self.register_buffer('history_valid', torch.zeros(max_history, dtype=torch.bool))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        
        # Pre-allocated workspace
        self.register_buffer('_mean_state', torch.zeros(state_dim))
        self.register_buffer('_std_state', torch.zeros(state_dim))
        
    def compute_metrics(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute emergence metrics with bounded memory"""
        batch_size = states.shape[0]
        
        # Add to circular buffer (only first item in batch to save memory)
        if batch_size > 0:
            idx = self.history_ptr.item()
            self.state_history[idx] = states[0].detach()
            self.history_valid[idx] = True
            self.history_ptr = (self.history_ptr + 1) % self.max_history
        
        # Get valid history
        valid_history = self.state_history[self.history_valid]
        
        if len(valid_history) < 2:
            # Not enough history for metrics
            return {
                'diversity': torch.tensor(0.0, device=states.device),
                'complexity': torch.tensor(0.0, device=states.device),
                'emergence': torch.tensor(0.0, device=states.device)
            }
        
        # Compute metrics efficiently
        # Diversity: variance across history
        diversity = valid_history.var(dim=0).mean()
        
        # Complexity: entropy estimate
        self._mean_state = valid_history.mean(dim=0)
        self._std_state = valid_history.std(dim=0).clamp(min=1e-6)
        normalized = (states - self._mean_state) / self._std_state
        complexity = -torch.sum(normalized * torch.log(normalized.abs() + 1e-6), dim=-1).mean()
        
        # Emergence: change from historical average
        emergence = torch.norm(states - self._mean_state, dim=-1).mean()
        
        return {
            'diversity': diversity,
            'complexity': complexity,
            'emergence': emergence
        }
    
    def reset(self):
        """Reset history"""
        self.state_history.zero_()
        self.history_valid.zero_()
        self.history_ptr.zero_()


class EmergenceEnhancerV3(BaseAGIModule):
    """
    Refactored Emergence Enhancer with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.config_v3 = EmergenceConfigV3()
        self.state_dim = self.config.hidden_size
        
        # Bounded chaos generator
        self.chaos_generator = BoundedChaosGenerator(self.state_dim, self.config_v3)
        
        # Bounded emergence metrics
        self.metrics_computer = BoundedEmergenceMetrics(
            self.state_dim, 
            self.config_v3.max_history_size
        )
        
        # Component networks
        self.components = nn.ModuleDict({
            'pattern_former': nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.LayerNorm(self.state_dim),
                nn.GELU(),
                nn.Linear(self.state_dim, self.state_dim)
            ),
            'boundary_dissolver': nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim * 2),
                nn.GELU(),
                nn.Linear(self.state_dim * 2, self.state_dim),
                nn.Dropout(0.1)
            ),
            'coherence_field': nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.LayerNorm(self.state_dim),
                nn.GELU(),
                nn.Linear(self.state_dim, self.state_dim)
            )
        })
        
        # Integration network
        self.integration_net = nn.Sequential(
            nn.Linear(self.state_dim * 4, self.state_dim * 2),
            nn.LayerNorm(self.state_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_dim * 2, self.state_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.config.output_dim)
        )
        
        # Bounded tracking
        self.emergence_history = self.create_buffer(100)
        self._forward_count = 0
        
    @RobustForward()
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded memory usage
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Ensure input has correct dimension
        if x.shape[-1] != self.state_dim:
            x = F.adaptive_avg_pool1d(
                x.unsqueeze(1),
                self.state_dim
            ).squeeze(1)
        
        # Generate bounded chaos
        chaos_features = self.chaos_generator.generate_chaos(x)
        
        # Apply components
        component_outputs = {}
        for name, component in self.components.items():
            component_outputs[name] = component(x)
        
        # Integrate with chaos
        integrated = torch.cat([
            x,
            chaos_features,
            component_outputs['pattern_former'],
            component_outputs['boundary_dissolver']
        ], dim=-1)
        
        # Final integration
        emergence_state = self.integration_net(integrated)
        
        # Add coherence field
        emergence_state = emergence_state + 0.1 * component_outputs['coherence_field']
        
        # Compute metrics
        metrics = self.metrics_computer.compute_metrics(emergence_state)
        
        # Generate output
        output = self.output_projection(emergence_state)
        
        # Track emergence (bounded)
        self.emergence_history.append(metrics['emergence'].detach())
        
        # Periodic cleanup
        self._forward_count += 1
        if self._forward_count % self.config_v3.clear_history_interval == 0:
            self._partial_cleanup()
        
        return {
            'output': output,
            'emergence_state': emergence_state,
            'chaos_contribution': chaos_features.norm(dim=-1).mean(),
            'diversity': metrics['diversity'],
            'complexity': metrics['complexity'],
            'emergence': metrics['emergence'],
            'components': {
                name: out.norm(dim=-1).mean()
                for name, out in component_outputs.items()
            }
        }
    
    def _partial_cleanup(self):
        """Partial cleanup to prevent long-term accumulation"""
        # Reset old history entries in metrics computer
        old_entries = self.metrics_computer.history_valid.sum() > self.config_v3.max_history_size * 0.8
        if old_entries:
            # Keep only recent half
            half = self.config_v3.max_history_size // 2
            self.metrics_computer.history_valid[:-half] = False
    
    def _cleanup_impl(self):
        """Full cleanup of resources"""
        # Clear history
        self.emergence_history.clear()
        
        # Reset metrics computer
        self.metrics_computer.reset()
        
        # Reset counter
        self._forward_count = 0
    
    def get_emergence_stats(self) -> Dict[str, Any]:
        """Get emergence statistics"""
        history = self.emergence_history.get_all()
        
        if history:
            # Handle case where history contains floats instead of tensors
            if isinstance(history[0], torch.Tensor):
                history_array = torch.stack(history).cpu().numpy()
            else:
                history_array = np.array(history)
            mean_emergence = history_array.mean()
            std_emergence = history_array.std()
            trend = np.polyfit(range(len(history_array)), history_array, 1)[0] if len(history_array) > 1 else 0
        else:
            mean_emergence = std_emergence = trend = 0
        
        return {
            'mean_emergence': mean_emergence,
            'std_emergence': std_emergence,
            'emergence_trend': trend,
            'history_size': len(history),
            'forward_count': self._forward_count,
            'memory_usage': self.get_memory_usage()
        }