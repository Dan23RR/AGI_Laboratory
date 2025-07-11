#!/usr/bin/env python3
"""
Global Integration Field V3 - Refactored with Memory Management
==============================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Bounded 3D fields with decay and clamping
- Pre-allocated projections for common dimensions
- Efficient field updates without cloning
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math

from .base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from .error_handling import RobustForward, handle_errors


class BoundedField3D(nn.Module):
    """Memory-efficient 3D field with bounded values"""
    
    def __init__(self, field_shape: Tuple[int, int, int], 
                 value_range: Tuple[float, float] = (-10.0, 10.0),
                 decay_rate: float = 0.99):
        super().__init__()
        self.field_shape = field_shape
        self.value_range = value_range
        self.decay_rate = decay_rate
        
        # Pre-allocated field with bounded initialization
        self.register_buffer('field', torch.zeros(field_shape))
        nn.init.normal_(self.field, 0, 0.1)
        self.field.clamp_(*value_range)
        
        # Pre-allocated workspace buffers
        self.register_buffer('_gradient_x', torch.zeros_like(self.field))
        self.register_buffer('_gradient_y', torch.zeros_like(self.field))
        self.register_buffer('_gradient_z', torch.zeros_like(self.field))
        self.register_buffer('_laplacian', torch.zeros_like(self.field))
        
    def update(self, time_derivative: torch.Tensor, integration_rate: float = 0.01):
        """Update field with bounds and decay"""
        # Apply decay to prevent unbounded growth
        self.field.mul_(self.decay_rate)
        
        # Integrate time derivative
        update = time_derivative * integration_rate
        self.field.add_(update)
        
        # Clamp to prevent memory explosion
        self.field.clamp_(*self.value_range)
        
    def compute_gradients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients using pre-allocated buffers"""
        # X gradient
        self._gradient_x[:-1] = self.field[1:] - self.field[:-1]
        self._gradient_x[-1] = 0
        
        # Y gradient
        self._gradient_y[:, :-1] = self.field[:, 1:] - self.field[:, :-1]
        self._gradient_y[:, -1] = 0
        
        # Z gradient
        self._gradient_z[:, :, :-1] = self.field[:, :, 1:] - self.field[:, :, :-1]
        self._gradient_z[:, :, -1] = 0
        
        return self._gradient_x, self._gradient_y, self._gradient_z
    
    def compute_laplacian(self) -> torch.Tensor:
        """Compute Laplacian using pre-allocated buffer"""
        # Reset buffer
        self._laplacian.zero_()
        
        # Second derivatives
        self._laplacian[1:-1] += self.field[2:] - 2 * self.field[1:-1] + self.field[:-2]
        self._laplacian[:, 1:-1] += self.field[:, 2:] - 2 * self.field[:, 1:-1] + self.field[:, :-2]
        self._laplacian[:, :, 1:-1] += self.field[:, :, 2:] - 2 * self.field[:, :, 1:-1] + self.field[:, :, :-2]
        
        return self._laplacian
    
    def get_field_view(self) -> torch.Tensor:
        """Get read-only view of field (no cloning)"""
        return self.field.detach()
    
    def reset(self):
        """Reset field to initial state"""
        nn.init.normal_(self.field, 0, 0.1)
        self.field.clamp_(*self.value_range)


class PreAllocatedProjections(nn.Module):
    """Pre-allocated projections for common dimensions"""
    
    def __init__(self, target_dim: int, common_dims: List[int] = None):
        super().__init__()
        self.target_dim = target_dim
        
        # Pre-allocate projections for common dimensions
        if common_dims is None:
            common_dims = [64, 128, 256, 512, 768, 1024]
        
        self.projections = nn.ModuleDict()
        for dim in common_dims:
            if dim != target_dim:
                self.projections[f'proj_{dim}'] = nn.Linear(dim, target_dim)
        
        # Fallback projection (will be resized if needed)
        self.adaptive_projection = nn.Linear(target_dim, target_dim)
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to target dimension"""
        input_dim = x.shape[-1]
        
        if input_dim == self.target_dim:
            return x
        
        # Use pre-allocated projection if available
        proj_key = f'proj_{input_dim}'
        if proj_key in self.projections:
            return self.projections[proj_key](x)
        
        # Fallback: resize adaptive projection if needed
        if self.adaptive_projection.in_features != input_dim:
            self.adaptive_projection = nn.Linear(input_dim, self.target_dim).to(x.device)
        
        return self.adaptive_projection(x)


class GlobalIntegrationFieldV3(BaseAGIModule):
    """
    Refactored Global Integration Field with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.field_shape = (32, 32, 16)  # Reduced from (64, 64, 32)
        self.input_dim = self.config.hidden_size
        self.integration_rate = 0.01
        self.diffusion_rate = 0.1
        self.nonlinear_gain = 1.5
        
        # Bounded 3D field
        self.bounded_field = BoundedField3D(
            self.field_shape,
            value_range=(-5.0, 5.0),
            decay_rate=0.99
        )
        
        # Pre-allocated projections
        self.input_projections = PreAllocatedProjections(
            np.prod(self.field_shape),
            common_dims=[64, 128, 256, 512, 768, 1024]
        )
        
        # Field dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.GELU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )
        
        # Output encoder
        self.output_encoder = nn.Sequential(
            nn.Linear(np.prod(self.field_shape), self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.output_dim)
        )
        
        # Bounded tracking
        self.energy_history = self.create_buffer(100)
        self.complexity_history = self.create_buffer(100)
        
        # Pre-allocated workspace
        self._workspace_shape = (1, 1) + self.field_shape
        self.register_buffer('_workspace', torch.zeros(self._workspace_shape))
        
    @RobustForward()
    def _forward_impl(self, input_signal: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded field evolution
        """
        if input_signal.dim() == 1:
            input_signal = input_signal.unsqueeze(0)
        batch_size = input_signal.shape[0]
        
        # Project input to field space
        field_input = self.input_projections.project(input_signal)
        field_input = field_input.view(-1, *self.field_shape).mean(0)
        
        # Compute field dynamics
        time_derivative = self._compute_field_dynamics(field_input)
        
        # Update field with bounds
        self.bounded_field.update(time_derivative, self.integration_rate)
        
        # Stabilization check
        field_norm = self.bounded_field.field.norm()
        if field_norm > 100:
            self.bounded_field.field.mul_(100 / field_norm)
        
        # Compute field properties (no cloning)
        properties = self._compute_field_properties()
        
        # Generate output from field state
        field_flat = self.bounded_field.field.flatten()
        output = self.output_encoder(field_flat.unsqueeze(0).expand(batch_size, -1))
        
        # Track bounded history
        self.energy_history.append(properties['energy'].detach())
        self.complexity_history.append(properties['complexity'].detach())
        
        return {
            'output': output,
            'field_energy': properties['energy'],
            'field_complexity': properties['complexity'],
            'synchrony': properties['synchrony'],
            'emergence': properties['emergence'],
            'field_norm': field_norm,
            'n_active_regions': properties['n_active_regions']
        }
    
    def _compute_field_dynamics(self, input_signal: torch.Tensor) -> torch.Tensor:
        """Compute field dynamics using pre-allocated workspace"""
        # Get current field state
        field = self.bounded_field.get_field_view()
        
        # Prepare workspace
        self._workspace[0, 0] = field
        
        # Apply dynamics network
        with torch.cuda.amp.autocast(enabled=False):  # Avoid mixed precision issues
            dynamics_output = self.dynamics_net(self._workspace)
        
        # Compute Laplacian for diffusion
        laplacian = self.bounded_field.compute_laplacian()
        
        # Field equation with bounded nonlinearity
        nonlinear_term = torch.tanh(field * self.nonlinear_gain)
        time_derivative = (
            self.diffusion_rate * laplacian +
            nonlinear_term +
            dynamics_output.squeeze() * 0.1 +
            input_signal * 0.1
        )
        
        return time_derivative
    
    def _compute_field_properties(self) -> Dict[str, torch.Tensor]:
        """Compute field properties without cloning"""
        field = self.bounded_field.get_field_view()
        
        # Energy (L2 norm)
        energy = torch.norm(field)
        
        # Complexity (gradient magnitude)
        grad_x, grad_y, grad_z = self.bounded_field.compute_gradients()
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        complexity = grad_magnitude.mean()
        
        # Synchrony (correlation between regions)
        partitions = self._generate_partitions()
        partition_means = torch.stack([p.mean() for p in partitions])
        synchrony = torch.corrcoef(partition_means).mean()
        
        # Emergence (detect coherent structures)
        n_active = self._detect_active_regions(field)
        emergence = torch.sigmoid(torch.tensor(n_active / 10.0))
        
        return {
            'energy': energy,
            'complexity': complexity,
            'synchrony': synchrony,
            'emergence': emergence,
            'n_active_regions': n_active
        }
    
    def _generate_partitions(self) -> List[torch.Tensor]:
        """Generate field partitions efficiently"""
        field = self.bounded_field.get_field_view()
        x_mid, y_mid, z_mid = [s // 2 for s in self.field_shape]
        
        # 8 octants
        partitions = [
            field[:x_mid, :y_mid, :z_mid],
            field[x_mid:, :y_mid, :z_mid],
            field[:x_mid, y_mid:, :z_mid],
            field[x_mid:, y_mid:, :z_mid],
            field[:x_mid, :y_mid, z_mid:],
            field[x_mid:, :y_mid, z_mid:],
            field[:x_mid, y_mid:, z_mid:],
            field[x_mid:, y_mid:, z_mid:]
        ]
        
        return partitions
    
    def _detect_active_regions(self, field: torch.Tensor, threshold: float = 0.5) -> int:
        """Detect active regions efficiently"""
        # Simple thresholding
        active_mask = field.abs() > threshold
        
        # Count connected components (simplified)
        # Just count 3x3x3 regions with majority active
        n_active = 0
        for i in range(0, field.shape[0] - 2, 3):
            for j in range(0, field.shape[1] - 2, 3):
                for k in range(0, field.shape[2] - 2, 3):
                    region = active_mask[i:i+3, j:j+3, k:k+3]
                    if region.sum() > 13:  # More than half active
                        n_active += 1
        
        return n_active
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear history
        self.energy_history.clear()
        self.complexity_history.clear()
        
        # Reset field periodically
        if hasattr(self, '_cleanup_counter'):
            self._cleanup_counter += 1
            if self._cleanup_counter % 1000 == 0:
                self.bounded_field.reset()
        else:
            self._cleanup_counter = 0
    
    def get_field_stats(self) -> Dict[str, Any]:
        """Get field statistics"""
        energy_hist = self.energy_history.get_all()
        complexity_hist = self.complexity_history.get_all()
        
        field = self.bounded_field.get_field_view()
        
        return {
            'field_shape': self.field_shape,
            'field_min': field.min().item(),
            'field_max': field.max().item(),
            'field_mean': field.mean().item(),
            'field_std': field.std().item(),
            'mean_energy': np.mean(energy_hist) if energy_hist else 0.0,
            'mean_complexity': np.mean(complexity_hist) if complexity_hist else 0.0,
            'history_size': len(energy_hist),
            'memory_usage': self.get_memory_usage()
        }