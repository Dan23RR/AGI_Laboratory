#!/usr/bin/env python3
"""
Dynamic Conceptual Field V3 - Refactored with Memory Management
==============================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Bounded 3D fields (no unbounded growth)
- Pre-allocated peak detection buffers
- Simplified architecture for efficiency
- Proper cleanup implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math

from .base_module import BaseAGIModule, CircularBuffer, ModuleConfig
from .error_handling import RobustForward, handle_errors


class BoundedFieldKernel(nn.Module):
    """Simplified interaction kernel with fixed size"""
    
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Create Mexican hat kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = x.unsqueeze(1)
        z = x.unsqueeze(1).unsqueeze(1)
        
        # 3D distance
        dist_sq = x**2 + y**2 + z**2
        
        # Mexican hat formula
        excitation = torch.exp(-dist_sq / 4.0)
        inhibition = 0.5 * torch.exp(-dist_sq / 16.0)
        kernel = excitation - inhibition
        
        # Normalize
        kernel = kernel / kernel.abs().sum()
        
        # Register as buffer
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Apply convolution with fixed kernel"""
        # Ensure 5D: [batch, channel, depth, height, width]
        if field.dim() == 3:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 4:
            field = field.unsqueeze(1)
            
        # Apply convolution with padding
        padding = self.kernel_size // 2
        return F.conv3d(field, self.kernel, padding=padding).squeeze(1)


class SimplifiedConceptBank(nn.Module):
    """Simplified concept storage with bounded size"""
    
    def __init__(self, num_concepts: int = 128, concept_dim: int = 64):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        
        # Fixed concept prototypes
        self.concepts = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.1)
        
        # Concept to field projection (simplified)
        self.concept_to_field = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.GELU(),
            nn.Linear(concept_dim * 2, concept_dim)
        )
        
        # Concept matcher
        self.concept_matcher = nn.Linear(concept_dim, num_concepts)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match features to concepts and get field contribution"""
        # Project features to concept dimension
        if features.shape[-1] != self.concept_dim:
            features = F.adaptive_avg_pool1d(
                features.unsqueeze(1), self.concept_dim
            ).squeeze(1)
        
        # Match to concepts
        concept_scores = self.concept_matcher(features)
        concept_weights = F.softmax(concept_scores, dim=-1)
        
        # Get weighted concept combination
        weighted_concepts = torch.matmul(concept_weights, self.concepts)
        
        # Project to field contribution
        field_contribution = self.concept_to_field(weighted_concepts)
        
        return field_contribution, concept_weights


class DynamicConceptualFieldV3(BaseAGIModule):
    """
    Refactored Dynamic Conceptual Field with proper memory management
    """
    
    def _build_module(self):
        """Build module with pre-allocated components"""
        # Field parameters (smaller for memory efficiency)
        self.field_spatial = 16  # For compatibility
        self.field_size = (self.field_spatial, self.field_spatial, 8)  # Reduced from (32, 32, 16)
        self.field_volume = np.prod(self.field_size)
        
        # Core parameters
        self.tau = 10.0  # Time constant
        self.resting_level = -5.0
        self.noise_strength = 0.01
        
        # Pre-allocated field buffers
        self.register_buffer('field_state', torch.zeros(1, *self.field_size))
        self.register_buffer('field_input', torch.zeros(1, *self.field_size))
        self.register_buffer('adaptation', torch.zeros(1, *self.field_size))
        
        # Interaction kernel (fixed size)
        self.kernel = BoundedFieldKernel(kernel_size=5)
        
        # Simplified concept bank
        self.concept_bank = SimplifiedConceptBank(
            num_concepts=128,
            concept_dim=self.config.hidden_size  # Match module hidden size
        )
        
        # Field encoder/decoder
        self.field_encoder = nn.Sequential(
            nn.Linear(self.field_volume, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.config.hidden_size)
        )
        
        self.field_decoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, self.field_volume)
        )
        
        # Peak detector network
        self.peak_detector = nn.Conv3d(
            1, 8, kernel_size=3, padding=1
        )
        
        # Bounded buffers for peaks and patterns
        self.peak_buffer = self.create_buffer(50)  # Max 50 peaks
        self.pattern_buffer = self.create_buffer(100)  # Max 100 patterns
        
        # Time step counter
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))
        
    @RobustForward()
    def _forward_impl(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded field dynamics
        """
        # Handle input format
        if isinstance(inputs, torch.Tensor):
            features = inputs
        else:
            features = inputs.get('features', inputs.get('observation', None))
            if features is None:
                raise ValueError("Input must contain 'features' or 'observation'")
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        batch_size = features.shape[0]
        
        # Ensure field state matches batch size
        if self.field_state.shape[0] != batch_size:
            # Create new tensors with correct batch size instead of expanding
            device = self.field_state.device
            
            # If batch size increased, repeat the first element
            if self.field_state.shape[0] < batch_size:
                self.field_state = self.field_state[0:1].repeat(batch_size, 1, 1, 1)
                self.field_input = self.field_input[0:1].repeat(batch_size, 1, 1, 1)
                self.adaptation = self.adaptation[0:1].repeat(batch_size, 1, 1, 1)
            else:
                # If batch size decreased, slice to match
                self.field_state = self.field_state[:batch_size].contiguous()
                self.field_input = self.field_input[:batch_size].contiguous()
                self.adaptation = self.adaptation[:batch_size].contiguous()
        
        # 1. Get concept contribution
        concept_contrib, concept_weights = self.concept_bank(features)
        
        # 2. Project to field space
        field_signal = self.field_decoder(concept_contrib)
        field_signal = field_signal.view(batch_size, *self.field_size)
        
        # 3. Update field input
        self.field_input = 0.9 * self.field_input + 0.1 * field_signal
        
        # 4. Compute field dynamics
        interaction = self.kernel(self.field_state)
        
        # 5. Update field state (bounded by tanh)
        field_derivative = (
            -self.field_state + 
            self.resting_level + 
            self.field_input + 
            interaction.squeeze(1) - 
            self.adaptation
        ) / self.tau
        
        # Add small noise for exploration
        noise = torch.randn_like(self.field_state) * self.noise_strength
        
        # Update with Euler method
        self.field_state = torch.tanh(
            self.field_state + field_derivative * 0.1 + noise
        )
        
        # 6. Update adaptation (slow negative feedback)
        self.adaptation = 0.99 * self.adaptation + 0.01 * torch.relu(self.field_state)
        
        # 7. Detect peaks (simplified)
        peaks = self._detect_peaks_simple(self.field_state)
        
        # 8. Extract thoughts from peaks
        thoughts = self._extract_thoughts(self.field_state, peaks)
        
        # 9. Encode field state
        field_flat = self.field_state.view(batch_size, -1)
        encoded_field = self.field_encoder(field_flat)
        
        # 10. Update time step
        self.time_step += 1
        
        return {
            'output': encoded_field,
            'field_state': self.field_state,
            'num_peaks': len(peaks),
            'concept_weights': concept_weights,
            'field_energy': self.field_state.abs().mean(),
            'thoughts': thoughts,
            'time_step': self.time_step.item()
        }
    
    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Main forward method"""
        return self._forward_impl(inputs)
    
    def _detect_peaks_simple(self, field: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Simplified peak detection"""
        batch_size = field.shape[0]
        peaks = []
        
        # Use max pooling to find local maxima
        pooled = F.max_pool3d(
            field.unsqueeze(1), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).squeeze(1)
        
        # Find where original equals pooled (local maxima)
        is_peak = (field == pooled) & (field > 0.5)  # Threshold
        
        # Get peak coordinates (limit to top 10 per batch)
        for b in range(batch_size):
            peak_coords = torch.nonzero(is_peak[b], as_tuple=False)
            if len(peak_coords) > 0:
                # Sort by field value and take top 10
                peak_values = field[b][is_peak[b]]
                top_indices = torch.topk(peak_values, min(10, len(peak_values)))[1]
                top_coords = peak_coords[top_indices]
                
                for coord in top_coords:
                    peaks.append(tuple(coord.tolist()))
                    
                    # Store in buffer (bounded)
                    if len(self.peak_buffer) < self.peak_buffer.max_size:
                        self.peak_buffer.append(coord.float())
        
        return peaks
    
    def _extract_thoughts(self, field: torch.Tensor, peaks: List[Tuple[int, int, int]]) -> List[Dict[str, Any]]:
        """Extract thought patterns from peaks"""
        thoughts = []
        
        for peak in peaks[:5]:  # Limit to 5 thoughts
            # Get field value at peak
            field_value = field[0, peak[0], peak[1], peak[2]].item()
            
            # Create thought representation
            thought = {
                'position': peak,
                'activation': field_value,
                'stability': 1.0 / (1.0 + self.adaptation[0, peak[0], peak[1], peak[2]].item()),
                'time': self.time_step.item()
            }
            thoughts.append(thought)
            
            # Store pattern (bounded)
            if len(self.pattern_buffer) < self.pattern_buffer.max_size:
                pattern_vec = torch.tensor([
                    peak[0] / self.field_size[0],
                    peak[1] / self.field_size[1], 
                    peak[2] / self.field_size[2],
                    field_value
                ])
                self.pattern_buffer.append(pattern_vec)
        
        return thoughts
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Reset field states
        self.field_state.zero_()
        self.field_input.zero_()
        self.adaptation.zero_()
        self.time_step.zero_()
        
        # Clear buffers
        self.peak_buffer.clear()
        self.pattern_buffer.clear()
    
    def get_field_stats(self) -> Dict[str, Any]:
        """Get field statistics"""
        with torch.no_grad():
            return {
                'field_mean': self.field_state.mean().item(),
                'field_std': self.field_state.std().item(),
                'field_energy': self.field_state.abs().mean().item(),
                'adaptation_level': self.adaptation.mean().item(),
                'num_stored_peaks': len(self.peak_buffer),
                'num_stored_patterns': len(self.pattern_buffer),
                'time_steps': self.time_step.item()
            }