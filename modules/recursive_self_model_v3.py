#!/usr/bin/env python3
"""
Recursive Self Model V3 - Refactored with Memory Management
===========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Bounded recursion depth with pre-allocated buffers
- No dynamic projection creation
- Efficient coherence computation
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


class BoundedRecursiveEncoder(nn.Module):
    """Recursive encoder with fixed depth and pre-allocated layers"""
    
    def __init__(self, input_dim: int, hidden_dim: int, max_depth: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        
        # Pre-allocate encoders for each depth level
        self.depth_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for i in range(max_depth)
        ])
        
        # Shared projection for all depths
        self.shared_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Pre-allocate state buffers
        self.register_buffer('state_buffer', torch.zeros(max_depth, hidden_dim))
        
    def forward(self, x: torch.Tensor, depth: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with bounded recursion"""
        if depth is None:
            depth = self.max_depth
        else:
            depth = min(depth, self.max_depth)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Pre-allocated buffer for recursive states
        recursive_states = torch.zeros(batch_size, depth, self.hidden_dim, device=device)
        
        # Initial encoding
        current = x
        for d in range(depth):
            # Apply depth-specific encoder
            encoded = self.depth_encoders[d](current)
            recursive_states[:, d] = encoded
            
            # Project for next level
            if d < depth - 1:
                current = self.shared_projection(encoded)
        
        # Combine states across depths
        unified = recursive_states.mean(dim=1)
        
        return {
            'unified_state': unified,
            'recursive_states': recursive_states,
            'depth': depth
        }


class EfficientCoherenceComputer(nn.Module):
    """Compute coherence efficiently without O(n²) operations"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Lightweight coherence network
        self.coherence_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Global coherence estimator
        self.global_coherence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coherence efficiently"""
        batch_size, n_states, hidden_dim = states.shape
        
        # Global coherence from mean state
        mean_state = states.mean(dim=1)
        global_coh = self.global_coherence(mean_state).squeeze(-1)
        
        # Pairwise coherence using anchor sampling (not full O(n²))
        # Sample a subset of pairs to estimate coherence
        n_samples = min(n_states * 2, 10)  # Limit samples
        pair_coherences = []
        
        for _ in range(n_samples):
            # Random pair indices
            idx1 = torch.randint(0, n_states, (batch_size,), device=states.device)
            idx2 = torch.randint(0, n_states, (batch_size,), device=states.device)
            
            # Get states
            state1 = states.gather(1, idx1.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_dim)).squeeze(1)
            state2 = states.gather(1, idx2.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_dim)).squeeze(1)
            
            # Compute coherence
            combined = torch.cat([state1, state2], dim=-1)
            coh = self.coherence_net(combined).squeeze(-1)
            pair_coherences.append(coh)
        
        # Average sampled coherences
        local_coh = torch.stack(pair_coherences).mean(dim=0)
        
        return local_coh, global_coh


class RecursiveSelfModelV3(BaseAGIModule):
    """
    Refactored Recursive Self Model with proper memory management
    """
    
    def _build_module(self):
        """Build module with pre-allocated components"""
        # Configuration
        self.max_recursion_depth = 3  # Hard limit
        self.hidden_dim = self.config.hidden_size
        
        # Fixed-size input projection (no dynamic creation)
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # Bounded recursive encoder
        self.recursive_encoder = BoundedRecursiveEncoder(
            self.hidden_dim,
            self.hidden_dim,
            self.max_recursion_depth
        )
        
        # Meta-cognition LSTM with bounded history
        self.meta_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Efficient coherence computation
        self.coherence_computer = EfficientCoherenceComputer(self.hidden_dim)
        
        # Self-awareness head
        self.self_awareness = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.config.output_dim)
        )
        
        # Bounded history buffer
        self.self_history = self.create_buffer(50)  # Reduced from unbounded
        
        # Pre-allocated LSTM states
        self.register_buffer('lstm_h0', torch.zeros(2, 1, self.hidden_dim))
        self.register_buffer('lstm_c0', torch.zeros(2, 1, self.hidden_dim))
        
    @RobustForward()
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded recursion and memory management
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Project input to fixed dimension
        x_proj = self.input_projection(x)
        
        # Recursive encoding with bounded depth
        recursion_depth = kwargs.get('recursion_depth', self.max_recursion_depth)
        recursion_depth = min(recursion_depth, self.max_recursion_depth)
        
        recursive_output = self.recursive_encoder(x_proj, recursion_depth)
        unified_state = recursive_output['unified_state']
        recursive_states = recursive_output['recursive_states']
        
        # Meta-cognition through LSTM
        # Use only recent states from history (bounded)
        history_states = self.self_history.get_all()
        if history_states and len(history_states) > 0:
            # Take only last state from history
            last_state = history_states[-1]
            # Ensure it matches current batch size
            if last_state.shape[0] != batch_size:
                # Use only unified_state if batch sizes don't match
                lstm_input = unified_state.unsqueeze(1)
            else:
                # Stack last state with current
                lstm_input = torch.stack([last_state, unified_state], dim=1)
        else:
            lstm_input = unified_state.unsqueeze(1)
        
        # Initialize LSTM states
        h0 = self.lstm_h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.lstm_c0.expand(-1, batch_size, -1).contiguous()
        
        # Process through LSTM
        lstm_out, (hn, cn) = self.meta_lstm(lstm_input, (h0, c0))
        meta_state = lstm_out[:, -1]  # Last timestep
        
        # Compute coherence efficiently
        local_coherence, global_coherence = self.coherence_computer(recursive_states)
        
        # Self-awareness computation
        awareness_input = torch.cat([unified_state, meta_state], dim=-1)
        self_awareness_state = self.self_awareness(awareness_input)
        
        # Generate output
        output = self.output_projection(self_awareness_state)
        
        # Store in bounded history
        self.self_history.append(unified_state.detach())
        
        # Compute self-complexity (simplified)
        self_complexity = self._compute_complexity(recursive_states)
        
        return {
            'output': output,
            'unified_self': unified_state,
            'meta_state': meta_state,
            'self_awareness': self_awareness_state,
            'local_coherence': local_coherence,
            'global_coherence': global_coherence,
            'self_complexity': self_complexity,
            'recursion_depth': recursion_depth
        }
    
    def _compute_complexity(self, states: torch.Tensor) -> torch.Tensor:
        """Compute self-complexity metric efficiently"""
        batch_size = states.shape[0]
        
        # Simple complexity based on state variance
        state_variance = states.var(dim=1).mean(dim=-1)
        
        # Normalize to [0, 1]
        complexity = torch.sigmoid(state_variance)
        
        return complexity
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear history
        self.self_history.clear()
        
        # Reset LSTM states
        self.lstm_h0.zero_()
        self.lstm_c0.zero_()
    
    def get_self_analysis(self) -> Dict[str, Any]:
        """Get self-analysis metrics"""
        history = self.self_history.get_all()
        
        if not history:
            return {
                'history_size': 0,
                'mean_variance': 0.0,
                'coherence_trend': 0.0
            }
        
        # Compute statistics over history
        history_tensor = torch.stack(history)
        
        return {
            'history_size': len(history),
            'mean_variance': history_tensor.var(dim=0).mean().item(),
            'coherence_trend': history_tensor.mean(dim=0).std().item(),
            'memory_usage': self.get_memory_usage()
        }