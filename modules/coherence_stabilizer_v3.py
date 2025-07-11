#!/usr/bin/env python3
"""
Coherence Stabilizer V3 - Refactored with Memory Management
==========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Fixed LSTM states not being cleaned with explicit reset
- Pre-allocated tensors for attention computations
- Efficient in-place operations for memory bank
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


class ManagedLSTMPredictor(nn.Module):
    """LSTM predictor with explicit state management"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM with state management
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Pre-allocated hidden states
        self.register_buffer('h0', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('c0', torch.zeros(num_layers, 1, hidden_dim))
        
        # Track if states need reset
        self._states_initialized = False
        self._current_batch_size = 0
        
    def forward(self, x: torch.Tensor, reset_states: bool = False) -> torch.Tensor:
        """Forward with managed LSTM states"""
        batch_size = x.shape[0]
        
        # Reset states if requested or batch size changed
        if reset_states or batch_size != self._current_batch_size or not self._states_initialized:
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
            self._current_batch_size = batch_size
            self._states_initialized = True
        else:
            # Use existing states
            h0, c0 = self._last_states
        
        # LSTM forward
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Store states (detached to prevent gradient accumulation)
        self._last_states = (hn.detach(), cn.detach())
        
        return lstm_out
    
    def reset_states(self):
        """Explicitly reset LSTM states"""
        self._states_initialized = False
        self._current_batch_size = 0
        if hasattr(self, '_last_states'):
            del self._last_states


class EfficientAttention(nn.Module):
    """Memory-efficient attention with pre-allocated buffers"""
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Attention projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Pre-allocated workspace
        self.register_buffer('_scores_buffer', torch.zeros(1, max_seq_len))
        self.register_buffer('_weights_buffer', torch.zeros(1, max_seq_len))
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with pre-allocated buffers"""
        batch_size = query.shape[0]
        seq_len = keys.shape[1]
        
        # Project
        Q = self.query_proj(query)
        K = self.key_proj(keys)
        V = self.value_proj(values)
        
        # Compute scores efficiently
        scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)).squeeze(1)
        scores = scores / math.sqrt(self.hidden_dim)
        
        # Use pre-allocated buffer if possible
        if batch_size == 1 and seq_len <= self.max_seq_len:
            self._weights_buffer[:, :seq_len] = F.softmax(scores[:, :seq_len], dim=-1)
            weights = self._weights_buffer[:, :seq_len]
        else:
            weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(weights.unsqueeze(1), V).squeeze(1)
        
        return attended, weights.detach()  # Detach weights to prevent gradient accumulation


class BoundedMemoryBank(nn.Module):
    """Memory bank with bounded storage and efficient updates"""
    
    def __init__(self, memory_size: int, state_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.state_dim = state_dim
        
        # Pre-allocated memory
        self.register_buffer('memory_keys', torch.randn(memory_size, state_dim) * 0.01)
        self.register_buffer('memory_values', torch.randn(memory_size, state_dim) * 0.01)
        self.register_buffer('access_counts', torch.zeros(memory_size))
        self.register_buffer('last_update', torch.zeros(memory_size))
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))
        
        # Attention for memory retrieval
        self.memory_attention = EfficientAttention(state_dim)
        
    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from memory with attention"""
        batch_size = query.shape[0]
        # Efficient retrieval - expand memory to match batch size
        retrieved, weights = self.memory_attention(
            query,
            self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        # Update access counts (bounded)
        top_k = min(5, self.memory_size)
        # Use mean over batch for access count update
        mean_weights = weights.mean(dim=0)
        _, top_indices = torch.topk(mean_weights.squeeze(), top_k)
        self.access_counts[top_indices] += 1
        self.access_counts.clamp_(max=1000)  # Prevent overflow
        
        return retrieved, weights
    
    def update(self, state: torch.Tensor, value: torch.Tensor, alpha: float = 0.1):
        """Update memory with momentum and bounds"""
        # Find least recently used slot
        scores = self.access_counts / (self.time_step - self.last_update + 1)
        min_idx = torch.argmin(scores)
        
        # Update with momentum (in-place to save memory)
        # Use only first element from batch for memory update
        state_single = state[0] if state.dim() > 1 else state
        value_single = value[0] if value.dim() > 1 else value
        self.memory_keys[min_idx].mul_(1 - alpha).add_(state_single * alpha)
        self.memory_values[min_idx].mul_(1 - alpha).add_(value_single * alpha)
        
        # Update metadata
        self.access_counts[min_idx] = 1
        self.last_update[min_idx] = self.time_step
        self.time_step += 1
        
        # Periodic reset to prevent overflow
        if self.time_step > 10000:
            self.time_step.fill_(0)
            self.last_update.fill_(0)


class CoherenceStabilizerV3(BaseAGIModule):
    """
    Refactored Coherence Stabilizer with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.state_dim = self.config.hidden_size
        self.history_size = 50  # Reduced from unbounded
        self.memory_size = 64   # Bounded memory bank
        
        # Managed LSTM predictor
        self.trend_predictor = ManagedLSTMPredictor(
            self.state_dim,
            self.state_dim,
            num_layers=2
        )
        
        # Bounded memory bank
        self.memory_bank = BoundedMemoryBank(self.memory_size, self.state_dim)
        
        # State attention (reuse efficient attention)
        self.state_attention = EfficientAttention(self.state_dim)
        
        # Coherence detector
        self.coherence_detector = nn.Sequential(
            nn.Linear(self.state_dim * 3, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_dim, 1),
            nn.Sigmoid()
        )
        
        # Stabilization network
        self.stabilizer = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.config.output_dim)
        )
        
        # Bounded circular history (renamed to avoid conflict with base class)
        self.register_buffer('coherence_state_history', torch.zeros(self.history_size, self.state_dim))
        self.register_buffer('coherence_history_valid', torch.zeros(self.history_size, dtype=torch.bool))
        self.register_buffer('coherence_history_ptr', torch.tensor(0, dtype=torch.long))
        
        # Pre-allocated workspace
        self.register_buffer('_coherence_tensor', torch.zeros(1))
        
        # Tracking
        self.coherence_history = self.create_buffer(100)
        self._forward_count = 0
        
    @RobustForward()
    def _forward_impl(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with managed memory
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        
        # Add to circular history
        idx = self.coherence_history_ptr.item()
        self.coherence_state_history[idx] = state[0].detach()
        self.coherence_history_valid[idx] = True
        self.coherence_history_ptr = (self.coherence_history_ptr + 1) % self.history_size
        
        # Get valid history for trend prediction
        valid_history = self.coherence_state_history[self.coherence_history_valid]
        
        if len(valid_history) > 1:
            # Predict trend with managed LSTM
            history_seq = valid_history.unsqueeze(0)  # [1, seq_len, dim]
            reset_lstm = kwargs.get('reset_lstm', False) or self._forward_count % 100 == 0
            predicted_trend = self.trend_predictor(history_seq, reset_states=reset_lstm)
            predicted_trend = predicted_trend[:, -1]  # Last timestep [1, dim]
            # Expand to match batch size
            predicted_trend = predicted_trend.expand(batch_size, -1)
            
            # Attend to historical states
            attended_history, _ = self.state_attention(
                state,
                valid_history.unsqueeze(0).expand(batch_size, -1, -1),
                valid_history.unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            predicted_trend = state
            attended_history = state
        
        # Retrieve from memory bank
        retrieved_memory, _ = self.memory_bank.retrieve(state)
        
        # Detect coherence
        coherence_input = torch.cat([state, predicted_trend, attended_history], dim=-1)
        coherence = self.coherence_detector(coherence_input)
        
        # Stabilize if needed
        # Handle batch dimension properly
        if coherence.mean().item() < 0.5:
            stab_input = torch.cat([state, retrieved_memory], dim=-1)
            stabilized = self.stabilizer(stab_input)
            output_state = 0.7 * state + 0.3 * stabilized
        else:
            output_state = state
        
        # Update memory bank
        self.memory_bank.update(state, output_state, alpha=0.1)
        
        # Generate output
        output = self.output_projection(output_state)
        
        # Track coherence (bounded)
        self.coherence_history.append(coherence.detach().squeeze())
        
        # Periodic cleanup
        self._forward_count += 1
        if self._forward_count % 1000 == 0:
            self._periodic_cleanup()
        
        return {
            'output': output,
            'coherence': coherence.squeeze(),
            'stabilized_state': output_state,
            'predicted_trend': predicted_trend,
            'history_size': self.coherence_history_valid.sum().item(),
            'was_stabilized': coherence.mean().item() < 0.5
        }
    
    def _periodic_cleanup(self):
        """Periodic cleanup to prevent long-term accumulation"""
        # Reset LSTM states
        self.trend_predictor.reset_states()
        
        # Clear old history entries
        if self.coherence_history_valid.sum() > self.history_size * 0.8:
            # Keep only recent half
            half = self.history_size // 2
            self.coherence_history_valid[:half] = False
    
    def _cleanup_impl(self):
        """Full cleanup of resources"""
        # Reset LSTM
        self.trend_predictor.reset_states()
        
        # Clear history
        self.coherence_state_history.zero_()
        self.coherence_history_valid.zero_()
        self.coherence_history_ptr.zero_()
        self.coherence_history.clear()
        
        # Reset counters
        self._forward_count = 0
    
    def get_coherence_stats(self) -> Dict[str, Any]:
        """Get coherence statistics"""
        coh_history = self.coherence_history.get_all()
        
        if coh_history:
            coh_array = torch.stack(coh_history).cpu().numpy()
            mean_coherence = coh_array.mean()
            std_coherence = coh_array.std()
            stability_ratio = (coh_array > 0.5).mean()
        else:
            mean_coherence = std_coherence = stability_ratio = 0
        
        return {
            'mean_coherence': mean_coherence,
            'std_coherence': std_coherence,
            'stability_ratio': stability_ratio,
            'history_size': self.coherence_history_valid.sum().item(),
            'memory_utilization': self.memory_bank.access_counts.sum().item() / self.memory_size,
            'forward_count': self._forward_count,
            'memory_usage': self.get_memory_usage()
        }