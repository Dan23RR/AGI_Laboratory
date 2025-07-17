#!/usr/bin/env python3
"""
Counterfactual Reasoner V3 - Refactored with Memory Management
==============================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Bounded ensemble sizes with memory limits
- Efficient parallel processing with batching
- Pre-allocated trajectory buffers
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math

from core.base_module import BaseAGIModule, ModuleConfig, CircularBuffer
from core.error_handling import RobustForward, handle_errors


class BoundedWorldModelEnsemble(nn.Module):
    """Memory-efficient world model ensemble"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_models: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_models = min(n_models, 3)  # Hard limit on ensemble size
        
        # Shared encoder for all models (memory efficient)
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Lightweight heads for each model
        self.model_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, state_dim)
            ) for _ in range(self.n_models)
        ])
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass"""
        # Shared encoding
        combined = torch.cat([state, action], dim=-1)
        shared_features = self.shared_encoder(combined)
        
        # Get predictions from all models
        predictions = []
        for head in self.model_heads:
            pred = head(shared_features)
            predictions.append(pred)
        
        # Return mean prediction
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred


class EfficientActionGenerator(nn.Module):
    """Memory-efficient action generator with bounded samples"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple generator network
        self.generator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # mean and log_std
        )
        
    def forward(self, state: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Generate bounded action samples"""
        batch_size = state.shape[0]
        n_samples = min(n_samples, 10)  # Hard limit
        
        # Get distribution parameters
        params = self.generator(state)
        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-2, 2))  # Bounded std
        
        # Sample actions
        actions = []
        for _ in range(n_samples):
            eps = torch.randn(batch_size, self.action_dim, device=state.device)
            action = torch.tanh(mean + std * eps)
            actions.append(action)
        
        # Stack: [batch_size, n_samples, action_dim]
        return torch.stack(actions, dim=1)


class TrajectoryBuffer:
    """Pre-allocated buffer for trajectory storage"""
    
    def __init__(self, max_trajectories: int, horizon: int, state_dim: int, device: torch.device):
        self.max_trajectories = max_trajectories
        self.horizon = horizon
        self.state_dim = state_dim
        self.device = device
        
        # Pre-allocate buffer
        self.buffer = torch.zeros(max_trajectories, horizon + 1, state_dim, device=device)
        self.valid_mask = torch.zeros(max_trajectories, dtype=torch.bool, device=device)
        self.write_idx = 0
        
    def add_trajectory(self, trajectory: torch.Tensor) -> int:
        """Add trajectory to buffer"""
        idx = self.write_idx % self.max_trajectories
        
        # Handle variable length trajectories
        traj_len = min(trajectory.shape[0], self.horizon + 1)
        self.buffer[idx, :traj_len] = trajectory[:traj_len]
        self.valid_mask[idx] = True
        
        self.write_idx += 1
        return idx
        
    def get_valid_trajectories(self) -> torch.Tensor:
        """Get all valid trajectories"""
        return self.buffer[self.valid_mask]
        
    def clear(self):
        """Clear buffer"""
        self.buffer.zero_()
        self.valid_mask.zero_()
        self.write_idx = 0


class CounterfactualReasonerV3(BaseAGIModule):
    """
    Refactored Counterfactual Reasoner with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.state_dim = self.config.hidden_size
        self.action_dim = self.config.hidden_size // 2
        self.hidden_dim = self.config.hidden_size
        self.horizon = 10  # Fixed horizon
        self.n_counterfactuals = 5  # Reduced from unbounded
        self.max_trajectories = 50  # Hard limit
        
        # Bounded world model ensemble
        self.world_ensemble = BoundedWorldModelEnsemble(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            n_models=3  # Reduced from 5
        )
        
        # Efficient action generators (reduced from 4 to 2)
        self.action_generators = nn.ModuleDict({
            'exploratory': EfficientActionGenerator(self.state_dim, self.action_dim, self.hidden_dim),
            'conservative': EfficientActionGenerator(self.state_dim, self.action_dim, self.hidden_dim)
        })
        
        # Lightweight consequence evaluator
        self.consequence_evaluator = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Counterfactual encoder
        # Make sure input dim matches state_dim (which is hidden_size)
        self.cf_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.config.output_dim)
        )
        
        # Pre-allocated trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(
            self.max_trajectories,
            self.horizon,
            self.state_dim,
            self.device
        )
        
        # Bounded history
        self.cf_history = self.create_buffer(100)
        
    @RobustForward()
    def _forward_impl(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded counterfactual reasoning
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0]
        
        # Clear trajectory buffer for new computation
        self.trajectory_buffer.clear()
        
        # Generate counterfactual actions (bounded)
        cf_actions = self._generate_counterfactual_actions(state)
        
        # Simulate counterfactual trajectories (bounded and batched)
        cf_trajectories = self._simulate_trajectories_batch(state, cf_actions)
        
        # Evaluate consequences efficiently
        consequences = self._evaluate_consequences_batch(state, cf_trajectories)
        
        # Encode counterfactual understanding
        cf_features = self._encode_counterfactuals(cf_trajectories, consequences)
        
        # Combine with current state
        state_encoded = self.cf_encoder(state)
        combined = torch.cat([state_encoded, cf_features], dim=-1)
        
        # Generate output
        output = self.output_network(combined)
        
        # Store in history (bounded)
        self.cf_history.append(consequences.mean().detach())
        
        # Compute counterfactual diversity
        cf_diversity = self._compute_diversity(cf_trajectories)
        
        return {
            'output': output,
            'counterfactual_features': cf_features,
            'consequences': consequences,
            'cf_diversity': cf_diversity,
            'n_counterfactuals': cf_actions.shape[1],
            'mean_consequence': consequences.mean()
        }
    
    def _generate_counterfactual_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate bounded counterfactual actions"""
        batch_size = state.shape[0]
        
        # Generate a fixed number of counterfactuals to avoid dimension mismatches
        # Always generate exactly n_counterfactuals actions
        all_actions = torch.zeros(batch_size, self.n_counterfactuals, self.action_dim, device=state.device)
        
        # Fill with actions from generators
        idx = 0
        samples_per_gen = self.n_counterfactuals // len(self.action_generators)
        remainder = self.n_counterfactuals % len(self.action_generators)
        
        for i, (gen_name, generator) in enumerate(self.action_generators.items()):
            # Add extra sample to first generators if there's a remainder
            n_samples = samples_per_gen + (1 if i < remainder else 0)
            
            if n_samples > 0:
                actions = generator(state, n_samples)
                all_actions[:, idx:idx+n_samples] = actions
                idx += n_samples
        
        return all_actions
    
    def _simulate_trajectories_batch(self, initial_state: torch.Tensor, 
                                   cf_actions: torch.Tensor) -> torch.Tensor:
        """Simulate trajectories in batches for efficiency"""
        batch_size = initial_state.shape[0]
        n_cf = cf_actions.shape[1]  # Number of counterfactuals per batch item
        
        # Pre-allocate trajectory tensor
        trajectories = torch.zeros(
            batch_size, n_cf, self.horizon + 1, self.state_dim,
            device=self.device
        )
        
        # Initial states
        trajectories[:, :, 0] = initial_state.unsqueeze(1).expand(-1, n_cf, -1)
        
        # Simulate in parallel batches
        for t in range(self.horizon):
            # Get current states for all counterfactuals
            current_states = trajectories[:, :, t].reshape(batch_size * n_cf, self.state_dim)
            
            # Sample random actions for future steps
            if t == 0:
                # Use provided counterfactual actions for first step
                # cf_actions shape: [batch_size, n_cf, action_dim]
                # We need to extract first action for each counterfactual
                actions = cf_actions.reshape(batch_size * n_cf, -1)
            else:
                # Use simple random policy for future steps
                actions = torch.tanh(torch.randn(
                    batch_size * n_cf, self.action_dim, device=self.device
                ) * 0.3)
            
            # Predict next states using ensemble
            with torch.no_grad():
                next_states = self.world_ensemble(current_states, actions)
            
            # Store in trajectories
            trajectories[:, :, t + 1] = next_states.reshape(batch_size, n_cf, self.state_dim)
            
            # Add to buffer (sample subset to avoid overflow)
            if t == self.horizon - 1:
                for i in range(min(n_cf, 10)):  # Store max 10 per batch
                    self.trajectory_buffer.add_trajectory(trajectories[0, i])
        
        return trajectories
    
    def _evaluate_consequences_batch(self, initial_state: torch.Tensor,
                                   trajectories: torch.Tensor) -> torch.Tensor:
        """Evaluate consequences efficiently in batches"""
        batch_size, n_cf = trajectories.shape[:2]
        
        # Get final states
        final_states = trajectories[:, :, -1]
        
        # Prepare for consequence evaluation
        initial_expanded = initial_state.unsqueeze(1).expand(-1, n_cf, -1)
        
        # Flatten for batch processing
        initial_flat = initial_expanded.reshape(batch_size * n_cf, self.state_dim)
        final_flat = final_states.reshape(batch_size * n_cf, self.state_dim)
        
        # Evaluate consequences
        combined = torch.cat([initial_flat, final_flat], dim=-1)
        consequences = self.consequence_evaluator(combined).squeeze(-1)
        
        # Reshape back
        consequences = consequences.reshape(batch_size, n_cf)
        
        return consequences
    
    def _encode_counterfactuals(self, trajectories: torch.Tensor,
                               consequences: torch.Tensor) -> torch.Tensor:
        """Encode counterfactual understanding"""
        batch_size = trajectories.shape[0]
        
        # Weighted average of trajectory features by consequences
        # Use softmax to weight by consequence values
        weights = F.softmax(consequences, dim=-1)  # [batch_size, n_cf]
        
        # Get trajectory features (use mean over time)
        traj_features = trajectories.mean(dim=2)  # [batch, n_cf, state_dim]
        
        # Ensure shapes match for weighted combination
        if weights.shape != (batch_size, traj_features.shape[1]):
            # Handle shape mismatch - likely batch_size got confused with n_cf
            if weights.shape[0] == traj_features.shape[1] and weights.shape[1] == batch_size:
                weights = weights.transpose(0, 1)
            else:
                # Fallback: use uniform weights
                weights = torch.ones(batch_size, traj_features.shape[1], device=weights.device) / traj_features.shape[1]
        
        # Weighted combination with proper broadcasting
        weights = weights.unsqueeze(-1)  # [batch_size, n_cf, 1]
        cf_features = (traj_features * weights).sum(dim=1)  # [batch_size, state_dim]
        
        # Ensure cf_features has correct shape [batch_size, state_dim]
        if cf_features.shape[0] != batch_size:
            # Handle case where dimensions got mixed up
            cf_features = cf_features.reshape(batch_size, -1)
            if cf_features.shape[-1] != self.state_dim:
                # Project to correct dimension
                cf_features = F.adaptive_avg_pool1d(cf_features.unsqueeze(1), self.state_dim).squeeze(1)
        
        # Encode
        cf_encoded = self.cf_encoder(cf_features)
        
        return cf_encoded
    
    def _compute_diversity(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute diversity of counterfactual trajectories"""
        batch_size, n_cf = trajectories.shape[:2]
        
        # Use final states for diversity computation
        final_states = trajectories[:, :, -1]  # [batch, n_cf, state_dim]
        
        # Compute pairwise distances (simplified)
        # Instead of full O(n²), compute distance to mean
        mean_state = final_states.mean(dim=1, keepdim=True)
        distances = torch.norm(final_states - mean_state, dim=-1)
        
        # Average distance as diversity metric
        diversity = distances.mean(dim=1)
        
        return diversity
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear trajectory buffer
        self.trajectory_buffer.clear()
        
        # Clear history
        self.cf_history.clear()
    
    def get_counterfactual_stats(self) -> Dict[str, Any]:
        """Get counterfactual reasoning statistics"""
        history = self.cf_history.get_all()
        
        return {
            'mean_consequence': np.mean(history) if history else 0.0,
            'std_consequence': np.std(history) if history else 0.0,
            'n_trajectories_stored': self.trajectory_buffer.valid_mask.sum().item(),
            'history_size': len(history),
            'memory_usage': self.get_memory_usage()
        }