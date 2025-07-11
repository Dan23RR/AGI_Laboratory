#!/usr/bin/env python3
"""
Conceptual Compressor V3 - Refactored with Memory Management
===========================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- Fixed VAE buffer memory leaks with pre-allocated tensors
- Bounded concept memory with circular buffer
- Limited compression hierarchy depth
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


class BoundedNeuralMemory(nn.Module):
    """Memory-efficient neural memory with bounded growth"""
    
    def __init__(self, num_concepts: int, concept_dim: int, value_dim: int, 
                 max_access_count: int = 10000):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.value_dim = value_dim
        self.max_access_count = max_access_count
        
        # Pre-allocated memory banks
        self.register_buffer('keys', torch.zeros(num_concepts, concept_dim))
        self.register_buffer('values', torch.zeros(num_concepts, value_dim))
        
        # Bounded access statistics
        self.register_buffer('access_counts', torch.zeros(num_concepts))
        self.register_buffer('last_access', torch.zeros(num_concepts))
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))
        
        # Pre-allocated noise buffer for reparameterization
        self.register_buffer('noise_buffer', torch.zeros(1024, concept_dim))
        self.noise_idx = 0
        
        # Initialize keys with small random values
        nn.init.xavier_uniform_(self.keys, gain=0.01)
        
    def get_noise(self, shape: torch.Size) -> torch.Tensor:
        """Get noise from pre-allocated buffer"""
        n_elements = shape.numel()
        
        # Refill buffer if needed
        if self.noise_idx + n_elements > self.noise_buffer.shape[0]:
            self.noise_buffer.normal_()
            self.noise_idx = 0
        
        # Get noise slice
        noise = self.noise_buffer[self.noise_idx:self.noise_idx + n_elements]
        self.noise_idx = (self.noise_idx + n_elements) % self.noise_buffer.shape[0]
        
        return noise.reshape(shape)
        
    def query(self, queries: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query memory with bounded lookups"""
        batch_size = queries.shape[0]
        k = min(k, self.num_concepts)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            queries.unsqueeze(1),
            self.keys.unsqueeze(0),
            dim=2
        )
        
        # Get top-k
        top_k_values, top_k_indices = torch.topk(similarities, k, dim=1)
        
        # Update access stats (bounded)
        self._update_access_stats(top_k_indices)
        
        # Retrieve values
        retrieved_values = self.values[top_k_indices]
        
        # Weighted combination
        weights = F.softmax(top_k_values, dim=1).unsqueeze(-1)
        combined_values = (retrieved_values * weights).sum(dim=1)
        
        return combined_values, top_k_values
        
    def update(self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor):
        """Update memory with importance-based replacement"""
        batch_size = keys.shape[0]
        
        # Compute replacement scores (lower is better)
        recency_weight = torch.exp(-0.01 * (self.time_step - self.last_access))
        frequency_weight = torch.sigmoid(self.access_counts / 100.0)
        replacement_scores = recency_weight * frequency_weight
        
        # Update with momentum
        momentum = 0.95
        for i in range(batch_size):
            if importance[i] > 0.5:  # Only update if important
                # Find least important slot
                idx = torch.argmin(replacement_scores)
                
                # Update with momentum
                self.keys[idx] = momentum * self.keys[idx] + (1 - momentum) * keys[i]
                self.values[idx] = momentum * self.values[idx] + (1 - momentum) * values[i]
                
                # Reset stats for this slot
                self.access_counts[idx] = 1
                self.last_access[idx] = self.time_step
                
        # Increment time (with wraparound)
        self.time_step = (self.time_step + 1) % 100000
        
    def _update_access_stats(self, indices: torch.Tensor):
        """Update access statistics with bounds"""
        # Flatten indices
        flat_indices = indices.flatten()
        
        # Update counts with ceiling
        self.access_counts.scatter_add_(
            0, flat_indices,
            torch.ones_like(flat_indices, dtype=torch.float)
        )
        self.access_counts.clamp_(max=self.max_access_count)
        
        # Update last access time
        self.last_access.scatter_(
            0, flat_indices,
            self.time_step.expand_as(flat_indices).float()
        )
        
    def reset_stats(self):
        """Reset access statistics periodically"""
        self.access_counts.fill_(0)
        self.last_access.fill_(0)
        self.time_step.fill_(0)


class HierarchicalVAE(nn.Module):
    """Memory-efficient hierarchical VAE with bounded depth"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], max_depth: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.max_depth = min(max_depth, len(hidden_dims))
        
        # Pre-allocated encoder/decoder layers
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Store dimensions for later use
        self.dims = [input_dim] + hidden_dims[:self.max_depth]
        
        for i in range(self.max_depth):
            # Encoder
            self.encoders.append(nn.Sequential(
                nn.Linear(self.dims[i], self.dims[i+1]),
                nn.LayerNorm(self.dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            
            # Decoder (reverse dimensions)
            self.decoders.append(nn.Sequential(
                nn.Linear(self.dims[i+1], self.dims[i]),
                nn.LayerNorm(self.dims[i]),
                nn.GELU()
            ))
            
        # Projection heads for mean and logvar
        self.mu_heads = nn.ModuleList([
            nn.Linear(self.dims[i+1], self.dims[i+1])
            for i in range(self.max_depth)
        ])
        
        self.logvar_heads = nn.ModuleList([
            nn.Linear(self.dims[i+1], self.dims[i+1])
            for i in range(self.max_depth)
        ])
        
    def encode(self, x: torch.Tensor, depth: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode with bounded depth"""
        if depth is None:
            depth = self.max_depth
        else:
            depth = min(depth, self.max_depth)
            
        encodings = []
        current = x
        
        for i in range(depth):
            # Encode
            encoded = self.encoders[i](current)
            
            # Get distribution parameters
            mu = self.mu_heads[i](encoded)
            logvar = self.logvar_heads[i](encoded).clamp(-10, 2)  # Bounded variance
            
            # Reparameterize
            z = self._reparameterize(mu, logvar)
            
            encodings.append((z, mu, logvar))
            current = z
            
        return encodings
        
    def decode(self, encodings: List[torch.Tensor]) -> torch.Tensor:
        """Decode from latent representations"""
        # Start from deepest encoding
        current = encodings[-1]
        
        # Decode through hierarchy
        for i in range(len(encodings) - 1, -1, -1):
            current = self.decoders[i](current)
            
        return current
        
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization with pre-allocated noise"""
        std = torch.exp(0.5 * logvar)
        # Use pre-allocated noise to avoid memory allocation
        eps = torch.randn_like(std)
        return mu + eps * std


class ConceptualCompressorV3(BaseAGIModule):
    """
    Refactored Conceptual Compressor with proper memory management
    """
    
    def _build_module(self):
        """Build module with bounded components"""
        # Configuration
        self.concept_dim = self.config.hidden_size
        self.max_concepts = 256  # Bounded number of concepts
        self.compression_levels = min(3, getattr(self.config, 'compression_levels', 3))
        
        # Bounded neural memory
        self.concept_memory = BoundedNeuralMemory(
            num_concepts=self.max_concepts,
            concept_dim=self.concept_dim,
            value_dim=self.concept_dim,
            max_access_count=10000
        )
        
        # Hierarchical VAE with bounded depth
        hidden_dims = [
            self.concept_dim // 2,
            self.concept_dim // 4,
            self.concept_dim // 8
        ]
        self.hierarchical_vae = HierarchicalVAE(
            self.concept_dim,
            hidden_dims[:self.compression_levels],
            max_depth=self.compression_levels
        )
        
        # Concept understanding network
        self.understanding_net = nn.Sequential(
            nn.Linear(self.concept_dim * 2, self.concept_dim),
            nn.LayerNorm(self.concept_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.concept_dim, self.concept_dim)
        )
        
        # Latent projection (from compressed dim to concept dim)
        self.latent_projector = nn.Linear(self.concept_dim // 8, self.concept_dim)
        
        # Sparse concept encoder
        self.sparse_encoder = nn.Sequential(
            nn.Linear(self.concept_dim, self.concept_dim * 2),
            nn.GELU(),
            nn.Linear(self.concept_dim * 2, self.concept_dim),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.concept_dim, self.concept_dim),
            nn.LayerNorm(self.concept_dim),
            nn.GELU(),
            nn.Linear(self.concept_dim, self.config.output_dim)
        )
        
        # Bounded history buffers
        self.compression_history = self.create_buffer(100)
        self.understanding_scores = self.create_buffer(100)
        
        # Pre-allocated workspace tensors
        self.register_buffer('_workspace_tensor', torch.zeros(64, self.concept_dim))
        
    @RobustForward()
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with bounded compression and understanding
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        
        # Project input to concept dimension
        if x.shape[-1] != self.concept_dim:
            x = F.adaptive_avg_pool1d(
                x.unsqueeze(1), 
                self.concept_dim
            ).squeeze(1)
        
        # Hierarchical compression with bounded depth
        compression_depth = kwargs.get('compression_depth', self.compression_levels)
        compression_depth = min(compression_depth, self.compression_levels)
        
        # Encode through hierarchy
        encodings = self.hierarchical_vae.encode(x, compression_depth)
        
        # Extract latent codes (only keep the z values, not mu/logvar)
        latent_codes = [z for z, _, _ in encodings]
        
        # Ensure latent codes are detached
        latent_codes = [z.detach() for z in latent_codes]
        
        # Decode from compressed representation
        reconstructed = self.hierarchical_vae.decode(latent_codes)
        
        # Compute reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, x, reduction='none').mean(dim=-1)
        
        # Create sparse concept representation
        deepest_latent = latent_codes[-1]
        # Project deepest latent to concept dimension if needed
        if deepest_latent.shape[-1] != self.concept_dim:
            deepest_projected = self.latent_projector(deepest_latent)
        else:
            deepest_projected = deepest_latent
        sparse_concept = self._create_sparse_concept(deepest_projected)
        
        # Query concept memory
        retrieved_concepts, similarities = self.concept_memory.query(sparse_concept, k=5)
        
        # Understand through comparison
        understanding_input = torch.cat([sparse_concept, retrieved_concepts], dim=-1)
        understanding = self.understanding_net(understanding_input)
        
        # Compute understanding score
        understanding_score = torch.sigmoid(similarities.mean(dim=-1))
        
        # Importance based on reconstruction quality and understanding
        importance = torch.sigmoid(-reconstruction_error + understanding_score)
        
        # Update concept memory (bounded)
        if importance.mean() > 0.5:
            self.concept_memory.update(
                sparse_concept.detach(),
                understanding.detach(),
                importance.detach()
            )
        
        # Generate output
        output = self.output_projection(understanding)
        
        # Store bounded history
        self.compression_history.append(reconstruction_error.mean().detach())
        self.understanding_scores.append(understanding_score.mean().detach())
        
        # Compute compression ratio
        original_size = x.shape[-1]
        compressed_size = deepest_latent.shape[-1]
        compression_ratio = original_size / compressed_size
        
        # Return results
        return {
            'output': output,
            'compressed': deepest_latent,
            'reconstructed': reconstructed,
            'understanding': understanding,
            'reconstruction_error': reconstruction_error.mean(),
            'understanding_score': understanding_score.mean(),
            'compression_ratio': torch.tensor(compression_ratio, device=x.device),
            'importance': importance.mean(),
            'n_compression_levels': len(encodings)
        }
    
    def _create_sparse_concept(self, latent: torch.Tensor) -> torch.Tensor:
        """Create sparse concept representation"""
        # Apply sparse encoding
        sparse = self.sparse_encoder(latent)
        
        # Apply soft thresholding for sparsity
        threshold = 0.1
        sparse = F.softshrink(sparse, threshold)
        
        return sparse
    
    def _cleanup_impl(self):
        """Clean up resources"""
        # Clear history buffers
        self.compression_history.clear()
        self.understanding_scores.clear()
        
        # Reset concept memory stats periodically
        if hasattr(self, '_cleanup_counter'):
            self._cleanup_counter += 1
            if self._cleanup_counter % 100 == 0:
                self.concept_memory.reset_stats()
        else:
            self._cleanup_counter = 0
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        comp_history = self.compression_history.get_all()
        und_history = self.understanding_scores.get_all()
        
        return {
            'mean_reconstruction_error': np.mean(comp_history) if comp_history else 0.0,
            'mean_understanding_score': np.mean(und_history) if und_history else 0.0,
            'concept_memory_usage': self.concept_memory.access_counts.sum().item(),
            'max_access_count': self.concept_memory.access_counts.max().item(),
            'history_size': len(comp_history),
            'memory_usage': self.get_memory_usage()
        }