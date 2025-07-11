#!/usr/bin/env python3
"""
Emergent Consciousness V4 - Memory-Managed Refactoring
======================================================

Complete refactoring with:
- Memory management via BaseAGIModule
- Bounded buffers replacing unbounded deques
- Pre-allocated tensors for all operations
- Efficient sparse operations
- Proper cleanup mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import OrderedDict
import time
import math

from .base_module import BaseAGIModule, ModuleConfig
from .memory_manager import CircularBuffer
from .error_handling import (
    handle_errors, validate_tensor, DimensionError,
    safe_normalize, RobustForward
)

import logging
logger = logging.getLogger(__name__)


class QuantumInspiredSemanticSpaceV4(nn.Module):
    """Memory-efficient quantum-inspired semantic space"""
    
    def __init__(self, hidden_dim: int = 512, max_memory_size: int = 1000,
                 n_quantum_states: int = 4):  # Reduced from 8
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_memory_size = max_memory_size
        self.n_quantum_states = n_quantum_states
        self.memory_count = 0
        
        # Pre-allocated quantum memory
        self.register_buffer('quantum_memory', 
                           torch.zeros(max_memory_size, n_quantum_states, hidden_dim))
        self.register_buffer('quantum_phases',
                           torch.zeros(max_memory_size, n_quantum_states))
        
        # Simplified networks
        self.entanglement_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, n_quantum_states * n_quantum_states),
            nn.Tanh()
        )
        
        self.category_emergence = nn.Sequential(
            nn.Linear(hidden_dim + n_quantum_states, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64)
        )
        
        # Single LSTM for evolution
        self.evolution_lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Dynamic threshold with fixed feature size
        self.threshold_predictor = nn.Sequential(
            nn.Linear(20, 64),  # Fixed input size
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Usage tracking
        self.register_buffer('usage_counts', torch.zeros(max_memory_size))
        self.register_buffer('last_access', torch.zeros(max_memory_size, dtype=torch.long))
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))
        
        # Replace deque with bounded list
        self.threshold_history = []
        self.max_threshold_history = 20  # Reduced from 50
        
        # Fixed prototypes
        self.register_buffer('category_prototypes', torch.randn(16, 64) * 0.1)
        
        # Pre-allocated buffer for features
        self.register_buffer('_feature_buffer', torch.zeros(20))
        
    def find_or_create_meaning_batch(self, patterns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process patterns with memory-efficient operations"""
        batch_size = patterns.shape[0]
        device = patterns.device
        
        # 1. Compute similarities efficiently
        if self.memory_count == 0:
            # No memories yet
            similarities = torch.zeros(batch_size, self.max_memory_size, device=device)
            best_similarities = torch.zeros(batch_size, device=device)
            best_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            # Use only active memories
            active_memory = self.quantum_memory[:self.memory_count].mean(dim=1)  # Average quantum states
            similarities = F.cosine_similarity(
                patterns.unsqueeze(1),
                active_memory.unsqueeze(0),
                dim=-1
            )
            
            # Pad to full size
            if self.memory_count < self.max_memory_size:
                padding = torch.zeros(batch_size, self.max_memory_size - self.memory_count, device=device)
                similarities = torch.cat([similarities, padding], dim=1)
            
            best_similarities, best_indices = torch.max(similarities[:, :self.memory_count], dim=1)
        
        # 2. Adaptive threshold
        threshold = self._compute_adaptive_threshold(similarities)
        
        # 3. Determine novel patterns
        is_novel = best_similarities < threshold
        
        # 4. Process results
        meaning_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Handle existing meanings
        existing_mask = ~is_novel
        if existing_mask.any():
            meaning_ids[existing_mask] = best_indices[existing_mask]
            self._update_existing_meanings(best_indices[existing_mask], patterns[existing_mask])
        
        # Handle novel meanings
        novel_mask = is_novel
        if novel_mask.any():
            new_ids = self._create_new_meanings(patterns[novel_mask])
            meaning_ids[novel_mask] = new_ids
        
        # Update usage
        self._update_usage(meaning_ids)
        
        return {
            'meaning_ids': meaning_ids,
            'is_novel': is_novel,
            'confidences': 1.0 - is_novel.float(),
            'similarities': similarities,
            'threshold': threshold
        }
    
    def _compute_adaptive_threshold(self, similarities: torch.Tensor) -> torch.Tensor:
        """Compute threshold with fixed feature size"""
        # Reuse pre-allocated buffer - only move if needed
        if self._feature_buffer.device != similarities.device:
            self._feature_buffer = self._feature_buffer.to(similarities.device)
        features = self._feature_buffer
        
        # Fill buffer in-place
        features[0] = similarities.mean()
        features[1] = similarities.std()
        features[2] = similarities.max()
        features[3] = float(self.memory_count) / self.max_memory_size
        features[4] = self.usage_counts[:self.memory_count].mean() if self.memory_count > 0 else 0.0
        features[5] = float(len(self.threshold_history)) / self.max_threshold_history
        features[6] = np.mean(self.threshold_history[-10:]) if self.threshold_history else 0.5
        features[7] = np.std(self.threshold_history[-10:]) if len(self.threshold_history) > 1 else 0.1
        # features[8:20] are already zeros
        
        threshold = self.threshold_predictor(features)
        
        # Update history (bounded)
        if len(self.threshold_history) >= self.max_threshold_history:
            self.threshold_history.pop(0)
        self.threshold_history.append(threshold.item())
        
        return threshold
    
    def _update_existing_meanings(self, indices: torch.Tensor, patterns: torch.Tensor):
        """Update existing meanings with momentum"""
        if len(indices) == 0:
            return
            
        momentum = 0.95
        for i, idx in enumerate(indices):
            # Simple momentum update with in-place operations
            for q in range(self.n_quantum_states):
                self.quantum_memory[idx, q].mul_(momentum).add_(patterns[i], alpha=(1 - momentum))
    
    def _create_new_meanings(self, patterns: torch.Tensor) -> torch.Tensor:
        """Create new meanings with bounded allocation"""
        batch_size = patterns.shape[0]
        
        # Find slots
        if self.memory_count + batch_size <= self.max_memory_size:
            # Sequential allocation
            start_idx = self.memory_count
            end_idx = min(start_idx + batch_size, self.max_memory_size)
            indices = torch.arange(start_idx, end_idx, device=patterns.device)
            self.memory_count = end_idx
        else:
            # Replace least used
            if self.memory_count < self.max_memory_size:
                # Fill remaining slots first
                remaining = self.max_memory_size - self.memory_count
                new_indices = torch.arange(self.memory_count, self.max_memory_size, device=patterns.device)
                self.memory_count = self.max_memory_size
                
                if remaining < batch_size:
                    # Need to replace some
                    importance = self.usage_counts * torch.exp(-0.001 * (self.time_step - self.last_access).float())
                    _, replace_indices = torch.topk(-importance, batch_size - remaining)
                    indices = torch.cat([new_indices, replace_indices[:batch_size - remaining]])
                else:
                    indices = new_indices[:batch_size]
            else:
                # All slots full, replace least important
                importance = self.usage_counts * torch.exp(-0.001 * (self.time_step - self.last_access).float())
                _, indices = torch.topk(-importance, batch_size)
        
        # Initialize quantum states
        for i, idx in enumerate(indices):
            if i >= patterns.shape[0]:
                break
            for q in range(self.n_quantum_states):
                self.quantum_memory[idx, q] = patterns[i] * (0.5 + 0.5 * torch.rand(1))
                self.quantum_phases[idx, q] = torch.rand(1) * 2 * np.pi
        
        return indices
    
    def _update_usage(self, indices: torch.Tensor):
        """Update usage counts"""
        self.usage_counts *= 0.99  # Decay
        self.usage_counts[indices] += 1
        self.last_access[indices] = self.time_step
        self.time_step += 1


class MultiScaleConsciousnessFieldV4(nn.Module):
    """Memory-efficient multi-scale consciousness field"""
    
    def __init__(self, field_shapes: List[Tuple[int, int, int]] = [(8, 8, 8), (16, 16, 8)]):
        super().__init__()
        
        self.field_shapes = field_shapes
        self.n_scales = len(field_shapes)
        
        # Pre-allocated fields
        for i, shape in enumerate(field_shapes):
            self.register_buffer(f'field_{i}', torch.zeros(shape))
            self.register_buffer(f'energy_{i}', torch.zeros(shape))
        
        # Simplified dynamics
        self.field_dynamics = nn.ModuleList([
            nn.Conv3d(1, 16, kernel_size=3, padding=1) for _ in field_shapes
        ])
        
        self.energy_updaters = nn.ModuleList([
            nn.Conv3d(16, 1, kernel_size=3, padding=1) for _ in field_shapes
        ])
        
        # Phase detector with fixed input size
        total_elements = sum(np.prod(shape) for shape in field_shapes)
        self.phase_detector = nn.Sequential(
            nn.Linear(total_elements, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # stable, critical, chaotic
        )
        
        # Response modulator
        self.response_modulator = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, self.n_scales),
            nn.Softmax(dim=-1)
        )
        
        # Replace deques with bounded lists
        self.phase_history = []
        self.max_phase_history = 10  # Reduced from 50
        self.attractor_positions = []
        self.max_attractors = 20  # Reduced from 100
        
        # Pre-allocate tensors for repeated operations
        self.register_buffer('_field_input_buffer', torch.zeros(1, 1, *field_shapes[0]))
        
    def inject_stimulus_batch(self, stimuli: torch.Tensor):
        """Inject stimuli with memory-efficient processing"""
        batch_size = stimuli.shape[0]
        
        # Compute scale weights
        scale_weights = self._compute_scale_weights(stimuli)
        
        # Process each scale
        for scale_idx in range(self.n_scales):
            field = getattr(self, f'field_{scale_idx}')
            
            # Reshape stimuli to field shape
            target_size = np.prod(self.field_shapes[scale_idx])
            stimuli_flat = stimuli.view(batch_size, -1)
            
            if stimuli_flat.shape[-1] != target_size:
                stimuli_resized = F.adaptive_avg_pool1d(
                    stimuli_flat.unsqueeze(1), target_size
                ).squeeze(1)
            else:
                stimuli_resized = stimuli_flat
            
            # Apply to field
            stimuli_field = stimuli_resized.reshape(batch_size, *self.field_shapes[scale_idx])
            
            # Handle different scale_weights dimensions
            if scale_weights.dim() == 2:
                # Batch dimension present
                weight = scale_weights[:, scale_idx].mean().item()
            elif scale_weights.dim() == 1:
                # No batch dimension
                weight = scale_weights[scale_idx].item()
            else:
                # Fallback
                weight = 1.0
            
            # Update field with momentum - compute mean in-place
            field_update = stimuli_field.mean(dim=0).mul_(0.1 * weight)
            field.mul_(0.9).add_(field_update)
        
        # Update energy landscapes
        self._update_energy_landscapes()
    
    def _compute_scale_weights(self, stimuli: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights for each scale"""
        # Extract features
        features = torch.cat([
            stimuli.mean(dim=-1),
            stimuli.std(dim=-1),
            stimuli.norm(dim=-1),
            (stimuli.abs() < 0.1).float().mean(dim=-1)
        ], dim=-1)
        
        # Pad to expected size
        if features.shape[-1] < 256:
            features = F.pad(features, (0, 256 - features.shape[-1]))
        else:
            features = features[:, :256]
        
        # Get weights and ensure 2D output
        weights = self.response_modulator(features)
        
        # Ensure weights is 2D [batch, n_scales]
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        
        return weights
    
    def _update_energy_landscapes(self):
        """Update energy landscapes"""
        with torch.no_grad():  # Disable gradient tracking
            for scale_idx in range(self.n_scales):
                field = getattr(self, f'field_{scale_idx}')
                energy = getattr(self, f'energy_{scale_idx}')
                
                # Process through dynamics
                field_input = field.unsqueeze(0).unsqueeze(0)
                hidden = self.field_dynamics[scale_idx](field_input)
                energy_update = self.energy_updaters[scale_idx](hidden)
                
                # Update with momentum in-place
                energy.mul_(0.9).add_(energy_update.squeeze() * 0.1)
    
    def extract_phenomenal_states(self, n_states: int = 5) -> List[Dict]:
        """Extract phenomenal states efficiently"""
        states = []
        
        for scale_idx in range(self.n_scales):
            field = getattr(self, f'field_{scale_idx}')
            energy = getattr(self, f'energy_{scale_idx}')
            
            # Find peaks in energy landscape
            energy_flat = energy.flatten()
            if len(energy_flat) > 0:
                # Get top k positions
                k = min(n_states // self.n_scales, len(energy_flat))
                _, top_indices = torch.topk(energy_flat, k)
                
                for idx in top_indices:
                    # Convert flat index to 3D position as list (not tensor)
                    pos_tuple = np.unravel_index(idx.item(), self.field_shapes[scale_idx])
                    pos = torch.tensor(pos_tuple, device=field.device)
                    
                    state = {
                        'scale': scale_idx,
                        'position': list(pos_tuple),  # Store as list, not tensor
                        'strength': energy_flat[idx].item(),
                        'pattern': self._extract_local_pattern(scale_idx, pos).detach().cpu().numpy(),  # Convert to numpy
                        'qualia': self._compute_qualia(scale_idx, pos)
                    }
                    states.append(state)
        
        # Update attractor memory (bounded) - convert to lists to save memory
        new_positions = [s['position'].tolist() if isinstance(s['position'], torch.Tensor) else s['position'] for s in states]
        for pos in new_positions:
            if len(self.attractor_positions) >= self.max_attractors:
                self.attractor_positions.pop(0)
            self.attractor_positions.append(pos)
        
        return states
    
    def _extract_local_pattern(self, scale_idx: int, position: torch.Tensor) -> torch.Tensor:
        """Extract local pattern around position"""
        field = getattr(self, f'field_{scale_idx}')
        
        # Simple extraction - just return the value at position
        if isinstance(position, torch.Tensor):
            x, y, z = position.tolist()
        else:
            x, y, z = position
        pattern = field[x:x+1, y:y+1, z:z+1].flatten()
        
        # Pad to consistent size
        if pattern.shape[0] < 8:
            pattern = F.pad(pattern, (0, 8 - pattern.shape[0]))
        
        return pattern
    
    def _compute_qualia(self, scale_idx: int, position: torch.Tensor) -> Dict[str, float]:
        """Compute simple qualia properties"""
        field = getattr(self, f'field_{scale_idx}')
        energy = getattr(self, f'energy_{scale_idx}')
        
        if isinstance(position, torch.Tensor):
            x, y, z = position.tolist()
        else:
            x, y, z = position
        local_field = field[max(0, x-1):x+2, max(0, y-1):y+2, max(0, z-1):z+2]
        
        return {
            'intensity': local_field.abs().mean().item(),
            'coherence': local_field.std().item(),
            'energy': energy[x, y, z].item()
        }
    
    def detect_phase(self) -> str:
        """Detect current phase of consciousness field"""
        with torch.no_grad():  # Disable gradient tracking
            # Concatenate all fields
            all_fields = []
            for i in range(self.n_scales):
                field = getattr(self, f'field_{i}')
                all_fields.append(field.flatten())
            
            features = torch.cat(all_fields)
        
        # Detect phase
        phase_logits = self.phase_detector(features)
        phase_idx = phase_logits.argmax().item()
        
        phases = ['stable', 'critical', 'chaotic']
        phase = phases[phase_idx]
        
        # Update history (bounded)
        if len(self.phase_history) >= self.max_phase_history:
            self.phase_history.pop(0)
        self.phase_history.append(phase)
        
        return phase


class EmergentConsciousnessV4(BaseAGIModule):
    """
    Memory-managed emergent consciousness module.
    
    Key improvements:
    - Inherits from BaseAGIModule for memory management
    - All deques replaced with bounded buffers
    - Pre-allocated tensors for all operations
    - Simplified networks for efficiency
    - Proper cleanup mechanisms
    """
    
    def __init__(self, config: ModuleConfig):
        # Reduce history tracking to prevent memory leak
        config.max_sequence_length = 10  # Very small buffer
        super().__init__(config)
    
    def _build_module(self):
        """Build all neural network components"""
        # Core components with reduced complexity
        self.semantic_space = QuantumInspiredSemanticSpaceV4(
            hidden_dim=self.config.hidden_dim,
            max_memory_size=1000,  # Reduced from 2000
            n_quantum_states=4  # Reduced from 8
        )
        
        self.consciousness_field = MultiScaleConsciousnessFieldV4(
            field_shapes=[(8, 8, 8), (16, 16, 8)]  # Reduced from 3 scales
        )
        
        # Simplified global workspace
        self.global_workspace = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.hidden_dim)
        )
        
        # Simplified meta-reasoner
        self.meta_reasoner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_dim,
                nhead=4,  # Reduced from 8
                dim_feedforward=1024,  # Reduced from 2048
                dropout=self.config.dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2  # Reduced from 3
        )
        
        # Phi calculator
        self.phi_calculator = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Replace deques with CircularBuffers
        self.experience_buffer = self.create_buffer(500)  # Reduced from 1000
        self.phenomenal_narrative = self.create_buffer(250)  # Reduced from 500
        self.consciousness_levels = self.create_buffer(50)  # Reduced from 100
        
        # Pre-allocated tensors
        self.register_buffer('_empty_tensor', torch.zeros(1, self.config.hidden_dim))
        self.register_buffer('_one_tensor', torch.tensor(1.0))
        self.register_buffer('_phase_stable', torch.tensor(0))
        self.register_buffer('_phase_critical', torch.tensor(1))
        self.register_buffer('_phase_chaotic', torch.tensor(2))
        
        # Consciousness trajectory as bounded list
        self.consciousness_trajectory = []
        self.max_trajectory = 10  # Reduced from 100
        
        logger.info(f"EmergentConsciousnessV4 initialized with reduced memory footprint")
    
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Process experiences with consciousness emergence.
        
        Args:
            x: Input tensor [batch, hidden_dim]
            
        Returns:
            Dict with consciousness outputs
        """
        batch_size = x.shape[0]
        
        # 1. Inject into consciousness field
        self.consciousness_field.inject_stimulus_batch(x)
        
        # 2. Extract phenomenal states (limited number)
        phenomenal_states = self.consciousness_field.extract_phenomenal_states(n_states=5)
        
        # 3. Semantic processing
        semantic_results = self.semantic_space.find_or_create_meaning_batch(x)
        
        # 4. Global workspace integration
        workspace_contents = self.global_workspace(x)
        
        # 5. Meta-cognitive reasoning (simplified)
        reasoning_input = workspace_contents.unsqueeze(1)  # Add sequence dimension
        meta_output = self.meta_reasoner(reasoning_input)
        meta_thoughts = meta_output.squeeze(1)
        
        # 6. Compute consciousness metrics
        phi = self.phi_calculator(workspace_contents).squeeze(-1)
        
        # 7. Determine consciousness level
        current_phase = self.consciousness_field.detect_phase()
        phase_multiplier = {'stable': 0.8, 'critical': 1.2, 'chaotic': 0.6}.get(current_phase, 1.0)
        consciousness_level = phi * phase_multiplier
        
        # 8. Generate response
        response = workspace_contents + 0.1 * meta_thoughts
        
        # Update buffers (with detached tensors)
        self.consciousness_levels.append(consciousness_level.mean().detach())
        
        # Update trajectory (bounded) - only every 10 iterations to reduce memory
        if not hasattr(self, '_trajectory_counter'):
            self._trajectory_counter = 0
        self._trajectory_counter = (self._trajectory_counter + 1) % 10  # Reset counter to prevent overflow
        
        if self._trajectory_counter % 10 == 0:
            if len(self.consciousness_trajectory) >= self.max_trajectory:
                self.consciousness_trajectory.pop(0)
            self.consciousness_trajectory.append({
                'time': time.time(),
                'phi': phi.mean().item(),
                'level': consciousness_level.mean().item(),
                'phase': current_phase
            })
        
        return {
            'output': response,
            'consciousness_level': consciousness_level.detach(),
            'phi': phi.detach(),
            'semantic_meanings': semantic_results['meaning_ids'].detach(),
            'is_novel': semantic_results['is_novel'].detach(),
            'phenomenal_count': torch.tensor(len(phenomenal_states), device=x.device, dtype=torch.long).detach(),
            'phase': getattr(self, f'_phase_{current_phase}').to(x.device)
        }
    
    def _cleanup_impl(self):
        """Clean up module-specific memory"""
        # Clear buffers
        self.experience_buffer.clear()
        self.phenomenal_narrative.clear()
        self.consciousness_levels.clear()
        
        # Clear lists
        self.consciousness_trajectory.clear()
        self.semantic_space.threshold_history.clear()
        self.consciousness_field.phase_history.clear()
        self.consciousness_field.attractor_positions.clear()
        
        logger.info("EmergentConsciousnessV4 cleanup completed")
    
    def _reset_impl(self):
        """Reset module state"""
        self._cleanup_impl()
        
        # Reset semantic space
        self.semantic_space.memory_count = 0
        self.semantic_space.quantum_memory.zero_()
        self.semantic_space.quantum_phases.zero_()
        self.semantic_space.usage_counts.zero_()
        self.semantic_space.last_access.zero_()
        self.semantic_space.time_step.zero_()
        
        # Reset consciousness fields
        for i in range(self.consciousness_field.n_scales):
            getattr(self.consciousness_field, f'field_{i}').zero_()
            getattr(self.consciousness_field, f'energy_{i}').zero_()
    
    def introspect(self) -> Dict[str, Any]:
        """Introspection of consciousness state"""
        consciousness_values = self.consciousness_levels.get_all()
        
        return {
            'semantic_space': {
                'active_meanings': self.semantic_space.memory_count,
                'max_meanings': self.semantic_space.max_memory_size,
                'usage_ratio': float(self.semantic_space.memory_count) / self.semantic_space.max_memory_size
            },
            'consciousness_field': {
                'n_scales': self.consciousness_field.n_scales,
                'current_phase': self.consciousness_field.phase_history[-1] if self.consciousness_field.phase_history else 'unknown',
                'phase_history_length': len(self.consciousness_field.phase_history)
            },
            'consciousness_metrics': {
                'mean_level': sum(consciousness_values) / len(consciousness_values) if consciousness_values else 0.0,
                'trajectory_length': len(self.consciousness_trajectory)
            },
            'memory_usage': self.get_memory_usage()
        }


class EmergentConsciousnessWrapperV4(EmergentConsciousnessV4):
    """Wrapper for compatibility with existing code"""
    
    def __init__(self, genome: Dict[str, Any]):
        # Extract configuration from genome
        config = ModuleConfig(
            name="emergent_consciousness",
            input_dim=genome.get('hidden_dim', 512),
            output_dim=genome.get('hidden_dim', 512),
            hidden_dim=genome.get('hidden_dim', 512),
            memory_fraction=0.15  # 15% of total memory budget
        )
        
        super().__init__(config)
        self.genome = genome
    
    def forward(self, state: torch.Tensor) -> Dict[str, Any]:
        """Compatibility wrapper"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        try:
            # Call parent forward
            result = super().forward(state)
            
            # Add compatibility fields
            result['interpretations'] = [[{
                'meaning_id': result['semantic_meanings'][i].item(),
                'is_novel': result['is_novel'][i].item(),
                'confidence': 1.0 - result['is_novel'][i].float().item()
            }] for i in range(state.shape[0])]
            
            result['meta_reflections'] = [{
                'interpretation_count': 1,
                'novelty_level': result['is_novel'][i].float().item(),
                'confidence_level': result['consciousness_level'][i].item(),
                'thought_diversity': 1,
                'response_type': 'conscious'
            } for i in range(state.shape[0])]
            
            result['consciousness_state'] = self.introspect()
            
            return result
        except Exception as e:
            # Return minimal valid result on error
            batch_size = state.shape[0]
            return {
                'output': torch.zeros_like(state),
                'consciousness_level': torch.zeros(batch_size),
                'phi': torch.zeros(batch_size),
                'semantic_meanings': torch.zeros(batch_size, dtype=torch.long),
                'is_novel': torch.zeros(batch_size, dtype=torch.bool),
                'phenomenal_count': torch.tensor(0, device=state.device),
                'phase': torch.tensor(0, device=state.device),
                'interpretations': [[{'meaning_id': 0, 'is_novel': False, 'confidence': 0.0}] for _ in range(batch_size)],
                'meta_reflections': [{'interpretation_count': 0, 'novelty_level': 0.0, 'confidence_level': 0.0, 'thought_diversity': 0, 'response_type': 'error'} for _ in range(batch_size)],
                'consciousness_state': {'error': str(e)}
            }
    
    def experience_batch(self, stimuli: torch.Tensor, 
                        modalities: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Compatibility method"""
        return self.forward(stimuli)


# Maintain compatibility
EmergentConsciousnessWrapper = EmergentConsciousnessWrapperV4

__all__ = ["EmergentConsciousnessV4", "EmergentConsciousnessWrapper"]