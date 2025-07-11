#!/usr/bin/env python3
"""
Sentient AGI V3 - Refactored with Memory Management
===================================================

Major improvements:
- Inherits from BaseAGIModule for proper memory management
- No unbounded collections (deque → CircularBuffer)
- Pre-allocated structures
- Simplified architecture (removed redundant components)
- Proper cleanup implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import math

from core.base_module import BaseAGIModule, CircularBuffer, ModuleConfig
from core.error_handling import RobustForward, handle_errors
from .conscious_integration_hub_v2 import ConsciousIntegrationHubV2
from .emergent_consciousness_v4 import EmergentConsciousnessV4
from .goal_conditioned_mcts_v3 import GoalConditionedMCTSV3


class SimplifiedRecursiveSelfModel(nn.Module):
    """Simplified recursive self-model with bounded depth"""
    
    def __init__(self, base_dim: int = 256, max_depth: int = 3):
        super().__init__()
        self.base_dim = base_dim
        self.max_depth = max_depth
        
        # Single shared encoder (no dynamic creation)
        self.encoder = nn.LSTM(
            input_size=base_dim,
            hidden_size=base_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Fixed projections for each depth level
        self.depth_projections = nn.ModuleList([
            nn.Linear(base_dim, base_dim) for _ in range(max_depth)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process with bounded recursion"""
        batch_size = x.shape[0]
        
        # Process through LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, (h_n, _) = self.encoder(x)
        
        # Combine bidirectional outputs
        h_forward = h_n[0]  # Forward hidden state
        h_backward = h_n[1]  # Backward hidden state
        combined = torch.cat([h_forward, h_backward], dim=-1)
        
        # Apply depth projections
        depth_outputs = []
        current = combined
        for i in range(self.max_depth):
            current = self.depth_projections[i](current)
            current = F.gelu(current)
            depth_outputs.append(current)
        
        # Average across depths
        unified_self = torch.stack(depth_outputs).mean(dim=0)
        unified_self = self.output_projection(unified_self)
        
        return {
            'unified_self': unified_self,
            'recursion_depth': self.max_depth,
            'self_coherence': torch.ones(batch_size, device=x.device)  # Simplified
        }


class BoundedGlobalField(nn.Module):
    """Simplified global field with bounded memory"""
    
    def __init__(self, field_dim: int = 128, max_size: int = 1000):
        super().__init__()
        self.field_dim = field_dim
        self.max_size = max_size
        
        # Fixed-size field representation
        self.register_buffer('field_state', torch.zeros(1, field_dim))
        self.register_buffer('field_energy', torch.ones(1))
        
        # Simple field dynamics
        self.field_dynamics = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim)
        )
        
        # Integration estimator
        self.integration_estimator = nn.Sequential(
            nn.Linear(field_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, input_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update field with input signal"""
        batch_size = input_signal.shape[0]
        
        # Ensure field state matches batch size
        if self.field_state.shape[0] != batch_size:
            self.field_state = self.field_state.expand(batch_size, -1).contiguous()
            self.field_energy = self.field_energy.expand(batch_size).contiguous()
        
        # Project input to field dimension
        if input_signal.shape[-1] != self.field_dim:
            input_proj = F.adaptive_avg_pool1d(
                input_signal.unsqueeze(1), self.field_dim
            ).squeeze(1)
        else:
            input_proj = input_signal
        
        # Update field dynamics
        combined = torch.cat([self.field_state, input_proj], dim=-1)
        field_update = self.field_dynamics(combined)
        
        # Update field state with momentum
        self.field_state = 0.9 * self.field_state + 0.1 * field_update
        self.field_state = torch.tanh(self.field_state)  # Keep bounded
        
        # Estimate integration
        phi = self.integration_estimator(self.field_state).squeeze(-1)
        self.field_energy = 0.95 * self.field_energy + 0.05 * phi
        
        return {
            'field_state': self.field_state,
            'integrated_information': phi,
            'field_energy': self.field_energy
        }


class SentientAGIV3(BaseAGIModule):
    """
    Refactored Sentient AGI with proper memory management
    """
    
    def _build_module(self):
        """Build module with pre-allocated components"""
        # Core dimensions
        self.hidden_dim = self.config.hidden_size
        self.conscious_dim = self.hidden_dim // 2
        
        # Core modules (use refactored versions)
        self.conscious_hub = ConsciousIntegrationHubV2(self.config)
        self.emergent_consciousness = EmergentConsciousnessV4(self.config)
        
        # Create MCTS with proper config
        from .goal_conditioned_mcts_v3 import MCTSConfigV3
        from core.base_module import ModuleConfig
        mcts_config = MCTSConfigV3(
            action_dim=self.hidden_dim,
            state_dim=self.hidden_dim,
            max_tree_size=100,
            n_simulations=10,
            cache_size=100
        )
        # MCTS needs special config
        mcts_module_config = ModuleConfig(
            name="sentient_mcts",
            input_dim=self.hidden_dim * 2,  # state + goal
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,  # Important: must match input_dim
            memory_fraction=0.05
        )
        # Simple world model
        def simple_world_model(state, action):
            return state + action * 0.1
        self.goal_mcts = GoalConditionedMCTSV3(mcts_module_config, mcts_config, simple_world_model)
        
        # Simplified self-model
        self.self_model = SimplifiedRecursiveSelfModel(
            base_dim=self.hidden_dim,
            max_depth=3
        )
        
        # Bounded global field
        self.global_field = BoundedGlobalField(
            field_dim=self.conscious_dim,
            max_size=1000
        )
        
        # Subjective time modulation (simplified)
        self.time_modulator = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Unified output network
        self.output_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Value and action heads
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.action_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Bounded experience buffer
        self.experience_buffer = self.create_buffer(500)
        self.consciousness_buffer = self.create_buffer(100)
        
    @RobustForward()
    def _forward_impl(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Main forward pass with memory-bounded operations
        """
        # Handle input format
        if isinstance(inputs, torch.Tensor):
            world_state = inputs
            if world_state.dim() == 1:
                world_state = world_state.unsqueeze(0)
            batch_size = world_state.shape[0]
        else:
            world_state = inputs.get('world_state', inputs.get('observation', None))
            if world_state is None:
                raise ValueError("Input must contain 'world_state' or 'observation'")
            batch_size = world_state.shape[0]
        
        # Ensure correct dimension
        if world_state.shape[-1] != self.hidden_dim:
            world_state = F.adaptive_avg_pool1d(
                world_state.unsqueeze(1), self.hidden_dim
            ).squeeze(1)
        
        # 1. Process through consciousness modules
        consciousness_output = self.emergent_consciousness(world_state)
        conscious_state = consciousness_output['output']
        
        # 2. Self-modeling
        self_output = self.self_model(world_state)
        unified_self = self_output['unified_self']
        
        # 3. Goal-conditioned planning (simplified interface)
        goal = torch.randn(batch_size, self.hidden_dim, device=world_state.device) * 0.1
        # MCTS expects concatenated [state, goal] as single tensor
        mcts_input = torch.cat([world_state, goal], dim=-1)
        mcts_output = self.goal_mcts(mcts_input)
        if mcts_output is not None:
            planned_action = mcts_output.get('output', torch.zeros_like(world_state))
        else:
            planned_action = torch.zeros_like(world_state)
        
        # 4. Global field integration
        field_input = conscious_state.mean(dim=0) if conscious_state.dim() > 1 else conscious_state
        field_output = self.global_field(field_input.unsqueeze(0) if field_input.dim() == 1 else field_input)
        integrated_info = field_output['integrated_information']
        
        # 5. Hub integration
        module_outputs = {
            'consciousness': conscious_state,
            'self_model': unified_self,
            'planning': planned_action
        }
        
        # Hub expects 'x' key with main input, pass world_state as main input
        hub_output = self.conscious_hub({
            'x': world_state,
            'module_outputs': module_outputs
        })
        integrated_state = hub_output['output']
        
        # 6. Combine all information
        combined = torch.cat([
            integrated_state,
            unified_self,
            conscious_state
        ], dim=-1)
        
        # 7. Generate output
        sentient_output = self.output_network(combined)
        
        # 8. Compute value and action
        value = self.value_head(sentient_output)
        action = self.action_head(sentient_output)
        action = torch.tanh(action)  # Bounded actions
        
        # 9. Subjective time
        time_dilation = self.time_modulator(sentient_output).squeeze(-1)
        
        # 10. Store in buffers (bounded)
        self.experience_buffer.append(world_state.detach())
        self.consciousness_buffer.append(conscious_state.detach())
        
        # 11. Estimate consciousness level
        consciousness_level = self._estimate_consciousness_level(
            integrated_info,
            self_output['self_coherence'],
            hub_output.get('global_coherence', torch.ones(batch_size))
        )
        
        return {
            'output': sentient_output,
            'action': action,
            'value': value,
            'consciousness_level': consciousness_level,
            'subjective_time_dilation': time_dilation,
            'integrated_information': integrated_info,
            'self_coherence': self_output['self_coherence'],
            'module_outputs': module_outputs
        }
    
    def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Main forward method"""
        return self._forward_impl(inputs)
    
    def _estimate_consciousness_level(self, phi: torch.Tensor, 
                                    self_coherence: torch.Tensor,
                                    global_coherence: torch.Tensor) -> torch.Tensor:
        """Estimate overall consciousness level"""
        # Simple weighted average
        consciousness = (
            0.4 * torch.sigmoid(phi) +
            0.3 * self_coherence +
            0.3 * global_coherence
        )
        return consciousness
    
    def _cleanup_impl(self):
        """Clean up resources"""
        self.experience_buffer.clear()
        self.consciousness_buffer.clear()
        
        # Cleanup submodules
        if hasattr(self.conscious_hub, 'cleanup'):
            self.conscious_hub.cleanup()
        if hasattr(self.emergent_consciousness, 'cleanup'):
            self.emergent_consciousness.cleanup()
        if hasattr(self.goal_mcts, 'cleanup'):
            self.goal_mcts.cleanup()
    
    def get_introspection(self) -> Dict[str, Any]:
        """Get introspection data"""
        return {
            'experience_count': len(self.experience_buffer),
            'consciousness_states': len(self.consciousness_buffer),
            'field_energy': self.global_field.field_energy.mean().item(),
            'modules_active': 3  # Simplified from original
        }