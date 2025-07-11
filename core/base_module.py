#!/usr/bin/env python3
"""
Base Module for AGI System
==========================

Integrates memory management, error handling, and secure checkpointing
into a base class that all AGI modules must inherit from.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

from .memory_manager import MemoryManagedModule, CircularBuffer, get_memory_manager
from .error_handling import (
    handle_errors, validate_tensor, RobustForward, 
    safe_divide, safe_log, safe_normalize,
    ModuleError, DimensionError, NumericalError
)
from .secure_checkpoint import get_checkpoint_manager

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Configuration for AGI modules"""
    name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 256
    memory_fraction: float = 0.05  # 5% of total memory budget
    use_cuda: bool = True
    dtype: torch.dtype = torch.float32
    dropout: float = 0.1
    max_sequence_length: int = 1000
    
    @property
    def hidden_size(self) -> int:
        """Alias for hidden_dim for backward compatibility"""
        return self.hidden_dim
    
    @property
    def memory_limit(self) -> int:
        """Memory limit based on sequence length for backward compatibility"""
        return self.max_sequence_length
    
    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads - default to 8"""
        return 8


class BaseAGIModule(nn.Module, MemoryManagedModule, ABC):
    """
    Base class for all AGI modules with integrated safety features.
    
    Features:
    - Automatic memory management with hard limits
    - Robust error handling and recovery
    - Secure checkpointing without pickle
    - Standardized interface for integration
    """
    
    def __init__(self, config: ModuleConfig):
        nn.Module.__init__(self)
        MemoryManagedModule.__init__(self, config.name, config.memory_fraction)
        
        self.config = config
        self.device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Pre-allocate common projections to avoid dynamic allocation
        self._projection_cache = {}
        self._init_projections()
        
        # Circular buffers for history
        self.state_history = self.create_buffer(config.max_sequence_length)
        self.output_history = self.create_buffer(config.max_sequence_length)
        
        # Error tracking
        self.error_count = 0
        self.last_error = None
        
        # Initialize module-specific components
        self._build_module()
        
        # Move to device
        self.to(self.device)
        
    def _init_projections(self):
        """Pre-allocate common dimension projections"""
        common_dims = [64, 128, 256, 512, 1024]
        
        for in_dim in common_dims:
            if in_dim == self.config.hidden_dim:
                continue
            key = f"proj_{in_dim}_to_{self.config.hidden_dim}"
            self._projection_cache[key] = nn.Linear(in_dim, self.config.hidden_dim).to(self.device)
            
    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to correct dimension with caching"""
        if x.shape[-1] == self.config.hidden_dim:
            return x
            
        key = f"proj_{x.shape[-1]}_to_{self.config.hidden_dim}"
        
        if key not in self._projection_cache:
            # Allocate new projection with memory tracking
            proj = self.allocate_module(
                nn.Linear(x.shape[-1], self.config.hidden_dim)
            )
            if proj is None:
                raise MemoryError(f"Cannot allocate projection for {key}")
            self._projection_cache[key] = proj.to(self.device)
            
        return self._projection_cache[key](x)
        
    def allocate_module(self, module: nn.Module) -> Optional[nn.Module]:
        """Allocate a module with memory tracking"""
        # Estimate module memory
        param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in module.buffers())
        total_memory = param_memory + buffer_memory
        
        if self.memory_manager.allocate(self.module_name, total_memory):
            return module
        else:
            logger.error(f"Failed to allocate {total_memory/1024**2:.2f}MB for module")
            return None
            
    @abstractmethod
    def _build_module(self):
        """Build module-specific components - must be implemented by subclasses"""
        pass
        
    @abstractmethod
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Module-specific forward implementation"""
        pass
        
    @RobustForward(max_retries=3, cleanup_on_error=True)
    @handle_errors(error_types=(RuntimeError, ValueError), propagate=False)
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Robust forward pass with automatic error handling.
        
        Args:
            x: Input tensor or dict of tensors
            **kwargs: Additional arguments
            
        Returns:
            Dict with at least 'output' key
        """
        # Handle dict input
        if isinstance(x, dict):
            x = x.get('output', x.get('hidden_state', x.get('x', None)))
            if x is None:
                raise ValueError("Dict input must contain 'output', 'hidden_state', or 'x' key")
                
        # Validate and prepare input
        x = validate_tensor(x, "input", check_nan=True, check_inf=True)
        
        # Ensure correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Project if needed
        if x.shape[-1] != self.config.hidden_dim:
            x = self.project_input(x)
            
        # Save to history
        self.state_history.append(x)
        
        # Call module implementation
        result = self._forward_impl(x, **kwargs)
        
        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {'output': result}
            
        # Validate output
        for key, tensor in result.items():
            if isinstance(tensor, torch.Tensor):
                result[key] = validate_tensor(tensor, f"output[{key}]")
                
        # Save output to history
        if 'output' in result:
            self.output_history.append(result['output'])
            
        return result
        
    def cleanup(self) -> None:
        """Clean up module memory"""
        # Clear histories
        self.state_history.clear()
        self.output_history.clear()
        
        # Clear any module-specific memory
        self._cleanup_impl()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"{self.module_name}: Memory cleaned up")
        
    def _cleanup_impl(self):
        """Module-specific cleanup - override if needed"""
        pass
        
    def get_state(self) -> Dict[str, Any]:
        """Get module state for checkpointing"""
        return {
            'state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'error_count': self.error_count,
            'device': str(self.device)
        }
        
    def load_state(self, state: Dict[str, Any]):
        """Load module state from checkpoint"""
        if 'state_dict' in state:
            self.load_state_dict(state['state_dict'])
        if 'config' in state:
            # Validate config compatibility
            for key, value in state['config'].items():
                if hasattr(self.config, key) and getattr(self.config, key) != value:
                    logger.warning(f"Config mismatch for {key}: {getattr(self.config, key)} != {value}")
                    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in self.buffers()) / 1024**2
        
        history_memory = (len(self.state_history) + len(self.output_history)) * \
                        self.config.hidden_dim * 4 / 1024**2  # Approximate
                        
        return {
            'parameters_mb': param_memory,
            'buffers_mb': buffer_memory,
            'history_mb': history_memory,
            'total_mb': param_memory + buffer_memory + history_memory
        }
        
    def reset(self):
        """Reset module state"""
        # Clear histories
        self.state_history.clear()
        self.output_history.clear()
        
        # Reset any RNN states
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                
        # Reset error tracking
        self.error_count = 0
        self.last_error = None
        
        # Module-specific reset
        self._reset_impl()
        
    def _reset_impl(self):
        """Module-specific reset - override if needed"""
        pass


class AGIModuleFactory:
    """Factory for creating AGI modules with proper configuration"""
    
    _module_registry = {}
    
    @classmethod
    def register(cls, name: str, module_class: type):
        """Register a module class"""
        if not issubclass(module_class, BaseAGIModule):
            raise ValueError(f"{module_class} must inherit from BaseAGIModule")
        cls._module_registry[name] = module_class
        
    @classmethod
    def create(cls, name: str, config: ModuleConfig) -> BaseAGIModule:
        """Create a module instance"""
        if name not in cls._module_registry:
            raise ValueError(f"Module {name} not registered")
            
        module_class = cls._module_registry[name]
        return module_class(config)
        
    @classmethod
    def list_modules(cls) -> List[str]:
        """List all registered modules"""
        return list(cls._module_registry.keys())


# Example of how to create a concrete module
class ExampleAGIModule(BaseAGIModule):
    """Example implementation of an AGI module"""
    
    def _build_module(self):
        """Build the neural network components"""
        # Pre-allocate all layers
        self.encoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim)
        )
        
        self.processor = nn.LSTM(
            self.config.hidden_dim,
            self.config.hidden_dim,
            batch_first=True,
            dropout=self.config.dropout
        )
        
        self.decoder = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        
        # Pre-allocate LSTM states
        self.hidden_state = None
        self.cell_state = None
        
    def _forward_impl(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Example forward implementation"""
        batch_size = x.shape[0]
        
        # Encode
        encoded = self.encoder(x)
        
        # Add sequence dimension if needed
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)
            
        # Process with LSTM
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.hidden_state = torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device)
            self.cell_state = torch.zeros(1, batch_size, self.config.hidden_dim, device=self.device)
            
        output, (self.hidden_state, self.cell_state) = self.processor(
            encoded, (self.hidden_state, self.cell_state)
        )
        
        # Decode
        if output.dim() == 3:
            output = output[:, -1, :]  # Take last timestep
            
        decoded = self.decoder(output)
        
        return {
            'output': decoded,
            'hidden_state': self.hidden_state.squeeze(0),
            'encoded': encoded.squeeze(1) if encoded.shape[1] == 1 else encoded
        }
        
    def _cleanup_impl(self):
        """Clean up LSTM states"""
        self.hidden_state = None
        self.cell_state = None
        
    def _reset_impl(self):
        """Reset LSTM states"""
        self.hidden_state = None
        self.cell_state = None


# Register the example module
AGIModuleFactory.register('example', ExampleAGIModule)