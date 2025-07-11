#!/usr/bin/env python3
"""
Robust Error Handling System for AGI
====================================

Replaces generic exception handling with specific error types,
detailed logging, and proper error propagation.
"""

import logging
import traceback
import sys
import torch
import numpy as np
from typing import Optional, Any, Dict, Callable, Union, Type
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
import warnings
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Custom Exception Hierarchy
class AGIError(Exception):
    """Base exception for all AGI-related errors"""
    pass


class MemoryError(AGIError):
    """Memory allocation or limit errors"""
    pass


class DimensionError(AGIError):
    """Tensor dimension mismatch errors"""
    pass


class NumericalError(AGIError):
    """Numerical instability (NaN, Inf) errors"""
    pass


class ModuleError(AGIError):
    """Module initialization or execution errors"""
    pass


class ConfigurationError(AGIError):
    """Configuration or parameter errors"""
    pass


class EvolutionError(AGIError):
    """Evolution process errors"""
    pass


class IntegrationError(AGIError):
    """Module integration errors"""
    pass


@dataclass
class ErrorContext:
    """Context information for errors"""
    module_name: str
    operation: str
    iteration: Optional[int] = None
    generation: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class ErrorHandler:
    """Central error handling system"""
    
    def __init__(self, log_file: Optional[str] = "agi_errors.log"):
        self.error_history = []
        self.error_counts = {}
        
        # Setup file logging if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    def log_error(self, error: Exception, context: ErrorContext) -> None:
        """Log error with full context"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # Update error counts
        error_key = f"{context.module_name}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log to logger
        logger.error(
            f"Error in {context.module_name}.{context.operation}: "
            f"{type(error).__name__}: {error}\n"
            f"Context: {context.additional_info}\n"
            f"{'='*60}\n"
            f"{traceback.format_exc()}"
        )
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:]
        }


# Global error handler
_error_handler = ErrorHandler()


def handle_errors(error_types: Union[Type[Exception], tuple] = Exception,
                 default_return: Any = None,
                 propagate: bool = False):
    """
    Decorator for handling errors in AGI modules.
    
    Args:
        error_types: Types of errors to catch
        default_return: Value to return on error
        propagate: Whether to re-raise the error after logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            context = ErrorContext(
                module_name=self.__class__.__name__,
                operation=func.__name__,
                iteration=getattr(self, 'iteration', None),
                generation=getattr(self, 'generation', None)
            )
            
            try:
                return func(self, *args, **kwargs)
            except error_types as e:
                _error_handler.log_error(e, context)
                
                if propagate:
                    raise
                    
                return default_return
                
        return wrapper
    return decorator


def validate_tensor(tensor: torch.Tensor, 
                   name: str = "tensor",
                   check_nan: bool = True,
                   check_inf: bool = True,
                   check_shape: Optional[tuple] = None,
                   check_range: Optional[tuple] = None) -> torch.Tensor:
    """
    Validate tensor properties and raise specific errors.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        check_shape: Expected shape (None to skip)
        check_range: Expected (min, max) range
        
    Returns:
        The tensor if valid
        
    Raises:
        NumericalError: If NaN or Inf found
        DimensionError: If shape mismatch
    """
    if check_nan and torch.isnan(tensor).any():
        raise NumericalError(f"{name} contains NaN values")
        
    if check_inf and torch.isinf(tensor).any():
        raise NumericalError(f"{name} contains Inf values")
        
    if check_shape is not None and tensor.shape != check_shape:
        raise DimensionError(
            f"{name} shape mismatch: expected {check_shape}, got {tensor.shape}"
        )
        
    if check_range is not None:
        min_val, max_val = check_range
        if tensor.min() < min_val or tensor.max() > max_val:
            raise NumericalError(
                f"{name} values out of range [{min_val}, {max_val}]: "
                f"got [{tensor.min():.3f}, {tensor.max():.3f}]"
            )
            
    return tensor


@contextmanager
def numerical_safety(clip_value: float = 10.0,
                    epsilon: float = 1e-8,
                    handle_nan: str = "zero"):
    """
    Context manager for numerical safety.
    
    Args:
        clip_value: Maximum absolute value for clipping
        epsilon: Small value to prevent division by zero
        handle_nan: How to handle NaN ("zero", "mean", "raise")
    """
    # Store original error state
    old_error_state = torch.get_default_dtype()
    
    # Set error handling
    torch.set_grad_enabled(True)
    
    def make_safe(x: torch.Tensor) -> torch.Tensor:
        """Make tensor numerically safe"""
        if torch.isnan(x).any():
            if handle_nan == "zero":
                x = torch.nan_to_num(x, nan=0.0)
            elif handle_nan == "mean":
                x = torch.nan_to_num(x, nan=x[~torch.isnan(x)].mean())
            else:
                raise NumericalError("NaN detected in computation")
                
        # Clip to prevent explosion
        x = torch.clamp(x, -clip_value, clip_value)
        
        return x
        
    try:
        # Monkey patch torch operations temporarily
        _original_add = torch.add
        _original_mul = torch.mul
        _original_div = torch.div
        
        torch.add = lambda a, b: make_safe(_original_add(a, b))
        torch.mul = lambda a, b: make_safe(_original_mul(a, b))
        torch.div = lambda a, b: make_safe(_original_div(a, b + epsilon))
        
        yield
        
    finally:
        # Restore original operations
        torch.add = _original_add
        torch.mul = _original_mul
        torch.div = _original_div


def safe_divide(numerator: torch.Tensor, 
                denominator: torch.Tensor,
                epsilon: float = 1e-8) -> torch.Tensor:
    """Safe division preventing div by zero"""
    return numerator / (denominator + epsilon)


def safe_log(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Safe logarithm preventing log(0)"""
    return torch.log(torch.clamp(x, min=epsilon))


def safe_sqrt(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Safe square root preventing sqrt of negative"""
    return torch.sqrt(torch.clamp(x, min=epsilon))


def safe_normalize(x: torch.Tensor, dim: int = -1, 
                  epsilon: float = 1e-8) -> torch.Tensor:
    """Safe normalization preventing division by zero"""
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / (norm + epsilon)


class RobustForward:
    """Decorator for robust forward passes with automatic error recovery"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 cleanup_on_error: bool = True,
                 fallback_mode: bool = True):
        self.max_retries = max_retries
        self.cleanup_on_error = cleanup_on_error
        self.fallback_mode = fallback_mode
        
    def __call__(self, forward_method):
        @wraps(forward_method)
        def wrapper(module_self, *args, **kwargs):
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    # Validate inputs
                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor):
                            validate_tensor(arg, f"input_{i}")
                            
                    # Run forward pass
                    with numerical_safety():
                        result = forward_method(module_self, *args, **kwargs)
                        
                    # Validate output
                    if isinstance(result, torch.Tensor):
                        validate_tensor(result, "output")
                    elif isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, torch.Tensor):
                                validate_tensor(v, f"output[{k}]")
                                
                    return result
                    
                except (NumericalError, RuntimeError) as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed "
                        f"in {module_self.__class__.__name__}: {e}"
                    )
                    
                    if self.cleanup_on_error and hasattr(module_self, 'cleanup'):
                        module_self.cleanup()
                        
                    # Try fallback mode on last attempt
                    if attempt == self.max_retries - 1 and self.fallback_mode:
                        logger.warning(f"Entering fallback mode for {module_self.__class__.__name__}")
                        return self._fallback_forward(module_self, args, kwargs)
                        
            # All retries failed
            raise ModuleError(
                f"Forward pass failed after {self.max_retries} attempts: {last_error}"
            )
            
        return wrapper
        
    def _fallback_forward(self, module, args, kwargs):
        """Simple fallback that returns zeros of expected shape"""
        # Try to infer output shape from input
        if args and isinstance(args[0], torch.Tensor):
            batch_size = args[0].shape[0]
            device = args[0].device
            dtype = args[0].dtype
            
            # Return zero tensor with reasonable shape
            if hasattr(module, 'output_dim'):
                return torch.zeros(batch_size, module.output_dim, device=device, dtype=dtype)
            else:
                return torch.zeros_like(args[0])
        else:
            raise ModuleError("Cannot determine fallback output shape")


def log_warnings(func):
    """Decorator to capture and log warnings"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)
            
            if w:
                for warning in w:
                    logger.warning(
                        f"Warning in {func.__name__}: {warning.category.__name__}: "
                        f"{warning.message}"
                    )
                    
        return result
    return wrapper


class ErrorAggregator:
    """Aggregate errors across modules for summary reporting"""
    
    def __init__(self):
        self.errors_by_module = defaultdict(list)
        self.errors_by_type = defaultdict(list)
        
    def add_error(self, module_name: str, error: Exception, context: Dict[str, Any] = None):
        """Add an error to the aggregator"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'type': type(error).__name__,
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.errors_by_module[module_name].append(error_info)
        self.errors_by_type[type(error).__name__].append(error_info)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            'total_errors': sum(len(errors) for errors in self.errors_by_module.values()),
            'modules_with_errors': list(self.errors_by_module.keys()),
            'error_types': {
                error_type: len(errors) 
                for error_type, errors in self.errors_by_type.items()
            },
            'most_problematic_module': max(
                self.errors_by_module.items(),
                key=lambda x: len(x[1])
            )[0] if self.errors_by_module else None
        }
        
    def clear(self):
        """Clear all errors"""
        self.errors_by_module.clear()
        self.errors_by_type.clear()


# Global error aggregator
error_aggregator = ErrorAggregator()