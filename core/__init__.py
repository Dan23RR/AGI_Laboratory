"""
Core Infrastructure for AGI System
==================================

Provides memory management, error handling, and secure checkpointing
for all AGI modules.
"""

from .memory_manager import (
    CentralMemoryManager,
    CircularBuffer,
    MemoryManagedModule,
    get_memory_manager,
    MemoryStats,
    MemoryAllocation
)

from .secure_checkpoint import (
    SecureCheckpointManager,
    get_checkpoint_manager,
    SecureCheckpointError
)

from .error_handling import (
    # Exceptions
    AGIError,
    MemoryError,
    DimensionError,
    NumericalError,
    ModuleError,
    ConfigurationError,
    EvolutionError,
    IntegrationError,
    
    # Decorators
    handle_errors,
    RobustForward,
    log_warnings,
    
    # Functions
    validate_tensor,
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_normalize,
    numerical_safety,
    
    # Classes
    ErrorContext,
    ErrorHandler,
    ErrorAggregator,
    error_aggregator
)

from .base_module import (
    BaseAGIModule,
    ModuleConfig,
    AGIModuleFactory,
    ExampleAGIModule
)

# Import refactored modules
try:
    from .conscious_integration_hub_v2 import ConsciousIntegrationHubV2
except ImportError:
    ConsciousIntegrationHubV2 = None

try:
    from .emergent_consciousness_v4 import EmergentConsciousnessV4, EmergentConsciousnessWrapper
except ImportError:
    EmergentConsciousnessV4 = None
    EmergentConsciousnessWrapper = None

try:
    from .goal_conditioned_mcts_v3 import GoalConditionedMCTSV3
except ImportError:
    GoalConditionedMCTSV3 = None

__all__ = [
    # Memory Management
    'CentralMemoryManager',
    'CircularBuffer',
    'MemoryManagedModule',
    'get_memory_manager',
    'MemoryStats',
    'MemoryAllocation',
    
    # Checkpointing
    'SecureCheckpointManager',
    'get_checkpoint_manager',
    'SecureCheckpointError',
    
    # Error Handling
    'AGIError',
    'MemoryError',
    'DimensionError', 
    'NumericalError',
    'ModuleError',
    'ConfigurationError',
    'EvolutionError',
    'IntegrationError',
    'handle_errors',
    'RobustForward',
    'log_warnings',
    'validate_tensor',
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'safe_normalize',
    'numerical_safety',
    'ErrorContext',
    'ErrorHandler',
    'ErrorAggregator',
    'error_aggregator',
    
    # Base Module
    'BaseAGIModule',
    'ModuleConfig',
    'AGIModuleFactory',
    'ExampleAGIModule',
    
    # Refactored Modules
    'ConsciousIntegrationHubV2',
    'EmergentConsciousnessV4',
    'EmergentConsciousnessWrapper',
    'GoalConditionedMCTSV3'
]