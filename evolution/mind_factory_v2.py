#!/usr/bin/env python3
"""
Mind Factory V2 - Clean Integration with ConsciousIntegrationHubV2
=================================================================

This factory creates minds using ConsciousIntegrationHubV2 as the central
orchestrator, with all modules communicating through it.

Key principles:
1. ConsciousIntegrationHubV2 is the mind (single nn.Module)
2. All modules are created from ModuleConfig
3. No complex graphs or dynamic topologies
4. Compatible with GeneralEvolutionLabV3
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import intel_extension_for_pytorch as ipex

from core.base_module import ModuleConfig, AGIModuleFactory, BaseAGIModule
from modules.conscious_integration_hub_v2 import ConsciousIntegrationHubV2
from core.memory_manager import CentralMemoryManager

# Import refactored V3 modules
from modules.feedback_loop_system_v3 import FeedbackLoopSystemV3
from modules.sentient_agi_v3 import SentientAGIV3
from modules.dynamic_conceptual_field_v3 import DynamicConceptualFieldV3
from modules.recursive_self_model_v3 import RecursiveSelfModelV3
from modules.counterfactual_reasoner_v3 import CounterfactualReasonerV3
from modules.conceptual_compressor_v3 import ConceptualCompressorV3
from modules.attractor_networks_v3 import HierarchicalAttractorNetworkV3
from modules.emergence_enhancer_v3 import EmergenceEnhancerV3
from modules.global_integration_field_v3 import GlobalIntegrationFieldV3
from modules.internal_goal_generation_v3 import InternalGoalGenerationV3
from modules.coherence_stabilizer_v3 import CoherenceStabilizerV3
from modules.empowerment_calculator_v3 import EmpowermentCalculatorV3, EmpowermentConfigV3
from modules.energy_based_world_model_v2 import EnergyBasedWorldModelV2
from modules.emergent_consciousness_v4 import EmergentConsciousnessV4

# Setup import path for world_model
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from world_model import PredictiveWorldModel
except ImportError:
    PredictiveWorldModel = None  # Optional module


@dataclass
class MindConfig:
    """Configuration for a complete mind"""
    hidden_dim: int = 512
    n_modules: int = 8
    num_attention_heads: int = 8
    n_meta_states: int = 16
    output_dim: int = 64
    device: str = "cpu"
    memory_fraction: float = 0.5  # Fraction of total memory for all modules


# WorldModel removed - using PredictiveWorldModel directly


class MindFactoryV2:
    """
    Factory for creating minds with ConsciousIntegrationHubV2 as orchestrator.
    """
    
    # Module registry - maps gene names to module classes
    MODULE_REGISTRY = {
        "FeedbackLoopSystemV3": FeedbackLoopSystemV3,
        "SentientAGIV3": SentientAGIV3,
        "DynamicConceptualFieldV3": DynamicConceptualFieldV3,
        "RecursiveSelfModelV3": RecursiveSelfModelV3,
        "CounterfactualReasonerV3": CounterfactualReasonerV3,
        "ConceptualCompressorV3": ConceptualCompressorV3,
        "HierarchicalAttractorNetworkV3": HierarchicalAttractorNetworkV3,
        "EmergenceEnhancerV3": EmergenceEnhancerV3,
        "GlobalIntegrationFieldV3": GlobalIntegrationFieldV3,
        "InternalGoalGeneratorV3": InternalGoalGenerationV3,
        "CoherenceStabilizerV3": CoherenceStabilizerV3,
        "EmpowermentCalculatorV3": EmpowermentCalculatorV3,
        "EnergyBasedWorldModelV2": EnergyBasedWorldModelV2,
        "EmergentConsciousnessV4": EmergentConsciousnessV4
    }
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_manager = CentralMemoryManager()
    
    def create_mind_from_genome(self, genome: Dict[str, Any], 
                               mind_config: Optional[MindConfig] = None) -> nn.Module:
        """
        Create a mind (ConsciousIntegrationHubV2) from genome specification.
        
        Args:
            genome: Dictionary with 'genes' (active modules) and 'hyperparameters'
            mind_config: Configuration for the mind
            
        Returns:
            ConsciousIntegrationHubV2 instance with registered modules
        """
        if mind_config is None:
            mind_config = MindConfig()
        
        # Create the hub with proper ModuleConfig
        hub_module_config = ModuleConfig(
            name="ConsciousIntegrationHub",
            input_dim=mind_config.hidden_dim,
            output_dim=mind_config.output_dim,
            hidden_dim=mind_config.hidden_dim,
            memory_fraction=0.2  # Hub gets 20% of memory
        )
        
        # Note: ConsciousIntegrationHubV2 uses the ModuleConfig parameters directly
        # It sets max_modules=32 internally and uses config.hidden_dim for all operations
        
        hub = ConsciousIntegrationHubV2(hub_module_config)
        hub = hub.to(self.device)

        # Create a dummy optimizer for IPEX optimization. In a real scenario,
        # you would integrate this with your actual optimizer creation logic.
        optimizer = torch.optim.Adam(hub.parameters(), lr=0.001)

        print("Applying Intel Extension for PyTorch (IPEX) CPU optimization...")
        hub, optimizer = ipex.optimize(hub, optimizer=optimizer)
        print("Optimization applied.")
        
        # Create and register modules based on genome
        active_genes = genome.get('genes', {})
        hyperparameters = genome.get('hyperparameters', {})
        
        # Calculate memory fraction per module
        n_active = sum(1 for active in active_genes.values() if active)
        if n_active > 0:
            memory_per_module = mind_config.memory_fraction / n_active
        else:
            memory_per_module = 0.1
        
        # Create modules
        for gene_name, is_active in active_genes.items():
            if is_active and gene_name in self.MODULE_REGISTRY:
                try:
                    # Create module config
                    module_config = self._create_module_config(
                        gene_name, 
                        hyperparameters.get(gene_name, {}),
                        memory_per_module,
                        mind_config
                    )
                    
                    # Create module instance
                    module = self._create_module(gene_name, module_config)
                    
                    if module is not None:
                        module = module.to(self.device)
                        # Register with hub
                        hub.register_module(gene_name, module)
                        
                except Exception as e:
                    print(f"Warning: Failed to create module {gene_name}: {e}")
        
        return hub
    
    def _create_module_config(self, gene_name: str, hyperparams: Dict[str, Any],
                            memory_fraction: float, mind_config: MindConfig) -> ModuleConfig:
        """Create ModuleConfig from gene hyperparameters"""
        
        # Base configuration
        config = ModuleConfig(
            name=gene_name,
            input_dim=hyperparams.get('input_dim', mind_config.hidden_dim),
            output_dim=hyperparams.get('output_dim', mind_config.hidden_dim),
            hidden_dim=hyperparams.get('hidden_dim', mind_config.hidden_dim),
            memory_fraction=memory_fraction
        )
        
        return config
    
    def _create_module(self, gene_name: str, config: ModuleConfig) -> Optional[nn.Module]:
        """Create module instance with special handling for modules with dependencies"""
        
        module_class = self.MODULE_REGISTRY.get(gene_name)
        if module_class is None:
            return None
        
        # Special case: EmpowermentCalculator needs world model
        if gene_name == "EmpowermentCalculatorV3":
            emp_config = EmpowermentConfigV3(
                state_dim=64,
                action_dim=4,
                horizon=5,
                n_action_samples=16,
                optimization_steps=3
            )
            # Create a world model instance dynamically
            wm_config = ModuleConfig(
                name="EnergyBasedWorldModelV2",
                input_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                memory_fraction=0.05
            )
            world_model = EnergyBasedWorldModelV2(wm_config).to(self.device)
            return module_class(config, emp_config, world_model)
        
        # Standard module creation
        try:
            return module_class(config)
        except Exception as e:
            print(f"Error creating {gene_name}: {e}")
            # Try using AGIModuleFactory as fallback
            try:
                return AGIModuleFactory.create_module(gene_name, config)
            except:
                return None
    
    def create_mind_from_config(self, module_configs: List[ModuleConfig],
                               mind_config: Optional[MindConfig] = None) -> nn.Module:
        """
        Create a mind from a list of ModuleConfig objects.
        
        This is the preferred method for GeneralEvolutionLabV3 integration.
        """
        if mind_config is None:
            mind_config = MindConfig()
        
        # Create the hub with proper ModuleConfig
        hub_module_config = ModuleConfig(
            name="ConsciousIntegrationHub",
            input_dim=mind_config.hidden_dim,
            output_dim=mind_config.output_dim,
            hidden_dim=mind_config.hidden_dim,
            memory_fraction=0.2
        )
        
        hub = ConsciousIntegrationHubV2(hub_module_config)
        hub = hub.to(self.device)
        
        # Create and register modules
        for config in module_configs:
            try:
                module = self._create_module(config.name, config)
                if module is not None:
                    module = module.to(self.device)
                    hub.register_module(config.name, module)
            except Exception as e:
                print(f"Warning: Failed to create module {config.name}: {e}")
        
        return hub
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'memory_manager'):
            # Memory manager cleanup if needed
            pass


# Convenience function for quick mind creation
def create_default_mind(active_modules: Optional[List[str]] = None) -> nn.Module:
    """
    Create a mind with default configuration and specified active modules.
    
    Args:
        active_modules: List of module names to activate. If None, uses a default set.
        
    Returns:
        ConsciousIntegrationHubV2 instance
    """
    if active_modules is None:
        active_modules = [
            "FeedbackLoopSystem",
            "RecursiveSelfModel",
            "CounterfactualReasoner",
            "CoherenceStabilizer",
            "InternalGoalGeneration"
        ]
    
    # Create genome-like structure
    genome = {
        'genes': {name: (name in active_modules) 
                 for name in MindFactoryV2.MODULE_REGISTRY.keys()},
        'hyperparameters': {}
    }
    
    factory = MindFactoryV2()
    return factory.create_mind_from_genome(genome)


__all__ = ['MindFactoryV2', 'MindConfig', 'create_default_mind']