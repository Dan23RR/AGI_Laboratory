#!/usr/bin/env python3
"""
Extended Genome with Dynamic Connection Topology
===============================================

This module implements an advanced genome structure that includes:
- Dynamic connection topology between components
- Self-modifying capabilities
- Intrinsic motivation parameters
"""

import random
import torch
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
import numpy as np

# Setup import path
import sys
import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)

# Add parent directory to Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config_v2 import GENE_HYPERPARAMETER_RANGES
except ImportError:
    try:
        from config import GENE_HYPERPARAMETER_RANGES
    except ImportError:
        # Define default ranges if config not found
        GENE_HYPERPARAMETER_RANGES = {
            'mutation_rate': (0.001, 0.5),
            'crossover_rate': (0.1, 0.9),
            'learning_rate': (0.0001, 0.1),
            'hidden_dim': (64, 1024),
            'num_layers': (1, 10),
            'dropout_rate': (0.0, 0.5)
        }


class ExtendedGenome:
    """
    Extended genome that evolves not just which components to use,
    but also how they connect and communicate.
    """
    
    def __init__(self):
        # Standard genome components
        self.genes: Dict[str, bool] = {}
        self.hyperparameters: Dict[str, Dict[str, Any]] = {}
        
        # Extended components for dynamic topology
        self.connections: Dict[str, Any] = {
            "topology_type": "dynamic_graph",  # dynamic_graph, sequential, parallel, hierarchical
            "edges": [],  # List of (source, target, weight, connection_type) tuples
            "execution_order": [],  # Topologically sorted execution order
            "connection_params": {}  # Parameters for each connection
        }
        
        # Intrinsic motivation configuration
        self.intrinsic_motivation: Dict[str, float] = {
            "empowerment_weight": 0.4,
            "coherence_weight": 0.3,
            "understanding_weight": 0.2,
            "novelty_weight": 0.1,
            "self_modification_threshold": 0.7
        }
        
        # Self-modification parameters
        self.self_modification: Dict[str, Any] = {
            "enabled": True,
            "mutation_rate": 0.1,
            "topology_mutation_rate": 0.05,
            "counterfactual_depth": 3,
            "modification_history": []
        }
        
        # Initialize all available genes
        self._initialize_genes()
        
    def _initialize_genes(self):
        """Initialize genes with activation probabilities"""
        try:
            from config_v2 import INITIAL_GENE_ACTIVATION_PROBABILITY
        except ImportError:
            from config import INITIAL_GENE_ACTIVATION_PROBABILITY
        
        for gene_name in GENE_HYPERPARAMETER_RANGES:
            # Activate gene with probability
            if random.random() < INITIAL_GENE_ACTIVATION_PROBABILITY:
                self.genes[gene_name] = True
                self.hyperparameters[gene_name] = {}
                
                # Initialize hyperparameters
                params = GENE_HYPERPARAMETER_RANGES[gene_name]
                for param_name, param_spec in params.items():
                    self.hyperparameters[gene_name][param_name] = self._sample_parameter(param_spec)
            else:
                self.genes[gene_name] = False
                
        # Initialize connections between active genes
        self._initialize_connections()
        
    def _sample_parameter(self, param_spec: Tuple) -> Any:
        """Sample a parameter value based on its specification"""
        if param_spec[0] == "int":
            return random.randint(param_spec[1], param_spec[2])
        elif param_spec[0] == "float":
            return random.uniform(param_spec[1], param_spec[2])
        elif param_spec[0] == "choice":
            return random.choice(param_spec[1])
        else:
            raise ValueError(f"Unknown parameter type: {param_spec[0]}")
            
    def _initialize_connections(self):
        """Initialize connection topology between active genes"""
        active_genes = [g for g, active in self.genes.items() if active]
        
        if len(active_genes) < 2:
            return
            
        # Create initial topology based on type
        if self.connections["topology_type"] == "dynamic_graph":
            # Create random directed graph with some structure
            n_connections = random.randint(len(active_genes), len(active_genes) * 2)
            
            for _ in range(n_connections):
                source = random.choice(active_genes)
                target = random.choice([g for g in active_genes if g != source])
                weight = random.uniform(0.1, 1.0)
                connection_type = random.choice(["forward", "lateral", "feedback"])
                
                # Avoid duplicate connections
                if not any(e[0] == source and e[1] == target for e in self.connections["edges"]):
                    self.connections["edges"].append((source, target, weight, connection_type))
                    
                    # Add connection-specific parameters
                    conn_id = f"{source}->{target}"
                    self.connections["connection_params"][conn_id] = {
                        "gate_type": random.choice(["linear", "sigmoid", "attention"]),
                        "modulation": random.uniform(0.5, 1.5)
                    }
                    
        # Update execution order
        self._update_execution_order()
        
    def _update_execution_order(self):
        """Compute topologically sorted execution order"""
        active_genes = [g for g, active in self.genes.items() if active]
        
        if not active_genes:
            self.connections["execution_order"] = []
            return
            
        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(active_genes)
        
        for source, target, _, _ in self.connections["edges"]:
            if source in active_genes and target in active_genes:
                G.add_edge(source, target)
                
        # Handle cycles by breaking them
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                # Break cycle at random edge
                idx = random.randint(0, len(cycle) - 1)
                source = cycle[idx]
                target = cycle[(idx + 1) % len(cycle)]
                # Check if edge exists before removing
                if G.has_edge(source, target):
                    G.remove_edge(source, target)
                
        # Compute topological order
        try:
            self.connections["execution_order"] = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Fallback to random order if still has cycles
            self.connections["execution_order"] = active_genes.copy()
            random.shuffle(self.connections["execution_order"])
            
    def mutate(self, mutation_probability: float):
        """Extended mutation including topology changes"""
        # Standard gene mutations
        self._mutate_genes(mutation_probability)
        
        # Mutate hyperparameters
        self._mutate_hyperparameters(mutation_probability)
        
        # Mutate connection topology
        if random.random() < self.self_modification["topology_mutation_rate"]:
            self._mutate_topology()
            
        # Mutate intrinsic motivation weights
        if random.random() < mutation_probability:
            self._mutate_intrinsic_motivation()
            
    def _mutate_genes(self, mutation_probability: float):
        """Mutate gene activation states"""
        for gene_name in self.genes:
            if random.random() < mutation_probability:
                # Flip activation state
                self.genes[gene_name] = not self.genes[gene_name]
                
                if self.genes[gene_name] and gene_name not in self.hyperparameters:
                    # Initialize hyperparameters for newly activated gene
                    self.hyperparameters[gene_name] = {}
                    params = GENE_HYPERPARAMETER_RANGES[gene_name]
                    for param_name, param_spec in params.items():
                        self.hyperparameters[gene_name][param_name] = self._sample_parameter(param_spec)
                elif not self.genes[gene_name] and gene_name in self.hyperparameters:
                    # Remove hyperparameters for deactivated gene
                    del self.hyperparameters[gene_name]
                    
    def _mutate_hyperparameters(self, mutation_probability: float):
        """Mutate hyperparameter values"""
        for gene_name, params in self.hyperparameters.items():
            if self.genes.get(gene_name):
                for param_name in params:
                    if random.random() < mutation_probability:
                        param_spec = GENE_HYPERPARAMETER_RANGES[gene_name][param_name]
                        
                        if param_spec[0] == "int":
                            change = random.randint(-2, 2)
                            new_val = params[param_name] + change
                            params[param_name] = max(param_spec[1], min(param_spec[2], new_val))
                        elif param_spec[0] == "float":
                            change = random.uniform(-0.1 * (param_spec[2] - param_spec[1]), 
                                                   0.1 * (param_spec[2] - param_spec[1]))
                            new_val = params[param_name] + change
                            params[param_name] = max(param_spec[1], min(param_spec[2], new_val))
                        elif param_spec[0] == "choice":
                            params[param_name] = random.choice(param_spec[1])
                            
    def _mutate_topology(self):
        """Mutate connection topology"""
        active_genes = [g for g, active in self.genes.items() if active]
        
        if len(active_genes) < 2:
            return
            
        mutation_type = random.choice(["add_edge", "remove_edge", "change_weight", "change_type"])
        
        if mutation_type == "add_edge" and len(self.connections["edges"]) < len(active_genes) ** 2:
            # Add new connection
            source = random.choice(active_genes)
            target = random.choice([g for g in active_genes if g != source])
            
            if not any(e[0] == source and e[1] == target for e in self.connections["edges"]):
                weight = random.uniform(0.1, 1.0)
                connection_type = random.choice(["forward", "lateral", "feedback"])
                self.connections["edges"].append((source, target, weight, connection_type))
                
                conn_id = f"{source}->{target}"
                self.connections["connection_params"][conn_id] = {
                    "gate_type": random.choice(["linear", "sigmoid", "attention"]),
                    "modulation": random.uniform(0.5, 1.5)
                }
                
        elif mutation_type == "remove_edge" and self.connections["edges"]:
            # Remove random connection
            idx = random.randint(0, len(self.connections["edges"]) - 1)
            edge = self.connections["edges"].pop(idx)
            conn_id = f"{edge[0]}->{edge[1]}"
            if conn_id in self.connections["connection_params"]:
                del self.connections["connection_params"][conn_id]
                
        elif mutation_type == "change_weight" and self.connections["edges"]:
            # Change connection weight
            idx = random.randint(0, len(self.connections["edges"]) - 1)
            source, target, weight, conn_type = self.connections["edges"][idx]
            new_weight = max(0.1, min(1.0, weight + random.uniform(-0.2, 0.2)))
            self.connections["edges"][idx] = (source, target, new_weight, conn_type)
            
        elif mutation_type == "change_type" and self.connections["edges"]:
            # Change connection type
            idx = random.randint(0, len(self.connections["edges"]) - 1)
            source, target, weight, _ = self.connections["edges"][idx]
            new_type = random.choice(["forward", "lateral", "feedback"])
            self.connections["edges"][idx] = (source, target, weight, new_type)
            
        # Update execution order after topology change
        self._update_execution_order()
        
    def _mutate_intrinsic_motivation(self):
        """Mutate intrinsic motivation weights"""
        weights = list(self.intrinsic_motivation.keys())
        if "self_modification_threshold" in weights:
            weights.remove("self_modification_threshold")
            
        # Mutate 1-2 weights
        n_mutations = random.randint(1, 2)
        for _ in range(n_mutations):
            weight_name = random.choice(weights)
            change = random.uniform(-0.1, 0.1)
            new_val = self.intrinsic_motivation[weight_name] + change
            self.intrinsic_motivation[weight_name] = max(0.0, min(1.0, new_val))
            
        # Normalize weights to sum to 1
        total = sum(self.intrinsic_motivation[w] for w in weights)
        if total > 0:
            for w in weights:
                self.intrinsic_motivation[w] /= total
                
    def propose_self_modification(self, current_state: Dict[str, Any]) -> 'ExtendedGenome':
        """Propose a modified version of self based on current state"""
        modified = deepcopy(self)
        
        # Analyze current state to determine modification strategy
        if "performance_history" in current_state:
            recent_performance = np.mean(current_state["performance_history"][-10:])
            
            # If performance is low, increase mutation rate
            if recent_performance < 0.3:
                modified.self_modification["mutation_rate"] *= 1.2
                modified.self_modification["topology_mutation_rate"] *= 1.2
            elif recent_performance > 0.8:
                # If performance is high, reduce mutation to maintain stability
                modified.self_modification["mutation_rate"] *= 0.8
                modified.self_modification["topology_mutation_rate"] *= 0.8
                
        # Apply targeted mutations based on state analysis
        if "module_usage" in current_state:
            # Strengthen connections to frequently used modules
            for edge_idx, (source, target, weight, conn_type) in enumerate(modified.connections["edges"]):
                if source in current_state["module_usage"] and target in current_state["module_usage"]:
                    usage_factor = (current_state["module_usage"][source] + 
                                  current_state["module_usage"][target]) / 2
                    new_weight = min(1.0, weight * (1 + 0.1 * usage_factor))
                    modified.connections["edges"][edge_idx] = (source, target, new_weight, conn_type)
                    
        # Apply mutations
        modified.mutate(modified.self_modification["mutation_rate"])
        
        # Record modification
        modified.self_modification["modification_history"].append({
            "timestamp": current_state.get("timestamp", 0),
            "trigger": "self_proposed",
            "performance": current_state.get("performance", 0)
        })
        
        return modified
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        # Deep copy and convert any sets to lists for JSON compatibility
        def clean_for_json(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)  # Convert tuples to lists for JSON
            return obj
        
        return {
            "genes": clean_for_json(self.genes.copy()),
            "hyperparameters": clean_for_json(deepcopy(self.hyperparameters)),
            "connections": clean_for_json(deepcopy(self.connections)),
            "intrinsic_motivation": clean_for_json(self.intrinsic_motivation.copy()),
            "self_modification": clean_for_json(deepcopy(self.self_modification))
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtendedGenome':
        """Create genome from dictionary"""
        genome = cls()
        genome.genes = data["genes"].copy()
        genome.hyperparameters = deepcopy(data["hyperparameters"])
        genome.connections = deepcopy(data["connections"])
        genome.intrinsic_motivation = data["intrinsic_motivation"].copy()
        genome.self_modification = deepcopy(data["self_modification"])
        return genome
        
    def crossover(self, other: 'ExtendedGenome') -> 'ExtendedGenome':
        """Extended crossover including topology exchange"""
        child = ExtendedGenome()
        
        # Standard gene crossover
        for gene_name in self.genes:
            if random.random() < 0.5:
                child.genes[gene_name] = self.genes[gene_name]
                if gene_name in self.hyperparameters:
                    child.hyperparameters[gene_name] = deepcopy(self.hyperparameters[gene_name])
            else:
                child.genes[gene_name] = other.genes[gene_name]
                if gene_name in other.hyperparameters:
                    child.hyperparameters[gene_name] = deepcopy(other.hyperparameters[gene_name])
                    
        # Crossover intrinsic motivation
        for key in self.intrinsic_motivation:
            if random.random() < 0.5:
                child.intrinsic_motivation[key] = self.intrinsic_motivation[key]
            else:
                child.intrinsic_motivation[key] = other.intrinsic_motivation[key]
                
        # Crossover topology - more complex
        if random.random() < 0.5:
            # Take topology structure from first parent
            child.connections = deepcopy(self.connections)
        else:
            # Take topology structure from second parent
            child.connections = deepcopy(other.connections)
            
        # Filter connections to only include active genes
        active_genes = [g for g, active in child.genes.items() if active]
        child.connections["edges"] = [
            edge for edge in child.connections["edges"]
            if edge[0] in active_genes and edge[1] in active_genes
        ]
        
        # Update execution order
        child._update_execution_order()
        
        return child
    
    def copy(self) -> 'ExtendedGenome':
        """Create a deep copy of this genome"""
        return ExtendedGenome.from_dict(self.to_dict())