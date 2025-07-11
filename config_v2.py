"""
Configuration File V2 for AGI Laboratory
=======================================

Enhanced configuration with all emergent hyperparameters from V2/V3 modules
for true general intelligence evolution.
"""

# --- General Project Settings ---
PROJECT_NAME = "AGI Laboratory - General Intelligence Evolution"
VERSION = "2.0.0"

# --- Evolutionary Engine Parameters ---
POPULATION_SIZE = 100  # Larger for more diversity
N_GENERATIONS = 1000  # Extended evolution
TOURNAMENT_SIZE = 7
MUTATION_PROBABILITY = 0.15
CROSSOVER_PROBABILITY = 0.7
ELITE_SIZE = 10  # Preserve best genomes

# --- Initial Genome Settings ---
INITIAL_GENE_ACTIVATION_PROBABILITY = 0.6  # Higher to encourage complexity
MIN_ACTIVE_MODULES = 3  # Minimum modules for viability
MAX_ACTIVE_MODULES = 12  # Maximum to prevent bloat

# --- AGI Fitness Weights (Multi-Domain) ---
FITNESS_WEIGHTS = {
    # Core AGI capabilities
    "generalization": 0.15,
    "emergence": 0.15,
    "adaptability": 0.12,
    "creativity": 0.10,
    "reasoning": 0.10,
    "consciousness": 0.10,
    "efficiency": 0.08,
    "robustness": 0.08,
    
    # Meta capabilities
    "meta_learning": 0.05,
    "abstraction": 0.04,
    "coherence": 0.03,
    
    # Penalties
    "complexity_penalty": -0.02,
    "instability_penalty": -0.03
}

# --- V2/V3 Gene Hyperparameter Search Space ---
GENE_HYPERPARAMETER_RANGES = {
    # V2 Feedback Loop System
    "FeedbackLoopSystemV2": {
        "hidden_size": ("choice", [256, 512, 1024]),
        "memory_limit": ("int", 500, 5000),
        "num_attention_heads": ("choice", [4, 8, 16]),
        "dropout_rate": ("float", 0.0, 0.3),
        "memory_decay_rate": ("float", 0.9, 0.99),
        "attention_temperature": ("float", 0.5, 2.0),
        "gating_threshold": ("float", 0.3, 0.7)
    },
    
    # V2 Sentient AGI
    "SentientAGIV2": {
        "state_dim": ("choice", [256, 512, 1024]),
        "action_dim": ("choice", [128, 256, 512]),
        "n_transformer_layers": ("int", 2, 8),
        "n_heads": ("choice", [4, 8, 16]),
        "recursion_depth": ("int", 1, 5),
        "self_model_update_freq": ("int", 10, 100),
        "consciousness_threshold": ("float", 0.3, 0.8),
        "integration_beta": ("float", 0.1, 1.0)
    },
    
    # V2 Dynamic Conceptual Field
    "DynamicConceptualFieldV2": {
        "field_size": ("choice", [(32,32,16), (64,64,32), (128,128,64)]),
        "semantic_dims": ("choice", [64, 128, 256]),
        "sparsity_ratio": ("float", 0.1, 0.5),
        "evolution_rate": ("float", 0.01, 0.1),
        "perception_radius": ("float", 2.0, 10.0),
        "thought_threshold": ("float", 0.2, 0.8),
        "n_thought_streams": ("int", 3, 10)
    },
    
    # V2 Emergence Enhancer
    "EmergenceEnhancerV2": {
        "n_components": ("int", 4, 16),
        "chaos_injection_strength": ("float", 0.05, 0.3),
        "coupling_strength": ("float", 0.1, 0.5),
        "resonance_blend": ("float", 0.1, 0.5),
        "optimal_correlation": ("float", 0.2, 0.6),
        "lorenz_dt": ("float", 0.005, 0.05),
        "n_chaos_generators": ("int", 2, 6),
        "n_resonators": ("int", 2, 8),
        "sparsity_ratio": ("float", 0.2, 0.5)
    },
    
    # V2 Empowerment Calculator
    "EmpowermentCalculatorV2": {
        "horizon": ("int", 3, 15),
        "n_action_samples": ("int", 16, 64),
        "optimization_steps": ("int", 5, 20),
        "mi_estimator_layers": ("int", 2, 5),
        "empowerment_weight": ("float", 0.5, 1.5),
        "curiosity_weight": ("float", 0.1, 0.5),
        "tool_threshold": ("float", 0.1, 0.5),
        "cache_size": ("int", 500, 2000),
        "gradient_checkpointing": ("choice", [True, False]),
        "batch_rollouts": ("choice", [True, False]),
        "max_parallel_rollouts": ("int", 32, 128)
    },
    
    # V2 Goal-Conditioned MCTS
    "GoalConditionedMCTSV2": {
        "discount": ("float", 0.9, 0.99),
        "c_puct": ("float", 0.5, 2.0),
        "n_simulations": ("int", 20, 200),
        "horizon": ("int", 5, 20),
        "progressive_widening_alpha": ("float", 0.3, 0.7),
        "progressive_widening_k": ("float", 1.0, 3.0),
        "cache_size": ("int", 5000, 20000),
        "value_network_hidden": ("choice", [128, 256, 512]),
        "use_value_approximation": ("choice", [True, False]),
        "parallel_simulations": ("int", 2, 8),
        "adaptive_budget": ("choice", [True, False]),
        "tree_reuse": ("choice", [True, False]),
        "continuous_action_opt_steps": ("int", 3, 10),
        "cem_elite_ratio": ("float", 0.1, 0.3),
        "cem_population_size": ("int", 32, 128)
    },
    
    # V3 Internal Goal Generator
    "InternalGoalGeneratorV3": {
        "max_goals": ("int", 5, 20),
        "goal_embedding_dim": ("choice", [64, 128, 256]),
        "temporal_horizons": ("choice", [[10,50,200], [5,25,100], [20,100,500]]),
        "n_goal_critics": ("int", 2, 6),
        "diversity_k": ("int", 3, 8),
        "achievement_threshold": ("float", 0.6, 0.9),
        "curiosity_beta": ("float", 0.1, 1.0),
        "feasibility_alpha": ("float", 0.1, 0.5),
        "cross_modal_dim": ("choice", [32, 64, 128])
    },
    
    # V3 Emergent Consciousness
    "EmergentConsciousnessV3": {
        "n_quantum_states": ("int", 4, 16),
        "n_consciousness_fields": ("int", 2, 6),
        "field_resolution": ("choice", [(8,8,4), (16,16,8), (32,32,16)]),
        "semantic_memory_size": ("int", 1000, 5000),
        "working_memory_size": ("int", 20, 100),
        "binding_threshold": ("float", 0.3, 0.7),
        "phi_computation_steps": ("int", 5, 20),
        "attention_heads": ("choice", [4, 8, 16]),
        "meta_state_dim": ("choice", [64, 128, 256]),
        "phase_coupling_strength": ("float", 0.1, 0.5),
        "consciousness_temperature": ("float", 0.5, 2.0)
    },
    
    # Advanced Integration Hub
    "ConsciousIntegrationHub": {
        "hidden_dim": ("choice", [256, 512, 1024]),
        "num_attention_heads": ("choice", [4, 8, 16]),
        "n_meta_states": ("int", 8, 32),
        "causal_dropout": ("float", 0.0, 0.3),
        "dependency_learning_rate": ("float", 0.001, 0.1),
        "coherence_target": ("float", 0.7, 0.95),
        "intent_encoding_dim": ("choice", [64, 128, 256]),
        "coordination_layers": ("int", 2, 5)
    },
    
    # Original modules (kept for compatibility)
    "RecursiveSelfModel": {
        "base_dim": ("choice", [128, 256, 512]),
        "recursion_depth": ("int", 2, 6),
        "hidden_ratio": ("float", 0.5, 0.9),
        "dropout_rate": ("float", 0.0, 0.3),
        "self_attention": ("choice", [True, False])
    },
    
    "CounterfactualReasoner": {
        "n_alternatives": ("int", 5, 20),
        "simulation_depth": ("int", 2, 8),
        "confidence_threshold": ("float", 0.5, 0.9),
        "n_critics": ("int", 2, 5),
        "ensemble_size": ("int", 3, 10)
    },
    
    "ConceptualCompressor": {
        "latent_dim": ("choice", [32, 64, 128]),
        "n_layers": ("int", 3, 8),
        "compression_ratio": ("float", 0.1, 0.5),
        "beta": ("float", 0.5, 10.0),
        "use_vae": ("choice", [True, False])
    },
    
    "ImprovedGlobalField": {
        "field_resolution": ("choice", [(16,16,8), (32,32,16), (64,64,32)]),
        "concept_dim": ("choice", [64, 128, 256]),
        "n_attractors": ("int", 5, 20),
        "field_coupling": ("float", 0.1, 0.5),
        "evolution_rate": ("float", 0.01, 0.1),
        "sparsity_penalty": ("float", 0.01, 0.1),
        "memory_horizon": ("int", 50, 200),
        "field_temperature": ("float", 0.5, 2.0)
    }
}

# --- Intrinsic Motivation Configuration ---
INTRINSIC_MOTIVATION_RANGES = {
    "empowerment_weight": ("float", 0.1, 1.0),
    "coherence_weight": ("float", 0.1, 1.0),
    "understanding_weight": ("float", 0.1, 1.0),
    "novelty_weight": ("float", 0.1, 1.0),
    "self_modification_threshold": ("float", 0.5, 0.9)
}

# --- Connection Topology Parameters ---
CONNECTION_TOPOLOGY_RANGES = {
    "topology_type": ("choice", ["dynamic_graph", "sequential", "parallel", "hierarchical", "small_world", "scale_free"]),
    "connection_density": ("float", 0.2, 0.8),
    "connection_weight_range": ("float", 0.1, 1.0),
    "rewiring_probability": ("float", 0.0, 0.2),
    "hub_preference": ("float", 0.0, 1.0),
    "modularity_strength": ("float", 0.0, 1.0)
}

# --- Environment Configuration ---
ENVIRONMENT_CONFIG = {
    "suite": "GeneralEnvironmentSuite",
    "environments": [
        {
            "name": "abstract_problem",
            "weight": 0.33,
            "difficulty_range": (0.2, 0.8),
            "max_steps": 100
        },
        {
            "name": "concept_learning", 
            "weight": 0.33,
            "difficulty_range": (0.2, 0.8),
            "max_steps": 200
        },
        {
            "name": "multi_agent",
            "weight": 0.34,
            "difficulty_range": (0.2, 0.8), 
            "max_steps": 150
        }
    ],
    "meta_environment": {
        "enabled": True,
        "switch_frequency": 50,
        "difficulty_progression": True
    }
}

# --- AGI Evaluation Settings ---
EVALUATION_CONFIG = {
    "n_test_episodes": 20,
    "test_difficulties": [0.3, 0.5, 0.7, 0.9],
    "cross_domain_transfer": True,
    "zero_shot_tests": True,
    "adaptation_tests": True,
    "creativity_tests": True,
    "min_performance_threshold": 0.3  # Minimum to be considered viable
}

# --- Self-Modification Parameters ---
SELF_MODIFICATION_CONFIG = {
    "enabled": True,
    "base_mutation_rate": 0.1,
    "topology_mutation_rate": 0.05,
    "counterfactual_depth": 3,
    "modification_cooldown": 10,
    "stability_threshold": 0.8,
    "improvement_threshold": 0.05
}

# --- Resource Management ---
RESOURCE_CONFIG = {
    "max_memory_gb": 32,
    "max_computation_time": 300,  # seconds per evaluation
    "early_stopping_patience": 50,
    "checkpoint_frequency": 10
}

# --- Logging and Monitoring ---
LOGGING_CONFIG = {
    "log_level": "INFO",
    "tensorboard": True,
    "save_best_n_genomes": 10,
    "save_frequency": 5,
    "track_module_interactions": True,
    "track_emergence_metrics": True
}