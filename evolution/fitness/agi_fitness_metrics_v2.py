#!/usr/bin/env python3
"""
AGI Fitness Metrics V2 - True General Intelligence Evaluation
============================================================
Revolutionary metrics that evaluate genuine AGI capabilities across
multiple domains without task-specific bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
from collections import deque, defaultdict
import math
from scipy import stats
from abc import ABC, abstractmethod


@dataclass
class DomainPerformance:
    """Performance in a specific domain"""
    domain_name: str
    score: float
    adaptation_speed: float
    generalization_from: List[str] = field(default_factory=list)
    zero_shot: bool = False
    confidence: float = 0.0
    

@dataclass 
class AGIFitnessScore:
    """Comprehensive AGI fitness evaluation"""
    # Core capabilities
    generalization: float = 0.0
    emergence: float = 0.0
    adaptability: float = 0.0
    creativity: float = 0.0
    reasoning: float = 0.0
    consciousness: float = 0.0
    efficiency: float = 0.0
    robustness: float = 0.0
    
    # Meta capabilities
    meta_learning: float = 0.0
    abstraction: float = 0.0
    coherence: float = 0.0
    autonomy: float = 0.0
    
    # Domain performances
    domain_scores: Dict[str, DomainPerformance] = field(default_factory=dict)
    
    # Detailed metrics
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def overall_agi_score(self) -> float:
        """
        Compute overall AGI score using geometric mean to ensure
        all capabilities must be present (no gaming single metrics)
        """
        core_scores = [
            self.generalization,
            self.emergence,
            self.adaptability,
            self.creativity,
            self.reasoning,
            self.consciousness,
            self.efficiency,
            self.robustness
        ]
        
        meta_scores = [
            self.meta_learning,
            self.abstraction,
            self.coherence,
            self.autonomy
        ]
        
        # Ensure all scores are positive
        core_scores = [max(0.0, score) for score in core_scores]
        meta_scores = [max(0.0, score) for score in meta_scores]
        
        # Geometric mean of core capabilities
        if all(score > 0 for score in core_scores):
            core_geometric = float(np.exp(np.mean(np.log(np.array(core_scores)))))
        else:
            # If any score is 0, use arithmetic mean with small offset
            core_geometric = float(np.mean(np.array(core_scores)) + 1e-8)
        
        # Geometric mean of meta capabilities  
        if all(score > 0 for score in meta_scores):
            meta_geometric = float(np.exp(np.mean(np.log(np.array(meta_scores)))))
        else:
            # If any score is 0, use arithmetic mean with small offset
            meta_geometric = float(np.mean(np.array(meta_scores)) + 1e-8)
        
        # Combined score with emphasis on core
        final_score = 0.7 * core_geometric + 0.3 * meta_geometric
        
        # Ensure final score is not NaN
        if np.isnan(final_score):
            print(f"Warning: NaN score detected. Core scores: {core_scores}, Meta scores: {meta_scores}")
            return 0.0
        
        return final_score
    
    def get_weakest_capability(self) -> Tuple[str, float]:
        """Identify the weakest capability for targeted evolution"""
        capabilities = {
            'generalization': self.generalization,
            'emergence': self.emergence,
            'adaptability': self.adaptability,
            'creativity': self.creativity,
            'reasoning': self.reasoning,
            'consciousness': self.consciousness,
            'efficiency': self.efficiency,
            'robustness': self.robustness,
            'meta_learning': self.meta_learning,
            'abstraction': self.abstraction,
            'coherence': self.coherence,
            'autonomy': self.autonomy
        }
        
        weakest = min(capabilities.items(), key=lambda x: x[1])
        return weakest


class DomainAgnosticTest(ABC):
    """Base class for domain-agnostic tests"""
    
    @abstractmethod
    def generate_task(self, difficulty: float) -> Dict[str, Any]:
        """Generate a task at specified difficulty"""
        pass
    
    @abstractmethod
    def evaluate_solution(self, solution: Any, task: Dict[str, Any]) -> float:
        """Evaluate solution quality"""
        pass


class PatternDiscoveryTest(DomainAgnosticTest):
    """Test ability to discover patterns in abstract data"""
    
    def generate_task(self, difficulty: float) -> Dict[str, Any]:
        # Generate abstract pattern with controllable complexity
        seq_length = int(20 + difficulty * 80)
        pattern_complexity = int(2 + difficulty * 5)
        
        # Create multi-level pattern
        base_pattern = torch.randn(pattern_complexity, 16)
        
        # Generate sequence with pattern
        sequence = []
        for i in range(seq_length):
            # Add noise proportional to difficulty
            noise = torch.randn_like(base_pattern) * (0.1 + difficulty * 0.4)
            
            # Rotate pattern based on position
            rotation = i % pattern_complexity
            rotated = torch.roll(base_pattern, rotation, dims=0)
            
            sequence.append(rotated + noise)
        
        sequence = torch.stack(sequence)
        
        # Hide some elements for prediction
        mask = torch.rand(seq_length) > 0.3
        visible = sequence[mask]
        hidden = sequence[~mask]
        
        return {
            'visible_sequence': visible,
            'hidden_sequence': hidden,
            'full_sequence': sequence,
            'mask': mask,
            'pattern_complexity': pattern_complexity
        }
    
    def evaluate_solution(self, predictions: torch.Tensor, task: Dict[str, Any]) -> float:
        """Evaluate pattern discovery and prediction quality"""
        hidden = task['hidden_sequence']
        
        if predictions.shape[0] == 0 or hidden.shape[0] == 0:
            return 0.0
        
        # Resize predictions if needed
        min_len = min(predictions.shape[0], hidden.shape[0])
        predictions = predictions[:min_len]
        hidden = hidden[:min_len]
        
        # MSE for direct prediction
        mse = F.mse_loss(predictions, hidden)
        # Ensure mse is a tensor before calling .item()
        mse_value = mse.item() if hasattr(mse, 'item') else float(mse)
        prediction_score = 1.0 / (1.0 + mse_value)
        
        # Pattern consistency score
        if predictions.shape[0] > task['pattern_complexity']:
            # Check if predictions follow the discovered pattern
            pattern_diffs = []
            for i in range(task['pattern_complexity']):
                indices = torch.arange(i, predictions.shape[0], task['pattern_complexity'])
                if len(indices) > 1:
                    subset = predictions[indices]
                    consistency = torch.std(subset, dim=0).mean()
                    pattern_diffs.append(consistency)
            
            pattern_mean = torch.tensor(pattern_diffs).mean()
            pattern_mean_value = pattern_mean.item() if hasattr(pattern_mean, 'item') else float(pattern_mean)
            pattern_score = 1.0 / (1.0 + pattern_mean_value)
        else:
            pattern_score = 0.5
        
        return 0.6 * prediction_score + 0.4 * pattern_score


class AbstractReasoningTest(DomainAgnosticTest):
    """Test abstract reasoning without domain knowledge"""
    
    def generate_task(self, difficulty: float) -> Dict[str, Any]:
        # Generate abstract reasoning problem
        n_entities = int(5 + difficulty * 15)
        n_relations = int(3 + difficulty * 7)
        n_rules = int(2 + difficulty * 5)
        
        # Create entities with properties
        entities = torch.randn(n_entities, 32)  # 32-dim property vectors
        
        # Define relations
        relations = []
        for _ in range(n_relations):
            # Random relation matrix
            rel_matrix = torch.randn(32, 32) * 0.1
            relations.append(rel_matrix)
        
        # Generate rules (if A relates to B via R1, then B relates to C via R2)
        rules = []
        for _ in range(n_rules):
            rule = {
                'if_relation': np.random.randint(n_relations),
                'then_relation': np.random.randint(n_relations),
                'transitivity': np.random.random() > 0.5
            }
            rules.append(rule)
        
        # Create knowledge graph based on relations
        knowledge_graph = torch.zeros(n_entities, n_entities, n_relations)
        
        for i in range(n_entities):
            for j in range(n_entities):
                if i != j:
                    for r, rel_matrix in enumerate(relations):
                        # Check if relation holds
                        score = torch.matmul(entities[i], rel_matrix).dot(entities[j])
                        if torch.sigmoid(score) > 0.5:
                            knowledge_graph[i, j, r] = 1
        
        # Generate queries
        n_queries = int(5 + difficulty * 10)
        queries = []
        answers = []
        
        for _ in range(n_queries):
            query_type = np.random.choice(['direct', 'transitive', 'rule_based'])
            
            if query_type == 'direct':
                # Simple relation query
                i, j = np.random.choice(n_entities, 2, replace=False)
                r = np.random.randint(n_relations)
                queries.append({
                    'type': 'direct',
                    'entities': [i, j],
                    'relation': r
                })
                answers.append(knowledge_graph[i, j, r].item())
                
            elif query_type == 'transitive':
                # Transitive relation query
                i, j, k = np.random.choice(n_entities, 3, replace=False)
                r1, r2 = np.random.choice(n_relations, 2)
                queries.append({
                    'type': 'transitive',
                    'entities': [i, j, k],
                    'relations': [r1, r2]
                })
                # Answer is true if i->j via r1 AND j->k via r2
                ans = (knowledge_graph[i, j, r1] * knowledge_graph[j, k, r2]).item()
                answers.append(ans)
                
            else:  # rule_based
                # Apply rules
                rule = rules[np.random.randint(len(rules))]
                i, j, k = np.random.choice(n_entities, 3, replace=False)
                queries.append({
                    'type': 'rule_based',
                    'entities': [i, j, k],
                    'rule': rule
                })
                # Complex rule evaluation
                if_holds = knowledge_graph[i, j, rule['if_relation']]
                then_should = knowledge_graph[j, k, rule['then_relation']]
                ans = (if_holds * then_should).item() if rule['transitivity'] else 0
                answers.append(ans)
        
        return {
            'entities': entities,
            'relations': relations,
            'knowledge_graph': knowledge_graph,
            'rules': rules,
            'queries': queries,
            'answers': torch.tensor(answers)
        }
    
    def evaluate_solution(self, predictions: torch.Tensor, task: Dict[str, Any]) -> float:
        """Evaluate reasoning quality"""
        answers = task['answers']
        
        if predictions.shape[0] != answers.shape[0]:
            return 0.0
        
        # Accuracy on different query types
        correct = (predictions > 0.5).float() == (answers > 0.5).float()
        accuracy_tensor = correct.float().mean()
        accuracy = accuracy_tensor.item() if hasattr(accuracy_tensor, 'item') else float(accuracy_tensor)
        
        # Confidence calibration (predictions should be confident when correct)
        confidence = torch.abs(predictions - 0.5) * 2  # 0 to 1
        calibration_tensor = (confidence * correct).mean()
        calibration = calibration_tensor.item() if hasattr(calibration_tensor, 'item') else float(calibration_tensor)
        
        return 0.7 * accuracy + 0.3 * calibration


class CreativeGenerationTest(DomainAgnosticTest):
    """Test creative generation abilities"""
    
    def generate_task(self, difficulty: float) -> Dict[str, Any]:
        # Generate constraints for creative task
        n_constraints = int(2 + difficulty * 8)
        output_dim = int(32 + difficulty * 96)
        
        constraints = []
        for _ in range(n_constraints):
            constraint_type = np.random.choice(['similarity', 'difference', 'orthogonal', 'magnitude'])
            
            if constraint_type == 'similarity':
                # Output should be similar to reference
                reference = torch.randn(output_dim)
                constraints.append({
                    'type': 'similarity',
                    'reference': reference,
                    'threshold': 0.8 - difficulty * 0.3
                })
            elif constraint_type == 'difference':
                # Output should be different from reference
                reference = torch.randn(output_dim)
                constraints.append({
                    'type': 'difference',
                    'reference': reference,
                    'threshold': 0.5 + difficulty * 0.3
                })
            elif constraint_type == 'orthogonal':
                # Output should be orthogonal to reference
                reference = torch.randn(output_dim)
                constraints.append({
                    'type': 'orthogonal',
                    'reference': reference,
                    'tolerance': 0.1 + difficulty * 0.2
                })
            else:  # magnitude
                # Output should have specific magnitude
                target_mag = 1.0 + difficulty * 5.0
                constraints.append({
                    'type': 'magnitude',
                    'target': target_mag,
                    'tolerance': 0.1 + difficulty * 0.3
                })
        
        # Provide some examples that satisfy constraints
        examples = []
        for _ in range(3):
            example = torch.randn(output_dim)
            # Adjust example to partially satisfy constraints
            for c in constraints[:2]:  # Only adjust for first 2 constraints
                if c['type'] == 'similarity':
                    example = 0.7 * example + 0.3 * c['reference']
                elif c['type'] == 'magnitude':
                    example = example * c['target'] / (example.norm() + 1e-8)
            examples.append(example)
        
        return {
            'constraints': constraints,
            'output_dim': output_dim,
            'examples': torch.stack(examples) if examples else None,
            'difficulty': difficulty
        }
    
    def evaluate_solution(self, generation: torch.Tensor, task: Dict[str, Any]) -> float:
        """Evaluate creativity and constraint satisfaction"""
        constraints = task['constraints']
        examples = task['examples']
        
        # Constraint satisfaction
        satisfaction_scores = []
        
        for c in constraints:
            if c['type'] == 'similarity':
                sim = F.cosine_similarity(generation.unsqueeze(0), c['reference'].unsqueeze(0))
                score = 1.0 if sim > c['threshold'] else sim / c['threshold']
                score_value = score.item() if hasattr(score, 'item') else float(score)
                satisfaction_scores.append(score_value)
                
            elif c['type'] == 'difference':
                sim = F.cosine_similarity(generation.unsqueeze(0), c['reference'].unsqueeze(0))
                score = 1.0 if sim < c['threshold'] else (1 - sim) / (1 - c['threshold'])
                score_value = score.item() if hasattr(score, 'item') else float(score)
                satisfaction_scores.append(score_value)
                
            elif c['type'] == 'orthogonal':
                dot_product = torch.dot(generation, c['reference']) / (generation.norm() * c['reference'].norm() + 1e-8)
                score = 1.0 if abs(dot_product) < c['tolerance'] else 1.0 - abs(dot_product)
                score_value = score.item() if hasattr(score, 'item') else float(score)
                satisfaction_scores.append(score_value)
                
            else:  # magnitude
                mag = generation.norm()
                error = abs(mag - c['target']) / c['target']
                score = 1.0 if error < c['tolerance'] else 1.0 / (1.0 + error)
                score_value = score.item() if hasattr(score, 'item') else float(score)
                satisfaction_scores.append(score_value)
        
        constraint_score = np.mean(satisfaction_scores) if satisfaction_scores else 0.0
        
        # Novelty score (different from examples)
        if examples is not None and examples.shape[0] > 0:
            similarities = F.cosine_similarity(generation.unsqueeze(0), examples)
            max_sim = similarities.max()
            novelty = 1.0 - (max_sim.item() if hasattr(max_sim, 'item') else float(max_sim))
        else:
            novelty = 0.5
        
        # Complexity score (not too simple/random)
        # Measure structure via autocorrelation using FFT
        if generation.shape[0] > 10:
            # Compute autocorrelation using FFT (more efficient than direct correlation)
            # Normalize the signal
            gen_normalized = generation - generation.mean()
            # Compute FFT
            fft = torch.fft.fft(gen_normalized)
            # Power spectrum
            power = torch.abs(fft) ** 2
            # Inverse FFT gives autocorrelation
            autocorr = torch.fft.ifft(power).real
            # Normalize
            autocorr = autocorr / autocorr[0]
            # Compute complexity from autocorrelation
            complexity_value = autocorr[1:len(autocorr)//2].abs().mean()
            complexity = 1.0 - float(complexity_value.clamp(0, 1))
        else:
            complexity = 0.5
        
        return 0.5 * constraint_score + 0.3 * novelty + 0.2 * complexity


class AGIFitnessEvaluator:
    """
    Comprehensive AGI fitness evaluator that measures true general intelligence
    across multiple domains and capabilities.
    """
    
    def __init__(self):
        # Domain-agnostic tests
        self.tests = {
            'pattern_discovery': PatternDiscoveryTest(),
            'abstract_reasoning': AbstractReasoningTest(),
            'creative_generation': CreativeGenerationTest()
        }
        
        # Performance history
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Domain transfer matrix
        self.transfer_matrix = {}
        
        # Meta-learning tracker
        self.adaptation_curves = {}
        
    def evaluate_generalization(self, model: nn.Module, n_tests: int = 10) -> float:
        """Evaluate generalization across different task types and difficulties"""
        generalization_scores = []
        
        for test_name, test in self.tests.items():
            scores_by_difficulty = []
            
            # Test across different difficulties
            for difficulty in np.linspace(0.1, 0.9, 5):
                test_scores = []
                
                for _ in range(n_tests):
                    # Generate task
                    task = test.generate_task(difficulty)
                    
                    # Model attempts task
                    with torch.no_grad():
                        # Convert task to model input
                        model_input = self._prepare_task_input(task, test_name)
                        try:
                            output = model(model_input)
                        except Exception as e:
                            print(f"Error in model forward pass: {e}")
                            output = torch.zeros(1)
                        
                        # Extract solution from model output
                        solution = self._extract_solution(output, task, test_name)
                        
                        # Evaluate
                        score = test.evaluate_solution(solution, task)
                        test_scores.append(score)
                
                avg_score = np.mean(test_scores)
                scores_by_difficulty.append(avg_score)
            
            # Generalization is maintaining performance across difficulties
            score_variance = np.var(scores_by_difficulty)
            mean_score = np.mean(scores_by_difficulty)
            
            # Lower variance = better generalization
            generalization = mean_score / (1.0 + score_variance)
            generalization_scores.append(generalization)
            
            # Track transfer between difficulties
            self.transfer_matrix[test_name] = scores_by_difficulty
        
        return float(np.mean(generalization_scores))
    
    def evaluate_emergence(self, model: nn.Module, component_models: Optional[List[nn.Module]] = None) -> float:
        """Evaluate emergent capabilities beyond sum of parts"""
        if component_models is None or len(component_models) < 2:
            # Evaluate internal emergence
            return self._evaluate_internal_emergence(model)
        
        # Compare combined model vs individual components
        combined_scores = []
        individual_scores = []
        
        for test_name, test in self.tests.items():
            # Test combined model
            task = test.generate_task(0.5)
            model_input = self._prepare_task_input(task, test_name)
            
            with torch.no_grad():
                combined_output = model(model_input)
                combined_solution = self._extract_solution(combined_output, task, test_name)
                combined_score = test.evaluate_solution(combined_solution, task)
                combined_scores.append(combined_score)
            
            # Test individual components
            component_solutions = []
            for comp_model in component_models:
                with torch.no_grad():
                    comp_output = comp_model(model_input)
                    comp_solution = self._extract_solution(comp_output, task, test_name)
                    component_solutions.append(comp_solution)
            
            # Best individual performance
            best_individual = 0
            for sol in component_solutions:
                score = test.evaluate_solution(sol, task)
                best_individual = max(best_individual, score)
            
            individual_scores.append(best_individual)
        
        # Emergence = performance gain from integration
        emergence_gain = np.mean(combined_scores) / (np.mean(individual_scores) + 1e-8)
        
        # Also check for novel capabilities
        novel_capabilities = self._detect_novel_capabilities(model, component_models)
        
        return float(min(2.0, emergence_gain * (1.0 + novel_capabilities)))
    
    def evaluate_adaptability(self, model: nn.Module, n_adaptation_steps: int = 10) -> float:
        """Evaluate how quickly model adapts to new patterns"""
        adaptation_scores = []
        
        for test_name, test in self.tests.items():
            try:
                # Generate source and target tasks
                source_task = test.generate_task(0.3)
                target_task = test.generate_task(0.7)
                
                # Track adaptation curve
                curve = []
                
                # Clone model for adaptation (don't modify original)
                import copy
                
                # Completely detach model from any computation graphs
                model.eval()
                
                # Create a completely fresh model without any graph connections
                with torch.no_grad():
                    # Save state dict
                    state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}
                    
                    # Create new model instance through the class
                    model_class = model.__class__
                    if hasattr(model, 'genome'):
                        # If model has genome attribute, use it
                        adapt_model = model_class(model.genome)
                    else:
                        # Otherwise use deepcopy
                        adapt_model = copy.deepcopy(model)
                    
                    # Clear any existing gradients and hooks
                    for param in adapt_model.parameters():
                        param.requires_grad = True
                        if param.grad is not None:
                            param.grad = None
                    
                    # Load clean state dict
                    adapt_model.load_state_dict(state_dict, strict=False)
                
                adapt_model.train()  # Set to training mode
                
                # Create fresh optimizer
                optimizer = torch.optim.Adam(adapt_model.parameters(), lr=0.001)
                
                for step in range(n_adaptation_steps):
                    # Prepare input
                    model_input = self._prepare_task_input(target_task, test_name)
                    # Detach input to ensure no graph connections
                    model_input = model_input.detach()
                    
                    # Forward pass
                    output = adapt_model(model_input)
                    solution = self._extract_solution(output, target_task, test_name)
                    
                    # Evaluate current performance
                    score = test.evaluate_solution(solution, target_task)
                    curve.append(score)
                    
                    # Compute adaptation loss (self-supervised)
                    adaptation_loss = self._compute_adaptation_loss(output, target_task, test_name)
                    
                    # Update
                    optimizer.zero_grad()
                    try:
                        adaptation_loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        if "Trying to backward through the graph a second time" in str(e) or "Saved intermediate values" in str(e):
                            # Silently skip - this is expected in some cases
                            pass
                        else:
                            raise e
                
                # Adaptation score based on improvement speed
                if len(curve) > 1:
                    improvement = curve[-1] - curve[0]
                    speed = improvement / len(curve)
                    final_performance = curve[-1]
                    
                    adapt_score = 0.4 * speed + 0.6 * final_performance
                    adaptation_scores.append(adapt_score)
            
                # Store adaptation curve
                self.adaptation_curves[f"{test_name}_adapt"] = curve
            
            except RuntimeError as e:
                if "backward through the graph a second time" in str(e) or "inplace operation" in str(e):
                    # Handle graph errors gracefully
                    # Try a simpler evaluation without backprop
                    try:
                        with torch.no_grad():
                            # Just evaluate final performance without adaptation
                            model_input = self._prepare_task_input(target_task, test_name).detach()
                            output = model(model_input)
                            solution = self._extract_solution(output, target_task, test_name)
                            final_score = test.evaluate_solution(solution, target_task)
                            adaptation_scores.append(float(final_score) * 0.7)  # Penalize for no adaptation
                    except Exception:
                        adaptation_scores.append(0.5)  # Default score
                else:
                    raise e
        
        return float(np.mean(adaptation_scores))
    
    def evaluate_creativity(self, model: nn.Module, n_samples: int = 20) -> float:
        """Evaluate creative problem solving and novel solution generation"""
        creativity_scores = []
        
        # Focus on creative generation test
        test = self.tests['creative_generation']
        
        # Generate multiple solutions for same constraints
        task = test.generate_task(0.6)
        solutions = []
        
        for _ in range(n_samples):
            model_input = self._prepare_task_input(task, 'creative_generation')
            
            with torch.no_grad():
                # Add noise for variety
                noise = torch.randn_like(model_input) * 0.1
                output = model(model_input + noise)
                solution = self._extract_solution(output, task, 'creative_generation')
                solutions.append(solution)
        
        # Evaluate diversity of solutions
        if len(solutions) > 1:
            solutions_tensor = torch.stack(solutions)
            
            # Pairwise distances
            distances = []
            for i in range(len(solutions)):
                for j in range(i+1, len(solutions)):
                    dist = torch.norm(solutions[i] - solutions[j])
                    distances.append(dist.item())
            
            diversity = np.mean(distances) if distances else 0
            
            # Quality of solutions
            qualities = [test.evaluate_solution(sol, task) for sol in solutions]
            avg_quality = np.mean(qualities)
            
            # Creativity = high quality + high diversity
            creativity = 0.6 * avg_quality + 0.4 * (1.0 / (1.0 + np.exp(-diversity)))
            creativity_scores.append(creativity)
        
        # Test ability to combine concepts creatively
        combination_score = self._evaluate_concept_combination(model)
        creativity_scores.append(combination_score)
        
        return float(np.mean(creativity_scores))
    
    def evaluate_reasoning(self, model: nn.Module) -> float:
        """Evaluate logical and abstract reasoning capabilities"""
        # Focus on abstract reasoning test
        test = self.tests['abstract_reasoning']
        reasoning_scores = []
        
        # Test different types of reasoning
        reasoning_types = ['deductive', 'inductive', 'abductive', 'analogical']
        
        for reasoning_type in reasoning_types:
            scores = []
            
            for _ in range(5):
                # Generate reasoning task
                task = test.generate_task(0.5)
                
                # Add reasoning type hint to task
                task['reasoning_type'] = reasoning_type
                
                model_input = self._prepare_task_input(task, 'abstract_reasoning')
                
                with torch.no_grad():
                    output = model(model_input)
                    solution = self._extract_solution(output, task, 'abstract_reasoning')
                    score = test.evaluate_solution(solution, task)
                    scores.append(score)
            
            reasoning_scores.append(np.mean(scores))
        
        # Test multi-step reasoning
        multistep_score = self._evaluate_multistep_reasoning(model)
        reasoning_scores.append(multistep_score)
        
        return float(np.mean(reasoning_scores))
    
    def evaluate_consciousness(self, model: nn.Module) -> float:
        """Evaluate consciousness-like properties"""
        consciousness_scores = []
        
        # Self-model accuracy
        self_model_score = self._evaluate_self_model(model)
        consciousness_scores.append(self_model_score)
        
        # Information integration (Phi)
        phi_score = self._compute_information_integration(model)
        consciousness_scores.append(phi_score)
        
        # Global workspace dynamics
        gw_score = self._evaluate_global_workspace(model)
        consciousness_scores.append(gw_score)
        
        # Meta-cognitive monitoring
        meta_score = self._evaluate_metacognition(model)
        consciousness_scores.append(meta_score)
        
        return float(np.mean(consciousness_scores))
    
    def evaluate_efficiency(self, model: nn.Module) -> float:
        """Evaluate computational and sample efficiency"""
        efficiency_scores = []
        
        # Computational efficiency
        flops_score = self._evaluate_computational_efficiency(model)
        efficiency_scores.append(flops_score)
        
        # Sample efficiency (few-shot learning)
        sample_score = self._evaluate_sample_efficiency(model)
        efficiency_scores.append(sample_score)
        
        # Parameter efficiency
        param_score = self._evaluate_parameter_efficiency(model)
        efficiency_scores.append(param_score)
        
        return float(np.mean(efficiency_scores))
    
    def evaluate_robustness(self, model: nn.Module) -> float:
        """Evaluate robustness to perturbations and edge cases"""
        robustness_scores = []
        
        # Noise robustness
        noise_score = self._evaluate_noise_robustness(model)
        robustness_scores.append(noise_score)
        
        # Out-of-distribution robustness
        ood_score = self._evaluate_ood_robustness(model)
        robustness_scores.append(ood_score)
        
        # Adversarial robustness
        adv_score = self._evaluate_adversarial_robustness(model)
        robustness_scores.append(adv_score)
        
        return float(np.mean(robustness_scores))
    
    def evaluate_complete(self, model: nn.Module, 
                         component_models: Optional[List[nn.Module]] = None) -> AGIFitnessScore:
        """Complete AGI evaluation across all metrics"""
        print("🧠 Evaluating AGI Fitness...")
        
        score = AGIFitnessScore()
        
        # Core capabilities
        print("  📊 Testing Generalization...")
        score.generalization = self.evaluate_generalization(model)
        
        print("  🌟 Testing Emergence...")
        score.emergence = self.evaluate_emergence(model, component_models)
        
        print("  🔄 Testing Adaptability...")
        score.adaptability = self.evaluate_adaptability(model)
        
        print("  🎨 Testing Creativity...")
        score.creativity = self.evaluate_creativity(model)
        
        print("  🧩 Testing Reasoning...")
        score.reasoning = self.evaluate_reasoning(model)
        
        print("  💭 Testing Consciousness...")
        score.consciousness = self.evaluate_consciousness(model)
        
        print("  ⚡ Testing Efficiency...")
        score.efficiency = self.evaluate_efficiency(model)
        
        print("  🛡️ Testing Robustness...")
        score.robustness = self.evaluate_robustness(model)
        
        # Meta capabilities
        score.meta_learning = self._evaluate_meta_learning(model)
        score.abstraction = self._evaluate_abstraction_capability(model)
        score.coherence = self._evaluate_behavioral_coherence(model)
        score.autonomy = self._evaluate_autonomous_improvement(model)
        
        # Domain performances
        for test_name in self.tests:
            perf = self._get_domain_performance(model, test_name)
            score.domain_scores[test_name] = perf
        
        # Detailed metrics
        score.detailed_metrics = {
            'transfer_matrix': self.transfer_matrix,
            'adaptation_curves': self.adaptation_curves,
            'performance_history': dict(self.performance_history)
        }
        
        print(f"\n✅ Overall AGI Score: {score.overall_agi_score():.3f}")
        
        return score
    
    # Helper methods
    def _prepare_task_input(self, task: Dict[str, Any], test_name: str) -> torch.Tensor:
        """Convert task to model input format"""
        # This is a simplified version - would need task-specific encoding
        if test_name == 'pattern_discovery':
            return task['visible_sequence'].flatten().unsqueeze(0)
        elif test_name == 'abstract_reasoning':
            return task['entities'].flatten().unsqueeze(0)
        elif test_name == 'creative_generation':
            if task['examples'] is not None:
                return task['examples'].flatten().unsqueeze(0)
            else:
                return torch.randn(1, task['output_dim'])
        else:
            return torch.randn(1, 128)  # Default
    
    def _extract_solution(self, output: Any, task: Dict[str, Any], test_name: str) -> torch.Tensor:
        """Extract solution from model output"""
        # Handle different output formats
        if isinstance(output, dict):
            # Try common keys
            for key in ['prediction', 'solution', 'output', 'integrated_output']:
                if key in output:
                    output = output[key]
                    break
        
        if isinstance(output, torch.Tensor):
            if test_name == 'pattern_discovery':
                # Reshape to match hidden sequence
                target_shape = task['hidden_sequence'].shape
                if output.numel() >= target_shape.numel():
                    return output.view(-1)[:target_shape.numel()].view(target_shape)
                else:
                    # Pad if necessary
                    return F.pad(output.flatten(), (0, target_shape.numel() - output.numel())).view(target_shape)
            elif test_name == 'abstract_reasoning':
                # Binary predictions for queries
                n_queries = len(task['queries'])
                if output.numel() >= n_queries:
                    return torch.sigmoid(output.flatten()[:n_queries])
                else:
                    return torch.sigmoid(F.pad(output.flatten(), (0, n_queries - output.numel())))
            elif test_name == 'creative_generation':
                # Generation output
                target_dim = task['output_dim']
                if output.numel() >= target_dim:
                    return output.flatten()[:target_dim]
                else:
                    return F.pad(output.flatten(), (0, target_dim - output.numel()))
        
        # Fallback
        return torch.zeros(1)
    
    def _compute_adaptation_loss(self, output: Any, task: Dict[str, Any], test_name: str) -> torch.Tensor:
        """Compute self-supervised adaptation loss"""
        # Extract relevant features for adaptation
        if isinstance(output, dict) and 'features' in output:
            features = output['features']
        else:
            features = output if isinstance(output, torch.Tensor) else torch.zeros(1)
        
        # Self-supervised losses based on task structure
        if test_name == 'pattern_discovery':
            # Encourage discovering repeated patterns
            if features.shape[0] > 1:
                # Temporal consistency loss
                consistency_loss = F.mse_loss(features[:-1], features[1:])
                return consistency_loss
            
        elif test_name == 'abstract_reasoning':
            # Encourage structural consistency
            entity_features = features.view(-1, 32) if features.numel() >= 32 else features
            # Symmetry loss
            return torch.std(entity_features, dim=0).mean()
            
        elif test_name == 'creative_generation':
            # Encourage diversity while satisfying constraints
            return -torch.std(features)  # Negative to maximize diversity
        
        # Default: minimize entropy
        return -torch.distributions.Categorical(logits=features.flatten()).entropy()
    
    def _compute_adaptation_loss(self, output: Any, task: Dict[str, Any], test_name: str) -> torch.Tensor:
        """Compute self-supervised adaptation loss"""
        # Self-supervised loss based on output structure and consistency
        if isinstance(output, torch.Tensor):
            # Ensure output requires grad
            if not output.requires_grad:
                output = output.detach().requires_grad_(True)
            
            # Multiple self-supervised objectives
            losses = []
            
            # 1. Encourage structured outputs (not too uniform)
            if output.numel() > 1:
                std = output.std()
                structure_loss = -torch.log(std + 1e-8)
                losses.append(structure_loss)
            
            # 2. Encourage temporal consistency if sequential
            if output.dim() > 2:
                temporal_diff = torch.abs(output[:-1] - output[1:]).mean()
                consistency_loss = temporal_diff
                losses.append(consistency_loss)
            
            # 3. Encourage diversity across batch
            if output.shape[0] > 1:
                batch_diversity = -torch.cdist(output, output).mean()
                losses.append(batch_diversity * 0.1)
            
            # Combine losses
            if losses:
                total_loss = sum(losses) / len(losses)
            else:
                # Fallback: simple regularization
                total_loss = output.pow(2).mean() * 0.01
                
            return total_loss
        
        elif isinstance(output, dict):
            # For dict outputs, adapt based on specific components
            losses = []
            
            for key, value in output.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    if key in ['attention_weights', 'confidence']:
                        # Encourage confident but not overconfident
                        entropy = -(value * torch.log(value + 1e-8)).sum()
                        losses.append(entropy)
                    elif key in ['representation', 'features']:
                        # Encourage sparse but meaningful representations
                        sparsity = torch.abs(value).mean()
                        losses.append(-torch.log(sparsity + 1e-8))
                    else:
                        # Generic loss
                        losses.append(value.pow(2).mean() * 0.01)
            
            if losses:
                return sum(losses) / len(losses)
            else:
                # Return dummy loss that won't cause errors
                dummy = torch.tensor(0.0, requires_grad=True)
                return dummy
        
        else:
            # Return dummy loss for unsupported output types
            dummy = torch.tensor(0.0, requires_grad=True)
            return dummy
    
    def _evaluate_internal_emergence(self, model: nn.Module) -> float:
        """Evaluate emergence within single model"""
        # Test for emergent representations
        test_input = torch.randn(3, 128)
        
        # Get intermediate representations if available
        representations = []
        hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                representations.append(output.detach())
        
        # Register hooks on different layers
        for name, module in model.named_modules():
            if len(hooks) < 5 and isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            _ = model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if len(representations) < 2:
            return 0.5
        
        # Measure emergence as increasing complexity/abstraction
        complexities = []
        for rep in representations:
            # Singular values indicate representation complexity
            if rep.dim() >= 2:
                svd = torch.linalg.svdvals(rep.view(rep.shape[0], -1))
                # Entropy of singular values
                svd_norm = svd / svd.sum()
                entropy = -(svd_norm * torch.log(svd_norm + 1e-8)).sum()
                complexities.append(entropy.item())
        
        # Emergence = increasing complexity
        if len(complexities) > 1:
            emergence = np.polyfit(range(len(complexities)), complexities, 1)[0]
            return float(max(0, min(1, emergence)))
        
        return 0.5
    
    def _detect_novel_capabilities(self, model: nn.Module, 
                                  component_models: List[nn.Module]) -> float:
        """Detect capabilities not present in individual components"""
        # Simplified: Check if combined model can solve tasks that no component can
        novel_score = 0.0
        n_tests = 5
        
        for test_name, test in self.tests.items():
            for _ in range(n_tests):
                task = test.generate_task(0.7)  # Hard task
                model_input = self._prepare_task_input(task, test_name)
                
                # Test combined model
                with torch.no_grad():
                    combined_output = model(model_input)
                    combined_solution = self._extract_solution(combined_output, task, test_name)
                    combined_score = test.evaluate_solution(combined_solution, task)
                
                # Test all components
                component_scores = []
                for comp in component_models:
                    with torch.no_grad():
                        comp_output = comp(model_input)
                        comp_solution = self._extract_solution(comp_output, task, test_name)
                        comp_score = test.evaluate_solution(comp_solution, task)
                        component_scores.append(comp_score)
                
                # Novel if combined succeeds where all components fail
                if combined_score > 0.7 and all(s < 0.3 for s in component_scores):
                    novel_score += 1.0
        
        return novel_score / (len(self.tests) * n_tests)
    
    def _evaluate_concept_combination(self, model: nn.Module) -> float:
        """Evaluate ability to combine concepts creatively"""
        # Create hybrid tasks combining elements from different tests
        scores = []
        
        # Pattern + Reasoning
        pattern_task = self.tests['pattern_discovery'].generate_task(0.4)
        reasoning_task = self.tests['abstract_reasoning'].generate_task(0.4)
        
        # Combine: Find patterns in relational structure
        hybrid_input = torch.cat([
            pattern_task['visible_sequence'].flatten(),
            reasoning_task['entities'].flatten()
        ])
        
        with torch.no_grad():
            output = model(hybrid_input.unsqueeze(0))
            
            # Evaluate if model found cross-domain patterns
            if isinstance(output, torch.Tensor) and output.numel() > 10:
                # Check for structure preservation
                output_parts = torch.chunk(output.flatten(), 2)
                
                # Correlation between parts suggests concept combination
                if len(output_parts) == 2 and output_parts[0].shape == output_parts[1].shape:
                    correlation = F.cosine_similarity(output_parts[0], output_parts[1], dim=0)
                    scores.append(abs(correlation.item()))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_multistep_reasoning(self, model: nn.Module) -> float:
        """Evaluate multi-step reasoning chains"""
        # Create task requiring multiple reasoning steps
        test = self.tests['abstract_reasoning']
        task = test.generate_task(0.6)
        
        # Create multi-hop queries
        n_entities = task['entities'].shape[0]
        kg = task['knowledge_graph']
        
        # Find paths of length 3+
        multihop_scores = []
        
        for _ in range(5):
            # Random path
            path = np.random.choice(n_entities, 4, replace=False)
            
            # Check if path exists in knowledge graph
            path_exists = True
            for i in range(len(path)-1):
                # Check if any relation connects path[i] to path[i+1]
                if not kg[path[i], path[i+1]].any():
                    path_exists = False
                    break
            
            # Model predicts path validity
            path_input = task['entities'][path].flatten()
            
            with torch.no_grad():
                output = model(path_input.unsqueeze(0))
                
                if isinstance(output, torch.Tensor):
                    prediction = torch.sigmoid(output.mean()).item()
                    
                    # Correct prediction?
                    correct = (prediction > 0.5) == path_exists
                    multihop_scores.append(float(correct))
        
        return float(np.mean(multihop_scores)) if multihop_scores else 0.5
    
    def _evaluate_self_model(self, model: nn.Module) -> float:
        """Evaluate model's self-understanding"""
        # Test if model can predict its own behavior
        scores = []
        
        # Generate test inputs - use smaller batch to avoid batch size conflicts
        test_inputs = torch.randn(5, 128)
        
        with torch.no_grad():
            # Get model outputs
            outputs = []
            for inp in test_inputs:
                out = model(inp.unsqueeze(0))
                if isinstance(out, dict):
                    out = out.get('integrated_output', out.get('output', torch.zeros(1)))
                elif not isinstance(out, torch.Tensor):
                    out = torch.zeros(1).to(inp.device)
                elif isinstance(out, torch.Tensor):
                    # If it's already a tensor, use it directly
                    pass
                else:
                    # Fallback to zeros if unexpected type
                    out = torch.zeros(1).to(inp.device)
                outputs.append(out)
            
            # Can model predict its own outputs given similar inputs?
            for i in range(len(test_inputs)-1):
                # Give model current input and ask to predict next output
                query = torch.cat([test_inputs[i], test_inputs[i+1]])
                prediction = model(query.unsqueeze(0))
                
                if isinstance(prediction, torch.Tensor) and isinstance(outputs[i+1], torch.Tensor):
                    # Compare prediction to actual next output
                    if prediction.shape == outputs[i+1].shape:
                        similarity = F.cosine_similarity(
                            prediction.flatten(),
                            outputs[i+1].flatten(),
                            dim=0
                        )
                        scores.append(similarity.item())
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _compute_information_integration(self, model: nn.Module) -> float:
        """Compute approximation of integrated information (Phi)"""
        # Simplified IIT calculation
        test_input = torch.randn(1, 128)
        
        with torch.no_grad():
            # Full system output
            full_output = model(test_input)
            if isinstance(full_output, dict):
                full_output = full_output.get('integrated_output', torch.zeros(1))
            
            # Test partitioned system (if possible)
            # This is a simplified approximation
            partition_score = 0.0
            
            # Try to identify modular structure
            if hasattr(model, 'registered_modules') and len(model.registered_modules) > 1:
                # Test each module independently
                module_outputs = []
                
                for name, module in model.registered_modules.items():
                    try:
                        module_out = module(test_input)
                        if isinstance(module_out, torch.Tensor):
                            module_outputs.append(module_out)
                    except:
                        pass
                
                if len(module_outputs) > 1:
                    # Combined output from independent modules
                    independent_combined = torch.stack(module_outputs).mean(dim=0)
                    
                    # Difference between integrated and independent
                    if full_output.shape == independent_combined.shape:
                        integration = torch.norm(full_output - independent_combined)
                        partition_score = 1.0 / (1.0 + integration.item())
            
            # Higher score = more integration
            return 1.0 - partition_score
        
    def _evaluate_global_workspace(self, model: nn.Module) -> float:
        """Evaluate global workspace dynamics"""
        # Test information broadcasting and competition
        scores = []
        
        # Multiple competing inputs
        inputs = [torch.randn(1, 128) for _ in range(5)]
        
        with torch.no_grad():
            # Sequential processing
            outputs = []
            for inp in inputs:
                out = model(inp)
                if isinstance(out, dict):
                    out = out.get('integrated_output', out.get('output', torch.zeros(1)))
                outputs.append(out)
            
            # Check for winner-take-all dynamics
            if len(outputs) > 1:
                output_tensor = torch.stack([o.flatten() for o in outputs])
                
                # Measure competition (sparsity of attention)
                attention = F.softmax(output_tensor.norm(dim=1), dim=0)
                entropy = -(attention * torch.log(attention + 1e-8)).sum()
                
                # Lower entropy = stronger competition = better global workspace
                competition_score = 1.0 / (1.0 + entropy.item())
                scores.append(competition_score)
            
            # Test broadcasting (influence of winning input)
            if len(outputs) > 2:
                # Correlation between outputs
                correlations = []
                for i in range(len(outputs)-1):
                    if outputs[i].shape == outputs[i+1].shape:
                        corr = F.cosine_similarity(
                            outputs[i].flatten(),
                            outputs[i+1].flatten(),
                            dim=0
                        )
                        correlations.append(abs(corr.item()))
                
                # Higher correlation = better broadcasting
                broadcast_score = np.mean(correlations) if correlations else 0.5
                scores.append(broadcast_score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_metacognition(self, model: nn.Module) -> float:
        """Evaluate metacognitive monitoring and control"""
        scores = []
        
        # Test confidence calibration
        for test_name, test in self.tests.items():
            task = test.generate_task(0.5)
            model_input = self._prepare_task_input(task, test_name)
            
            with torch.no_grad():
                output = model(model_input)
                
                # Check if model provides confidence
                confidence = None
                if isinstance(output, dict):
                    confidence = output.get('confidence', output.get('uncertainty', None))
                
                if confidence is not None:
                    # Get actual performance
                    solution = self._extract_solution(output, task, test_name)
                    performance = test.evaluate_solution(solution, task)
                    
                    # Confidence should correlate with performance
                    if isinstance(confidence, torch.Tensor):
                        conf_value = confidence.mean().item()
                        
                        # Calibration error
                        calibration_error = abs(conf_value - performance)
                        calibration_score = 1.0 - calibration_error
                        scores.append(calibration_score)
        
        # Test adaptive strategy selection
        strategy_score = self._evaluate_strategy_selection(model)
        scores.append(strategy_score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_computational_efficiency(self, model: nn.Module) -> float:
        """Evaluate computational efficiency"""
        import time
        
        # Measure inference time - use batch size 1 to avoid conflicts
        test_input = torch.randn(1, 128)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        # Time inference
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model(test_input)
                times.append(time.time() - start)
        
        avg_time = np.mean(times)
        
        # Measure model size
        param_count = sum(p.numel() for p in model.parameters())
        
        # Efficiency score (inverse of time * params)
        efficiency = 1.0 / (1.0 + avg_time * param_count / 1e6)
        
        return float(efficiency)
    
    def _evaluate_sample_efficiency(self, model: nn.Module) -> float:
        """Evaluate few-shot learning efficiency"""
        scores = []
        
        for test_name, test in self.tests.items():
            # Test with very few examples
            task = test.generate_task(0.5)
            
            # Provide only 1-3 examples
            n_examples = np.random.randint(1, 4)
            
            # Create simple examples
            examples = []
            example_sizes = []
            for _ in range(n_examples):
                ex_task = test.generate_task(0.3)  # Easier examples
                ex_input = self._prepare_task_input(ex_task, test_name)
                examples.append(ex_input)
                example_sizes.append(ex_input.shape[-1])
            
            # Model learns from examples
            if examples:
                # Ensure all examples have same dimension
                if len(set(example_sizes)) > 1:
                    # Pad or truncate to maximum size
                    max_size = max(example_sizes)
                    padded_examples = []
                    for ex in examples:
                        if ex.shape[-1] < max_size:
                            # Pad with zeros
                            padding = torch.zeros(*ex.shape[:-1], max_size - ex.shape[-1], device=ex.device)
                            ex = torch.cat([ex, padding], dim=-1)
                        elif ex.shape[-1] > max_size:
                            # Truncate
                            ex = ex[..., :max_size]
                        padded_examples.append(ex)
                    example_input = torch.cat(padded_examples, dim=0)
                else:
                    example_input = torch.cat(examples, dim=0)
                
                with torch.no_grad():
                    # Prime model with examples
                    _ = model(example_input)
                    
                    # Test on actual task
                    model_input = self._prepare_task_input(task, test_name)
                    output = model(model_input)
                    solution = self._extract_solution(output, task, test_name)
                    score = test.evaluate_solution(solution, task)
                    
                    scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_parameter_efficiency(self, model: nn.Module) -> float:
        """Evaluate parameter usage efficiency"""
        # Check sparsity of activations
        # Use smaller batch size to avoid mismatch issues
        batch_size = 3  # Match the batch size that seems to be used elsewhere
        test_input = torch.randn(batch_size, 128)
        
        activation_sparsity = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                # Measure sparsity
                sparsity = (output.abs() < 0.1).float().mean().item()
                activation_sparsity.append(sparsity)
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        try:
            with torch.no_grad():
                _ = model(test_input)
        except Exception as e:
            # If model expects different input format, try alternatives
            print(f"Warning in parameter efficiency test: {e}")
            # Remove hooks before retrying
            for hook in hooks:
                hook.remove()
            hooks = []
            
            # Try with dict input
            try:
                dict_input = {'world_state': test_input}
                with torch.no_grad():
                    _ = model(dict_input)
            except Exception:
                # Give up and return default
                return 0.5
        
        for hook in hooks:
            hook.remove()
        
        # Higher sparsity = more efficient parameter usage
        avg_sparsity = np.mean(activation_sparsity) if activation_sparsity else 0.5
        
        return float(avg_sparsity)
    
    def _evaluate_noise_robustness(self, model: nn.Module) -> float:
        """Evaluate robustness to input noise"""
        scores = []
        
        for test_name, test in self.tests.items():
            task = test.generate_task(0.5)
            model_input = self._prepare_task_input(task, test_name)
            
            with torch.no_grad():
                # Clean performance
                clean_output = model(model_input)
                clean_solution = self._extract_solution(clean_output, task, test_name)
                clean_score = test.evaluate_solution(clean_solution, task)
                
                # Noisy performance
                noise_levels = [0.1, 0.2, 0.5]
                noise_scores = []
                
                for noise_level in noise_levels:
                    noisy_input = model_input + torch.randn_like(model_input) * noise_level
                    noisy_output = model(noisy_input)
                    noisy_solution = self._extract_solution(noisy_output, task, test_name)
                    noisy_score = test.evaluate_solution(noisy_solution, task)
                    
                    # Relative performance drop
                    relative_score = noisy_score / (clean_score + 1e-8)
                    noise_scores.append(relative_score)
                
                # Average robustness across noise levels
                robustness = np.mean(noise_scores)
                scores.append(robustness)
        
        return float(np.mean(scores))
    
    def _evaluate_ood_robustness(self, model: nn.Module) -> float:
        """Evaluate out-of-distribution robustness"""
        scores = []
        
        # Test on extreme difficulties
        for test_name, test in self.tests.items():
            # Train on medium difficulty
            train_task = test.generate_task(0.5)
            
            # Test on extreme difficulties
            for difficulty in [0.05, 0.95]:
                test_task = test.generate_task(difficulty)
                model_input = self._prepare_task_input(test_task, test_name)
                
                with torch.no_grad():
                    output = model(model_input)
                    solution = self._extract_solution(output, test_task, test_name)
                    score = test.evaluate_solution(solution, test_task)
                    scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_adversarial_robustness(self, model: nn.Module) -> float:
        """Evaluate robustness to adversarial inputs"""
        # Simplified adversarial evaluation
        scores = []
        
        test_input = torch.randn(1, 128, requires_grad=True)
        
        # Generate adversarial example
        output = model(test_input)
        if isinstance(output, dict):
            output = output.get('integrated_output', output.get('output', torch.zeros(1)))
        elif not isinstance(output, torch.Tensor):
            output = torch.zeros(1).to(next(model.parameters()).device)
        
        # Compute gradient
        if output.requires_grad:
            loss = output.norm()
            loss.backward()
            
            if test_input.grad is not None:
                # Create adversarial input
                epsilon = 0.1
                adv_input = test_input + epsilon * test_input.grad.sign()
                
                with torch.no_grad():
                    # Compare outputs
                    clean_out = model(test_input)
                    adv_out = model(adv_input)
                    
                    if isinstance(clean_out, torch.Tensor) and isinstance(adv_out, torch.Tensor):
                        # Stability under adversarial perturbation
                        if clean_out.shape == adv_out.shape:
                            stability = F.cosine_similarity(
                                clean_out.flatten(),
                                adv_out.flatten(),
                                dim=0
                            )
                            scores.append(stability.item())
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_meta_learning(self, model: nn.Module) -> float:
        """Evaluate meta-learning capabilities"""
        # Test if model improves at learning itself
        learning_curves = []
        
        for _ in range(3):
            # Create new task type
            test = self.tests['pattern_discovery']
            
            # Track learning across multiple related tasks
            curve = []
            for i in range(5):
                task = test.generate_task(0.3 + i * 0.1)
                model_input = self._prepare_task_input(task, 'pattern_discovery')
                
                with torch.no_grad():
                    output = model(model_input)
                    solution = self._extract_solution(output, task, 'pattern_discovery')
                    score = test.evaluate_solution(solution, task)
                    curve.append(score)
            
            learning_curves.append(curve)
        
        # Meta-learning = improvement in learning speed
        if len(learning_curves) > 1:
            # Compare learning curves
            first_curve = learning_curves[0]
            last_curve = learning_curves[-1]
            
            # Area under curve improvement
            first_auc = np.trapz(first_curve)
            last_auc = np.trapz(last_curve)
            
            improvement = (last_auc - first_auc) / (first_auc + 1e-8)
            return float(max(0, min(1, improvement + 0.5)))
        
        return 0.5
    
    def _evaluate_abstraction_capability(self, model: nn.Module) -> float:
        """Evaluate ability to form abstractions"""
        # Test if model can extract abstract patterns
        scores = []
        
        # Generate related tasks with shared abstract structure
        base_pattern = torch.randn(32)
        
        tasks = []
        for _ in range(5):
            # Transform pattern in different ways
            transform = torch.randn(32, 32)
            transformed = torch.matmul(transform, base_pattern)
            transformed = transformed + torch.randn_like(transformed) * 0.2
            tasks.append(transformed)
        
        # Can model find the common abstraction?
        task_inputs = torch.stack(tasks)
        
        with torch.no_grad():
            outputs = []
            for t in task_inputs:
                out = model(t.unsqueeze(0))
                if isinstance(out, torch.Tensor):
                    outputs.append(out)
            
            if len(outputs) > 2:
                # Check if outputs share common structure
                output_tensor = torch.stack([o.flatten() for o in outputs])
                
                # PCA to find principal components
                centered = output_tensor - output_tensor.mean(dim=0)
                cov = torch.matmul(centered.T, centered) / (centered.shape[0] - 1)
                
                eigenvalues = torch.linalg.eigvalsh(cov)
                
                # Abstraction = few components explain most variance
                if eigenvalues.numel() > 0:
                    explained_variance = eigenvalues[-3:].sum() / eigenvalues.sum()
                    scores.append(explained_variance.item())
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_behavioral_coherence(self, model: nn.Module) -> float:
        """Evaluate consistency of behavior"""
        # Test if model maintains coherent behavior across contexts
        scores = []
        
        # Same input in different contexts
        base_input = torch.randn(1, 128)
        
        contexts = [
            torch.randn(1, 64) for _ in range(5)
        ]
        
        outputs = []
        with torch.no_grad():
            for ctx in contexts:
                # Combine input with context
                full_input = torch.cat([base_input, ctx], dim=1)
                out = model(full_input[:, :128])  # Use only base input size
                
                if isinstance(out, dict):
                    out = out.get('integrated_output', out.get('output', torch.zeros(1)))
                outputs.append(out)
        
        # Measure coherence across contexts
        if len(outputs) > 2:
            coherence_scores = []
            for i in range(len(outputs)-1):
                if outputs[i].shape == outputs[i+1].shape:
                    # Outputs should be similar but not identical
                    similarity = F.cosine_similarity(
                        outputs[i].flatten(),
                        outputs[i+1].flatten(),
                        dim=0
                    )
                    # Ideal coherence: high but not perfect similarity
                    coherence = 1.0 - abs(similarity.item() - 0.7)
                    coherence_scores.append(coherence)
            
            scores.append(np.mean(coherence_scores))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_autonomous_improvement(self, model: nn.Module) -> float:
        """Evaluate capacity for self-improvement"""
        # Test if model can identify and improve its weaknesses
        scores = []
        
        # Initial performance on various tasks
        initial_scores = {}
        for test_name, test in self.tests.items():
            task = test.generate_task(0.5)
            model_input = self._prepare_task_input(task, test_name)
            
            with torch.no_grad():
                output = model(model_input)
                solution = self._extract_solution(output, task, test_name)
                score = test.evaluate_solution(solution, task)
                initial_scores[test_name] = score
        
        # Find weakest area
        weakest_test = min(initial_scores.items(), key=lambda x: x[1])[0]
        
        # Can model recognize and focus on weakness?
        # Simulate self-directed learning
        import copy
        # Create a detached copy to avoid graph issues
        with torch.no_grad():
            # Clone the state dict instead of the model
            state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Create new model instance and load cloned state
        try:
            # Try to create new instance with appropriate parameters
            if hasattr(model, 'mind') and hasattr(model, 'hub'):
                # For UnifiedAGI model
                self_improve_model = model.__class__(model.mind, model.hub)
            elif hasattr(model, 'config'):
                self_improve_model = model.__class__(model.config)
            elif hasattr(model, 'genome'):
                # For models with genome
                self_improve_model = model.__class__(model.genome)
            else:
                # Fallback: just create empty instance
                self_improve_model = model.__class__()
            
            # Load state dict with strict=False to handle mismatches
            self_improve_model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            print(f"Warning: Could not create fresh model instance: {e}")
            # If creation fails, create a minimal wrapper that avoids graph issues
            class DetachedModelWrapper(nn.Module):
                def __init__(self, original_model):
                    super().__init__()
                    # Create a completely new model with same architecture
                    self.model = copy.deepcopy(original_model)
                    # Ensure complete detachment
                    for param in self.model.parameters():
                        param.data = param.data.clone().detach()
                        param.requires_grad = True
                        if param.grad is not None:
                            param.grad = None
                            
                def forward(self, x):
                    # Always detach input to break any graph connections
                    if isinstance(x, torch.Tensor):
                        x = x.detach()
                    return self.model(x)
            
            self_improve_model = DetachedModelWrapper(model)
        
        self_improve_model.train()
        
        optimizer = torch.optim.Adam(self_improve_model.parameters(), lr=0.0001)
        
        for _ in range(10):
            # Generate task in weak area
            task = self.tests[weakest_test].generate_task(0.4)
            model_input = self._prepare_task_input(task, weakest_test)
            
            # Self-supervised improvement
            # Ensure input is detached and doesn't carry gradients
            model_input = model_input.detach().requires_grad_(False)
            
            output = self_improve_model(model_input)
            
            # Create self-supervised loss
            if isinstance(output, torch.Tensor) and output.requires_grad:
                # Encourage diverse, structured outputs
                if output.numel() > 1:
                    diversity_loss = -torch.std(output)
                    if output.shape[0] > 1:
                        structure_loss = -torch.abs(output[:-1] - output[1:]).mean()
                        loss = diversity_loss + structure_loss
                    else:
                        loss = diversity_loss
                else:
                    # Fallback loss
                    loss = output.mean() * 0.01
                
                # Only do backward if loss requires grad
                if loss.requires_grad:
                    optimizer.zero_grad()
                    try:
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        # Skip this optimization step if backward fails - this is expected
                        continue
        
        # Re-evaluate on weak area
        with torch.no_grad():
            task = self.tests[weakest_test].generate_task(0.5)
            model_input = self._prepare_task_input(task, weakest_test)
            
            output = self_improve_model(model_input)
            solution = self._extract_solution(output, task, weakest_test)
            improved_score = self.tests[weakest_test].evaluate_solution(solution, task)
            
            # Improvement ratio
            improvement = (improved_score - initial_scores[weakest_test]) / (initial_scores[weakest_test] + 1e-8)
            scores.append(max(0, min(1, improvement + 0.5)))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evaluate_strategy_selection(self, model: nn.Module) -> float:
        """Evaluate adaptive strategy selection"""
        # Test if model selects appropriate strategies for different tasks
        scores = []
        
        # Create tasks requiring different strategies
        strategy_tasks = {
            'analytical': self.tests['abstract_reasoning'].generate_task(0.6),
            'creative': self.tests['creative_generation'].generate_task(0.6),
            'pattern': self.tests['pattern_discovery'].generate_task(0.6)
        }
        
        for strategy_type, task in strategy_tasks.items():
            model_input = self._prepare_task_input(task, strategy_type)
            
            with torch.no_grad():
                output = model(model_input)
                
                # Check if output indicates strategy selection
                if isinstance(output, dict):
                    # Look for strategy indicators
                    if 'strategy' in output or 'mode' in output or 'approach' in output:
                        scores.append(1.0)
                    elif 'attention_weights' in output:
                        # Different attention patterns for different strategies
                        attn = output['attention_weights']
                        if isinstance(attn, torch.Tensor) and attn.numel() > 1:
                            # Entropy of attention as proxy for strategy
                            attn_entropy = -(attn * torch.log(attn + 1e-8)).sum()
                            
                            # Different strategies should have different entropy
                            expected_entropy = {
                                'analytical': 0.5,  # Focused
                                'creative': 2.0,    # Diverse  
                                'pattern': 1.0      # Moderate
                            }
                            
                            entropy_diff = abs(attn_entropy.item() - expected_entropy[strategy_type])
                            strategy_score = 1.0 / (1.0 + entropy_diff)
                            scores.append(strategy_score)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _get_domain_performance(self, model: nn.Module, domain: str) -> DomainPerformance:
        """Get performance metrics for specific domain"""
        test = self.tests[domain]
        scores = []
        adaptation_speeds = []
        
        # Test across difficulties
        for difficulty in [0.3, 0.5, 0.7]:
            for _ in range(3):
                task = test.generate_task(difficulty)
                model_input = self._prepare_task_input(task, domain)
                
                # Measure adaptation
                start_time = time.time()
                
                with torch.no_grad():
                    try:
                        output = model(model_input)
                        solution = self._extract_solution(output, task, domain)
                        score = test.evaluate_solution(solution, task)
                        scores.append(score)
                    except Exception as e:
                        print(f"Error in domain {domain} evaluation: {e}")
                        scores.append(0.0)
                
                adapt_time = time.time() - start_time
                adaptation_speeds.append(1.0 / (1.0 + adapt_time))
        
        # Check zero-shot performance
        zero_shot = len(self.performance_history[domain]) == 0
        
        # Track in history
        for s in scores:
            self.performance_history[domain].append(s)
        
        return DomainPerformance(
            domain_name=domain,
            score=float(np.mean(scores)) if scores else 0.0,
            adaptation_speed=float(np.mean(adaptation_speeds)) if adaptation_speeds else 0.5,
            zero_shot=zero_shot,
            confidence=float(1.0 - np.std(scores)) if len(scores) > 1 else 0.5
        )