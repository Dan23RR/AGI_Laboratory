#!/usr/bin/env python3
"""
General AGI Test Environments - Domain-Agnostic Challenges
=========================================================
Diverse environments to test true general intelligence without
biasing towards specific tasks or domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from collections import deque, defaultdict


class GeneralEnvironment(ABC):
    """Base class for general AGI test environments"""
    
    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment to initial state"""
        pass
    
    @abstractmethod 
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Execute action and return (observation, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """Get observation dimension"""
        pass
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Get action dimension"""
        pass


class AbstractProblemSolvingEnv(GeneralEnvironment):
    """
    Environment for abstract problem solving without domain-specific knowledge.
    Tests pure reasoning, pattern recognition, and creative problem solving.
    """
    
    def __init__(self, difficulty: float = 0.5, max_steps: int = 100):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.current_step = 0
        
        # Problem parameters scale with difficulty
        self.problem_dim = int(16 + difficulty * 48)
        self.constraint_count = int(3 + difficulty * 7)
        self.solution_complexity = int(2 + difficulty * 5)
        
        # State includes: current configuration, constraints, goal
        self.state_dim = self.problem_dim * 3 + self.constraint_count * 4
        self.action_dim = self.problem_dim + 10  # Modifications + meta-actions
        
        self.reset()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Generate new abstract problem"""
        self.current_step = 0
        
        # Generate problem structure
        self.current_state = torch.randn(self.problem_dim)
        self.goal_state = torch.randn(self.problem_dim)
        
        # Generate constraints (relations that must be maintained)
        self.constraints = []
        for _ in range(self.constraint_count):
            constraint_type = np.random.choice(['distance', 'angle', 'magnitude', 'relation'])
            
            if constraint_type == 'distance':
                # Maintain distance between elements
                indices = np.random.choice(self.problem_dim, 2, replace=False)
                target_dist = np.random.uniform(0.5, 2.0)
                self.constraints.append({
                    'type': 'distance',
                    'indices': indices,
                    'target': target_dist,
                    'weight': np.random.uniform(0.5, 1.5)
                })
            
            elif constraint_type == 'angle':
                # Maintain angle between vectors
                indices = np.random.choice(self.problem_dim, 3, replace=False)
                target_angle = np.random.uniform(0, np.pi)
                self.constraints.append({
                    'type': 'angle',
                    'indices': indices,
                    'target': target_angle,
                    'weight': np.random.uniform(0.5, 1.5)
                })
            
            elif constraint_type == 'magnitude':
                # Maintain magnitude of subset
                n_elements = np.random.randint(2, min(5, self.problem_dim))
                indices = np.random.choice(self.problem_dim, n_elements, replace=False)
                target_mag = np.random.uniform(1.0, 5.0)
                self.constraints.append({
                    'type': 'magnitude',
                    'indices': indices,
                    'target': target_mag,
                    'weight': np.random.uniform(0.5, 1.5)
                })
            
            else:  # relation
                # Maintain relationship between elements
                indices = np.random.choice(self.problem_dim, 4, replace=False)
                relation_matrix = torch.randn(2, 2)
                self.constraints.append({
                    'type': 'relation',
                    'indices': indices,
                    'matrix': relation_matrix,
                    'weight': np.random.uniform(0.5, 1.5)
                })
        
        # Create observation
        return self._create_observation()
    
    def _create_observation(self) -> Dict[str, torch.Tensor]:
        """Create observation from current state"""
        # Encode constraints
        constraint_encoding = []
        for c in self.constraints:
            if c['type'] == 'distance':
                enc = torch.tensor([1, 0, 0, 0, c['target'], c['weight'], 
                                   c['indices'][0], c['indices'][1]])
            elif c['type'] == 'angle':
                enc = torch.tensor([0, 1, 0, 0, c['target'], c['weight'],
                                   c['indices'][0], c['indices'][1]])
            elif c['type'] == 'magnitude':
                enc = torch.tensor([0, 0, 1, 0, c['target'], c['weight'],
                                   len(c['indices']), c['indices'][0]])
            else:  # relation
                enc = torch.tensor([0, 0, 0, 1, c['matrix'].sum().item(), c['weight'],
                                   c['indices'][0], c['indices'][1]])
            constraint_encoding.append(enc)
        
        # Pad constraint encoding
        while len(constraint_encoding) < self.constraint_count:
            constraint_encoding.append(torch.zeros(8))
        
        constraint_tensor = torch.stack(constraint_encoding).flatten()
        
        # Combine all information
        observation = torch.cat([
            self.current_state,
            self.goal_state,
            self.current_state - self.goal_state,  # Difference
            constraint_tensor
        ])
        
        return {
            'observation': observation.unsqueeze(0),
            'current_state': self.current_state.unsqueeze(0),
            'goal_state': self.goal_state.unsqueeze(0),
            'constraints': self.constraints,
            'step': self.current_step
        }
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Apply action to problem state"""
        self.current_step += 1
        action = action.squeeze()
        
        # Parse action
        if action.shape[0] >= self.action_dim:
            action = action[:self.action_dim]
        else:
            # Pad if needed
            action = F.pad(action, (0, self.action_dim - action.shape[0]))
        
        # State modifications
        state_changes = action[:self.problem_dim]
        meta_actions = action[self.problem_dim:]
        
        # Apply meta-actions (e.g., constraint relaxation, transform)
        transform_strength = torch.sigmoid(meta_actions[0])
        constraint_relax = torch.sigmoid(meta_actions[1]) 
        exploration_bonus = torch.sigmoid(meta_actions[2])
        
        # Update state with transformations
        if transform_strength > 0.5:
            # Apply random rotation
            rotation = torch.randn(self.problem_dim, self.problem_dim) * 0.1
            rotation = rotation - rotation.T  # Make antisymmetric
            rotation_matrix = torch.matrix_exp(rotation)
            self.current_state = torch.matmul(rotation_matrix, self.current_state)
        
        # Apply direct state changes
        self.current_state = self.current_state + state_changes * 0.1
        
        # Compute reward
        reward = self._compute_reward(constraint_relax.item(), exploration_bonus.item())
        
        # Check if done
        goal_distance = torch.norm(self.current_state - self.goal_state)
        constraints_satisfied = self._check_constraints()
        
        done = (goal_distance < 0.1 and constraints_satisfied) or self.current_step >= self.max_steps
        
        # Create new observation
        obs = self._create_observation()
        
        info = {
            'goal_distance': goal_distance.item(),
            'constraints_satisfied': constraints_satisfied,
            'constraint_violations': self._get_constraint_violations(),
            'exploration_used': exploration_bonus.item()
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, constraint_relax: float, exploration_bonus: float) -> float:
        """Compute reward based on progress and constraint satisfaction"""
        # Goal achievement
        goal_distance = torch.norm(self.current_state - self.goal_state)
        goal_reward = 1.0 / (1.0 + goal_distance.item())
        
        # Constraint satisfaction
        constraint_reward = 0.0
        total_weight = 0.0
        
        for c in self.constraints:
            weight = c['weight'] * (1.0 - constraint_relax * 0.5)  # Relaxation reduces weight
            violation = self._compute_constraint_violation(c)
            c_reward = weight / (1.0 + violation)
            constraint_reward += c_reward
            total_weight += weight
        
        if total_weight > 0:
            constraint_reward /= total_weight
        
        # Exploration bonus for novel states
        if exploration_bonus > 0.5:
            novelty = torch.std(self.current_state).item()
            exploration_reward = exploration_bonus * novelty * 0.1
        else:
            exploration_reward = 0.0
        
        # Combine rewards
        total_reward = (0.5 * goal_reward + 
                       0.4 * constraint_reward + 
                       0.1 * exploration_reward)
        
        # Penalty for time
        time_penalty = 0.001 * self.current_step
        
        return total_reward - time_penalty
    
    def _compute_constraint_violation(self, constraint: Dict) -> float:
        """Compute violation of a single constraint"""
        c_type = constraint['type']
        indices = constraint['indices']
        
        if c_type == 'distance':
            actual_dist = torch.norm(
                self.current_state[indices[0]] - self.current_state[indices[1]]
            )
            violation = abs(actual_dist.item() - constraint['target'])
            
        elif c_type == 'angle':
            v1 = self.current_state[indices[1]] - self.current_state[indices[0]]
            v2 = self.current_state[indices[2]] - self.current_state[indices[1]]
            
            cos_angle = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            actual_angle = torch.acos(torch.clamp(cos_angle, -1, 1))
            violation = abs(actual_angle.item() - constraint['target'])
            
        elif c_type == 'magnitude':
            subset = self.current_state[list(indices)]
            actual_mag = torch.norm(subset)
            violation = abs(actual_mag.item() - constraint['target'])
            
        else:  # relation
            # Check if relation matrix is satisfied
            subset = self.current_state[list(indices)].view(2, 2)
            expected = torch.matmul(constraint['matrix'], subset)
            violation = torch.norm(expected - subset).item()
        
        return violation
    
    def _check_constraints(self) -> bool:
        """Check if all constraints are satisfied"""
        for c in self.constraints:
            if self._compute_constraint_violation(c) > 0.1:
                return False
        return True
    
    def _get_constraint_violations(self) -> List[float]:
        """Get list of all constraint violations"""
        return [self._compute_constraint_violation(c) for c in self.constraints]
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim


class DynamicConceptLearningEnv(GeneralEnvironment):
    """
    Environment for learning and manipulating abstract concepts that
    change over time. Tests concept formation, tracking, and application.
    """
    
    def __init__(self, difficulty: float = 0.5, max_steps: int = 200):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.current_step = 0
        
        # Concept parameters
        self.n_concepts = int(3 + difficulty * 7)
        self.concept_dim = int(8 + difficulty * 24)
        self.n_instances = int(10 + difficulty * 20)
        self.evolution_rate = 0.01 + difficulty * 0.04
        
        # State: current instance + concept history + context
        self.state_dim = self.concept_dim * 3 + self.n_concepts * 4 + 10
        # Actions: concept selection + transformation + creation
        self.action_dim = self.n_concepts + self.concept_dim + 5
        
        self.reset()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Initialize new concept learning episode"""
        self.current_step = 0
        
        # Generate initial concepts (prototypes)
        self.concepts = {}
        self.concept_evolution = defaultdict(list)
        
        for i in range(self.n_concepts):
            # Each concept has a prototype and variance
            prototype = torch.randn(self.concept_dim)
            variance = torch.abs(torch.randn(self.concept_dim)) * 0.5 + 0.1
            
            self.concepts[i] = {
                'prototype': prototype,
                'variance': variance,
                'instances': [],
                'creation_time': 0,
                'last_accessed': 0
            }
            
            # Generate initial instances
            for _ in range(np.random.randint(2, 6)):
                instance = prototype + torch.randn(self.concept_dim) * variance
                self.concepts[i]['instances'].append(instance)
        
        # Hidden concept relationships
        self.concept_relations = torch.randn(self.n_concepts, self.n_concepts) * 0.3
        self.concept_relations = (self.concept_relations + self.concept_relations.T) / 2
        
        # Current context
        self.current_instance = self._generate_instance()
        self.target_concept = np.random.randint(self.n_concepts)
        
        return self._create_observation()
    
    def _generate_instance(self) -> torch.Tensor:
        """Generate new instance that may belong to a concept or be novel"""
        if np.random.random() < 0.7:  # Known concept
            concept_id = np.random.randint(self.n_concepts)
            concept = self.concepts[concept_id]
            instance = concept['prototype'] + torch.randn(self.concept_dim) * concept['variance']
            self.true_concept = concept_id
        else:  # Novel combination or outlier
            # Blend multiple concepts
            n_blend = min(3, self.n_concepts)
            blend_ids = np.random.choice(self.n_concepts, n_blend, replace=False)
            weights = F.softmax(torch.randn(n_blend), dim=0)
            
            instance = torch.zeros(self.concept_dim)
            for i, c_id in enumerate(blend_ids):
                instance += weights[i] * self.concepts[c_id]['prototype']
            
            instance += torch.randn(self.concept_dim) * 0.2
            self.true_concept = -1  # Novel
        
        return instance
    
    def _create_observation(self) -> Dict[str, torch.Tensor]:
        """Create observation encoding current state"""
        # Encode concept statistics
        concept_stats = []
        for i in range(self.n_concepts):
            c = self.concepts[i]
            stats = torch.tensor([
                len(c['instances']),
                c['prototype'].norm().item(),
                c['variance'].mean().item(),
                self.current_step - c['last_accessed']
            ])
            concept_stats.append(stats)
        
        concept_tensor = torch.stack(concept_stats).flatten()
        
        # Recent concept evolution
        recent_evolution = torch.zeros(self.concept_dim)
        for i in range(self.n_concepts):
            if len(self.concept_evolution[i]) > 0:
                recent = self.concept_evolution[i][-1]
                recent_evolution += recent / self.n_concepts
        
        # Context encoding
        context = torch.tensor([
            self.current_step / self.max_steps,
            len(self.concept_evolution),
            self.evolution_rate,
            float(self.true_concept) / self.n_concepts if self.true_concept >= 0 else -1,
            float(self.target_concept) / self.n_concepts,
        ])
        
        # Pad context to fixed size
        context = F.pad(context, (0, 10 - context.shape[0]))
        
        # Combine observation
        observation = torch.cat([
            self.current_instance,
            recent_evolution,
            self._compute_instance_similarities(),
            concept_tensor,
            context
        ])
        
        return {
            'observation': observation.unsqueeze(0),
            'current_instance': self.current_instance.unsqueeze(0),
            'concepts': self.concepts,
            'step': self.current_step
        }
    
    def _compute_instance_similarities(self) -> torch.Tensor:
        """Compute similarities between current instance and all concepts"""
        similarities = []
        
        for i in range(self.n_concepts):
            c = self.concepts[i]
            # Mahalanobis distance
            diff = self.current_instance - c['prototype']
            weighted_diff = diff / (c['variance'] + 1e-6)
            distance = torch.norm(weighted_diff)
            similarity = 1.0 / (1.0 + distance)
            similarities.append(similarity)
        
        return torch.tensor(similarities)
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Execute concept learning action"""
        self.current_step += 1
        action = action.squeeze()
        
        # Parse action
        concept_probs = F.softmax(action[:self.n_concepts], dim=0)
        selected_concept = torch.argmax(concept_probs).item()
        
        transformation = action[self.n_concepts:self.n_concepts + self.concept_dim]
        meta_actions = action[-5:]
        
        # Meta-action flags
        create_new = torch.sigmoid(meta_actions[0]) > 0.7
        merge_concepts = torch.sigmoid(meta_actions[1]) > 0.7
        split_concept = torch.sigmoid(meta_actions[2]) > 0.7
        forget_old = torch.sigmoid(meta_actions[3]) > 0.7
        transform_strength = torch.sigmoid(meta_actions[4])
        
        # Execute actions
        reward = 0.0
        
        # Primary action: classify current instance
        if self.true_concept >= 0:
            if selected_concept == self.true_concept:
                reward += 1.0
                # Add to concept instances
                self.concepts[selected_concept]['instances'].append(self.current_instance)
                self.concepts[selected_concept]['last_accessed'] = self.current_step
            else:
                reward -= 0.5
        
        # Meta-actions
        if create_new and self.n_concepts < 20:
            # Create new concept from current instance
            self._create_new_concept(self.current_instance)
            reward += 0.1
        
        if merge_concepts and self.n_concepts > 2:
            # Merge two similar concepts
            self._merge_similar_concepts()
            reward += 0.05
        
        if split_concept and selected_concept < self.n_concepts:
            # Split selected concept if it has high variance
            if self._split_concept(selected_concept):
                reward += 0.05
        
        if forget_old:
            # Forget least used concept
            self._forget_unused_concept()
        
        # Apply transformation to selected concept
        if selected_concept < self.n_concepts:
            self._transform_concept(selected_concept, transformation, transform_strength.item())
        
        # Evolve all concepts
        self._evolve_concepts()
        
        # Generate new instance
        self.current_instance = self._generate_instance()
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Compute final classification accuracy
        if done:
            accuracy = self._compute_final_accuracy()
            reward += accuracy * 5.0
        
        obs = self._create_observation()
        
        info = {
            'n_concepts': self.n_concepts,
            'selected_concept': selected_concept,
            'true_concept': self.true_concept,
            'concept_similarities': self._compute_instance_similarities().tolist(),
            'concepts_evolved': len(self.concept_evolution)
        }
        
        return obs, reward, done, info
    
    def _create_new_concept(self, instance: torch.Tensor):
        """Create new concept from instance"""
        new_id = max(self.concepts.keys()) + 1 if self.concepts else 0
        
        self.concepts[new_id] = {
            'prototype': instance.clone(),
            'variance': torch.ones(self.concept_dim) * 0.3,
            'instances': [instance],
            'creation_time': self.current_step,
            'last_accessed': self.current_step
        }
        
        self.n_concepts += 1
        
        # Update relations
        new_relations = torch.randn(self.n_concepts, 1) * 0.3
        self.concept_relations = F.pad(self.concept_relations, (0, 1, 0, 1))
        self.concept_relations[-1, :-1] = new_relations.squeeze()
        self.concept_relations[:-1, -1] = new_relations.squeeze()
    
    def _merge_similar_concepts(self):
        """Merge two most similar concepts"""
        if self.n_concepts <= 2:
            return
        
        # Find most similar pair
        max_sim = -float('inf')
        merge_pair = None
        
        for i in range(self.n_concepts):
            for j in range(i+1, self.n_concepts):
                if i in self.concepts and j in self.concepts:
                    sim = F.cosine_similarity(
                        self.concepts[i]['prototype'].unsqueeze(0),
                        self.concepts[j]['prototype'].unsqueeze(0)
                    ).item()
                    
                    if sim > max_sim:
                        max_sim = sim
                        merge_pair = (i, j)
        
        if merge_pair and max_sim > 0.8:
            i, j = merge_pair
            # Merge j into i
            c_i = self.concepts[i]
            c_j = self.concepts[j]
            
            # Weighted average of prototypes
            n_i = len(c_i['instances'])
            n_j = len(c_j['instances'])
            total = n_i + n_j
            
            if total > 0:
                c_i['prototype'] = (n_i * c_i['prototype'] + n_j * c_j['prototype']) / total
                c_i['variance'] = (c_i['variance'] + c_j['variance']) / 2
                c_i['instances'].extend(c_j['instances'])
            
            # Remove j
            del self.concepts[j]
            self.n_concepts -= 1
    
    def _split_concept(self, concept_id: int) -> bool:
        """Split concept if it has high internal variance"""
        if concept_id not in self.concepts:
            return False
        
        c = self.concepts[concept_id]
        if len(c['instances']) < 4:
            return False
        
        # Check variance
        instances = torch.stack(c['instances'])
        total_var = torch.var(instances, dim=0).mean()
        
        if total_var > 1.0:
            # K-means split
            # Simple 2-means
            center1 = instances[0]
            center2 = instances[-1]
            
            for _ in range(5):
                # Assign to clusters
                dist1 = torch.norm(instances - center1, dim=1)
                dist2 = torch.norm(instances - center2, dim=1)
                
                cluster1 = instances[dist1 < dist2]
                cluster2 = instances[dist1 >= dist2]
                
                if len(cluster1) > 0 and len(cluster2) > 0:
                    center1 = cluster1.mean(dim=0)
                    center2 = cluster2.mean(dim=0)
            
            # Create new concept
            if len(cluster2) > 0:
                new_id = max(self.concepts.keys()) + 1
                self.concepts[new_id] = {
                    'prototype': center2,
                    'variance': torch.var(cluster2, dim=0),
                    'instances': cluster2.tolist(),
                    'creation_time': self.current_step,
                    'last_accessed': self.current_step
                }
                
                # Update original
                c['prototype'] = center1
                c['variance'] = torch.var(cluster1, dim=0) if len(cluster1) > 0 else c['variance']
                c['instances'] = cluster1.tolist() if len(cluster1) > 0 else []
                
                self.n_concepts += 1
                return True
        
        return False
    
    def _forget_unused_concept(self):
        """Remove least recently used concept"""
        if self.n_concepts <= 2:
            return
        
        # Find LRU concept
        lru_id = None
        lru_time = float('inf')
        
        for c_id, c in self.concepts.items():
            if c['last_accessed'] < lru_time:
                lru_time = c['last_accessed']
                lru_id = c_id
        
        if lru_id is not None and self.current_step - lru_time > 20:
            del self.concepts[lru_id]
            self.n_concepts -= 1
    
    def _transform_concept(self, concept_id: int, transformation: torch.Tensor, strength: float):
        """Apply transformation to concept"""
        if concept_id not in self.concepts:
            return
        
        c = self.concepts[concept_id]
        
        # Apply transformation with strength
        c['prototype'] = c['prototype'] + transformation * strength * 0.1
        
        # Record evolution
        self.concept_evolution[concept_id].append(transformation * strength * 0.1)
        
        # Limit history
        if len(self.concept_evolution[concept_id]) > 10:
            self.concept_evolution[concept_id].pop(0)
    
    def _evolve_concepts(self):
        """Natural evolution of all concepts"""
        for c_id, c in self.concepts.items():
            # Drift based on instances
            if len(c['instances']) > 0:
                recent_instances = c['instances'][-5:]
                recent_mean = torch.stack(recent_instances).mean(dim=0)
                
                # Move prototype towards recent instances
                c['prototype'] = (1 - self.evolution_rate) * c['prototype'] + self.evolution_rate * recent_mean
            
            # Variance adaptation
            if len(c['instances']) > 2:
                instances = torch.stack(c['instances'])
                new_variance = torch.var(instances, dim=0)
                c['variance'] = (1 - self.evolution_rate) * c['variance'] + self.evolution_rate * new_variance
    
    def _compute_final_accuracy(self) -> float:
        """Compute classification accuracy over all instances"""
        correct = 0
        total = 0
        
        for c_id, c in self.concepts.items():
            for instance in c['instances']:
                # Find closest concept
                min_dist = float('inf')
                predicted = -1
                
                for other_id, other_c in self.concepts.items():
                    dist = torch.norm(instance - other_c['prototype'])
                    if dist < min_dist:
                        min_dist = dist
                        predicted = other_id
                
                if predicted == c_id:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim


class MultiAgentCoordinationEnv(GeneralEnvironment):
    """
    Environment for learning coordination without explicit communication.
    Tests emergent coordination, theory of mind, and collective intelligence.
    """
    
    def __init__(self, difficulty: float = 0.5, max_steps: int = 150):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.current_step = 0
        
        # Multi-agent parameters
        self.n_agents = int(3 + difficulty * 5)
        self.world_size = int(10 + difficulty * 20)
        self.n_resources = int(5 + difficulty * 10)
        self.n_goals = int(2 + difficulty * 4)
        
        # State: all agent positions + resources + goals + history
        self.state_dim = (self.n_agents * 4 +  # Position + velocity
                         self.n_resources * 3 +  # Resource position + value
                         self.n_goals * 3 +  # Goal position + progress
                         20)  # Global info
        
        # Action: movement + resource interaction + signaling
        self.action_dim = 4 + 3 + 5  # Move + interact + signals
        
        self.reset()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Initialize multi-agent scenario"""
        self.current_step = 0
        
        # Agent positions and velocities
        self.agent_positions = torch.rand(self.n_agents, 2) * self.world_size
        self.agent_velocities = torch.zeros(self.n_agents, 2)
        self.agent_resources = torch.zeros(self.n_agents)
        
        # Resources
        self.resources = []
        for _ in range(self.n_resources):
            self.resources.append({
                'position': torch.rand(2) * self.world_size,
                'value': torch.rand(1).item() * (1 + self.difficulty),
                'respawn_time': 0,
                'collected': False
            })
        
        # Goals requiring coordination
        self.goals = []
        for i in range(self.n_goals):
            goal_type = np.random.choice(['collective', 'sequential', 'simultaneous'])
            
            if goal_type == 'collective':
                # All agents must be in region
                center = torch.rand(2) * self.world_size
                radius = 2.0 + self.difficulty * 3.0
                required_agents = min(self.n_agents, 2 + i)
                
                self.goals.append({
                    'type': 'collective',
                    'center': center,
                    'radius': radius,
                    'required_agents': required_agents,
                    'progress': 0.0,
                    'completed': False
                })
            
            elif goal_type == 'sequential':
                # Agents must visit in order
                waypoints = [torch.rand(2) * self.world_size for _ in range(3)]
                
                self.goals.append({
                    'type': 'sequential',
                    'waypoints': waypoints,
                    'current_waypoint': 0,
                    'visited_by': set(),
                    'progress': 0.0,
                    'completed': False
                })
            
            else:  # simultaneous
                # Multiple locations activated together
                locations = [torch.rand(2) * self.world_size for _ in range(2)]
                
                self.goals.append({
                    'type': 'simultaneous',
                    'locations': locations,
                    'activation_radius': 1.5,
                    'time_window': 10,
                    'last_activation': -100,
                    'progress': 0.0,
                    'completed': False
                })
        
        # Communication/signaling state
        self.signals = torch.zeros(self.n_agents, 5)
        self.signal_history = deque(maxlen=10)
        
        return self._create_observation()
    
    def _create_observation(self) -> Dict[str, torch.Tensor]:
        """Create observation for single agent perspective"""
        # For each agent, create observation from their perspective
        observations = []
        
        for agent_id in range(self.n_agents):
            # Own state
            own_pos = self.agent_positions[agent_id]
            own_vel = self.agent_velocities[agent_id]
            own_resources = self.agent_resources[agent_id]
            
            # Other agents (relative positions)
            other_agents = []
            for i in range(self.n_agents):
                if i != agent_id:
                    rel_pos = self.agent_positions[i] - own_pos
                    rel_vel = self.agent_velocities[i] - own_vel
                    other_agents.append(torch.cat([rel_pos, rel_vel]))
            
            # Pad if needed
            while len(other_agents) < self.n_agents - 1:
                other_agents.append(torch.zeros(4))
            
            other_tensor = torch.stack(other_agents).flatten()
            
            # Resources (nearest k)
            resource_info = []
            for r in self.resources:
                if not r['collected']:
                    dist = torch.norm(r['position'] - own_pos)
                    resource_info.append((dist, r))
            
            resource_info.sort(key=lambda x: x[0])
            resource_tensor = []
            
            for i in range(min(5, len(resource_info))):
                _, r = resource_info[i]
                rel_pos = r['position'] - own_pos
                value = torch.tensor([r['value']])
                resource_tensor.append(torch.cat([rel_pos, value]))
            
            while len(resource_tensor) < 5:
                resource_tensor.append(torch.zeros(3))
            
            resource_tensor = torch.stack(resource_tensor).flatten()
            
            # Goals
            goal_tensor = []
            for g in self.goals:
                if g['type'] == 'collective':
                    rel_center = g['center'] - own_pos
                    progress = torch.tensor([g['progress'], g['radius'], float(g['required_agents'])])
                    goal_tensor.append(torch.cat([rel_center, progress]))
                    
                elif g['type'] == 'sequential':
                    current_wp = g['waypoints'][g['current_waypoint']]
                    rel_wp = current_wp - own_pos
                    progress = torch.tensor([g['progress'], g['current_waypoint'], len(g['visited_by'])])
                    goal_tensor.append(torch.cat([rel_wp, progress]))
                    
                else:  # simultaneous
                    # Nearest location
                    dists = [torch.norm(loc - own_pos) for loc in g['locations']]
                    nearest_idx = np.argmin(dists)
                    rel_loc = g['locations'][nearest_idx] - own_pos
                    time_since = self.current_step - g['last_activation']
                    progress = torch.tensor([g['progress'], time_since / g['time_window'], 0])
                    goal_tensor.append(torch.cat([rel_loc, progress]))
            
            goal_tensor = torch.stack(goal_tensor).flatten() if goal_tensor else torch.zeros(self.n_goals * 5)
            
            # Global information
            global_info = torch.tensor([
                self.current_step / self.max_steps,
                sum(g['completed'] for g in self.goals) / self.n_goals,
                self.agent_resources.mean().item(),
                self.agent_resources.std().item(),
                torch.norm(self.agent_positions.std(dim=0)).item(),  # Spread
            ])
            
            # Recent signals
            if self.signal_history:
                recent_signals = torch.stack(list(self.signal_history)[-3:]).flatten()
            else:
                recent_signals = torch.zeros(15)
            
            # Combine observation
            obs = torch.cat([
                own_pos,
                own_vel,
                torch.tensor([own_resources]),
                other_tensor,
                resource_tensor,
                goal_tensor,
                global_info,
                recent_signals
            ])
            
            observations.append(obs)
        
        # Return observation for first agent (in full system, each agent would act)
        return {
            'observation': observations[0].unsqueeze(0),
            'all_observations': torch.stack(observations),
            'agent_positions': self.agent_positions,
            'goals': self.goals,
            'step': self.current_step
        }
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Execute coordinated actions"""
        self.current_step += 1
        
        # For simplicity, action controls first agent, others use simple policy
        # In full implementation, all agents would be controlled
        action = action.squeeze()
        
        # Parse action
        movement = action[:4]  # dx, dy, speed, turn
        interaction = action[4:7]  # collect, share, activate
        signals = action[7:12] if action.shape[0] >= 12 else torch.zeros(5)
        
        # Update first agent
        agent_id = 0
        
        # Movement
        speed = torch.sigmoid(movement[2]) * 2.0
        turn = torch.tanh(movement[3]) * 0.5
        
        # Update velocity
        current_angle = torch.atan2(self.agent_velocities[agent_id, 1], 
                                   self.agent_velocities[agent_id, 0])
        new_angle = current_angle + turn
        
        self.agent_velocities[agent_id, 0] = speed * torch.cos(new_angle)
        self.agent_velocities[agent_id, 1] = speed * torch.sin(new_angle)
        
        # Update position
        self.agent_positions[agent_id] += self.agent_velocities[agent_id] * 0.1
        
        # Boundary
        self.agent_positions[agent_id] = torch.clamp(self.agent_positions[agent_id], 0, self.world_size)
        
        # Simple policy for other agents
        for i in range(1, self.n_agents):
            # Move towards nearest goal or resource
            target = self._get_agent_target(i)
            if target is not None:
                direction = target - self.agent_positions[i]
                direction = direction / (torch.norm(direction) + 1e-6)
                self.agent_velocities[i] = direction * 1.5
                self.agent_positions[i] += self.agent_velocities[i] * 0.1
                self.agent_positions[i] = torch.clamp(self.agent_positions[i], 0, self.world_size)
        
        # Process interactions
        reward = 0.0
        
        # Resource collection
        if torch.sigmoid(interaction[0]) > 0.5:
            collected_value = self._collect_resources(agent_id)
            reward += collected_value
        
        # Resource sharing
        if torch.sigmoid(interaction[1]) > 0.5:
            shared = self._share_resources(agent_id)
            reward += shared * 0.5  # Cooperation bonus
        
        # Goal activation
        if torch.sigmoid(interaction[2]) > 0.5:
            goal_reward = self._try_activate_goals()
            reward += goal_reward
        
        # Update signals
        self.signals[agent_id] = torch.sigmoid(signals)
        self.signal_history.append(self.signals.clone())
        
        # Check goal completion
        goals_completed = sum(g['completed'] for g in self.goals)
        
        # Coordination bonus
        coord_bonus = self._compute_coordination_bonus()
        reward += coord_bonus
        
        # Done condition
        all_goals_done = all(g['completed'] for g in self.goals)
        done = all_goals_done or self.current_step >= self.max_steps
        
        if done and all_goals_done:
            # Big bonus for completing all goals
            reward += 10.0 * (1.0 - self.current_step / self.max_steps)
        
        obs = self._create_observation()
        
        info = {
            'goals_completed': goals_completed,
            'total_resources': self.agent_resources.sum().item(),
            'coordination_score': coord_bonus,
            'agent_distances': self._compute_agent_distances()
        }
        
        return obs, reward, done, info
    
    def _get_agent_target(self, agent_id: int) -> Optional[torch.Tensor]:
        """Get target position for agent's simple policy"""
        # Find nearest incomplete goal
        best_target = None
        best_distance = float('inf')
        
        for g in self.goals:
            if not g['completed']:
                if g['type'] == 'collective':
                    target = g['center']
                elif g['type'] == 'sequential':
                    target = g['waypoints'][g['current_waypoint']]
                else:
                    target = g['locations'][0]
                
                dist = torch.norm(target - self.agent_positions[agent_id])
                if dist < best_distance:
                    best_distance = dist
                    best_target = target
        
        # If no goals, go to nearest resource
        if best_target is None:
            for r in self.resources:
                if not r['collected']:
                    dist = torch.norm(r['position'] - self.agent_positions[agent_id])
                    if dist < best_distance:
                        best_distance = dist
                        best_target = r['position']
        
        return best_target
    
    def _collect_resources(self, agent_id: int) -> float:
        """Collect nearby resources"""
        collected_value = 0.0
        agent_pos = self.agent_positions[agent_id]
        
        for r in self.resources:
            if not r['collected']:
                dist = torch.norm(r['position'] - agent_pos)
                if dist < 1.0:  # Collection radius
                    self.agent_resources[agent_id] += r['value']
                    collected_value += r['value']
                    r['collected'] = True
                    r['respawn_time'] = self.current_step + 20
        
        return collected_value
    
    def _share_resources(self, agent_id: int) -> float:
        """Share resources with nearby agents"""
        shared = 0.0
        agent_pos = self.agent_positions[agent_id]
        
        for i in range(self.n_agents):
            if i != agent_id:
                dist = torch.norm(self.agent_positions[i] - agent_pos)
                if dist < 2.0:  # Sharing radius
                    # Transfer half of difference
                    diff = self.agent_resources[agent_id] - self.agent_resources[i]
                    if diff > 0:
                        transfer = diff * 0.5
                        self.agent_resources[agent_id] -= transfer
                        self.agent_resources[i] += transfer
                        shared += transfer
        
        return shared
    
    def _try_activate_goals(self) -> float:
        """Try to activate/progress goals based on agent positions"""
        total_reward = 0.0
        
        for g in self.goals:
            if g['completed']:
                continue
            
            if g['type'] == 'collective':
                # Count agents in region
                center = g['center']
                radius = g['radius']
                
                agents_in_region = 0
                for pos in self.agent_positions:
                    if torch.norm(pos - center) <= radius:
                        agents_in_region += 1
                
                if agents_in_region >= g['required_agents']:
                    g['progress'] = min(1.0, g['progress'] + 0.1)
                    total_reward += 0.5
                    
                    if g['progress'] >= 1.0:
                        g['completed'] = True
                        total_reward += 5.0
                else:
                    g['progress'] = max(0.0, g['progress'] - 0.05)
            
            elif g['type'] == 'sequential':
                # Check if agent at current waypoint
                current_wp = g['waypoints'][g['current_waypoint']]
                
                for i, pos in enumerate(self.agent_positions):
                    if torch.norm(pos - current_wp) < 1.5:
                        if i not in g['visited_by'] or len(g['visited_by']) == 0:
                            g['visited_by'].add(i)
                            g['progress'] += 0.3
                            total_reward += 1.0
                            
                            # Move to next waypoint
                            if g['current_waypoint'] < len(g['waypoints']) - 1:
                                g['current_waypoint'] += 1
                                g['visited_by'].clear()
                            else:
                                g['completed'] = True
                                total_reward += 5.0
            
            else:  # simultaneous
                # Check if agents at all locations
                agents_at_locations = []
                
                for loc in g['locations']:
                    agent_present = False
                    for pos in self.agent_positions:
                        if torch.norm(pos - loc) < g['activation_radius']:
                            agent_present = True
                            break
                    agents_at_locations.append(agent_present)
                
                if all(agents_at_locations):
                    if self.current_step - g['last_activation'] <= g['time_window']:
                        g['progress'] = min(1.0, g['progress'] + 0.2)
                        total_reward += 1.0
                        
                        if g['progress'] >= 1.0:
                            g['completed'] = True
                            total_reward += 5.0
                    else:
                        g['last_activation'] = self.current_step
                        g['progress'] = 0.2
                        total_reward += 0.2
        
        return total_reward
    
    def _compute_coordination_bonus(self) -> float:
        """Compute bonus for coordinated behavior"""
        bonus = 0.0
        
        # Proximity bonus (agents working together)
        distances = self._compute_agent_distances()
        avg_distance = distances.mean()
        
        # Optimal distance: not too close, not too far
        optimal_dist = 3.0
        proximity_score = 1.0 / (1.0 + abs(avg_distance - optimal_dist))
        bonus += proximity_score * 0.1
        
        # Signal coordination (similar signals = coordination)
        if self.signal_history:
            recent = torch.stack(list(self.signal_history)[-3:])
            signal_variance = recent.var(dim=0).mean()
            signal_coordination = 1.0 / (1.0 + signal_variance)
            bonus += signal_coordination * 0.1
        
        # Resource balance (sharing)
        resource_variance = self.agent_resources.var()
        balance_score = 1.0 / (1.0 + resource_variance)
        bonus += balance_score * 0.05
        
        return bonus
    
    def _compute_agent_distances(self) -> torch.Tensor:
        """Compute pairwise distances between agents"""
        n = self.n_agents
        distances = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(self.agent_positions[i] - self.agent_positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim


class EnvironmentSuite:
    """
    Suite of diverse environments for testing general intelligence.
    Provides unified interface and cross-environment learning.
    """
    
    def __init__(self):
        self.environments = {
            'abstract_problem': AbstractProblemSolvingEnv,
            'concept_learning': DynamicConceptLearningEnv,
            'multi_agent': MultiAgentCoordinationEnv
        }
        
        self.current_env = None
        self.env_name = None
        self.performance_history = defaultdict(list)
        
    def create_environment(self, env_type: str, difficulty: float = 0.5) -> GeneralEnvironment:
        """Create environment of specified type"""
        if env_type not in self.environments:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        env_class = self.environments[env_type]
        env = env_class(difficulty=difficulty)
        
        self.current_env = env
        self.env_name = env_type
        
        return env
    
    def create_curriculum(self, difficulties: List[float]) -> List[Tuple[str, GeneralEnvironment]]:
        """Create curriculum of environments with increasing difficulty"""
        curriculum = []
        
        for difficulty in difficulties:
            # Rotate through environment types
            for env_type in self.environments:
                env = self.create_environment(env_type, difficulty)
                curriculum.append((env_type, env))
        
        return curriculum
    
    def create_mixed_batch(self, batch_size: int, difficulty_range: Tuple[float, float] = (0.2, 0.8)) -> List[Tuple[str, GeneralEnvironment]]:
        """Create batch of mixed environments"""
        batch = []
        
        for _ in range(batch_size):
            # Random environment type
            env_type = np.random.choice(list(self.environments.keys()))
            
            # Random difficulty
            difficulty = np.random.uniform(*difficulty_range)
            
            env = self.create_environment(env_type, difficulty)
            batch.append((env_type, env))
        
        return batch
    
    def evaluate_transfer(self, model: nn.Module, source_env: str, target_env: str,
                         n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate transfer learning between environments"""
        # Train on source
        source = self.create_environment(source_env, 0.5)
        source_scores = []
        
        for _ in range(n_episodes):
            obs = source.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                with torch.no_grad():
                    action = model(obs['observation'])
                    
                obs, reward, done, _ = source.step(action)
                episode_reward += reward
            
            source_scores.append(episode_reward)
        
        # Test on target (zero-shot)
        target = self.create_environment(target_env, 0.5)
        target_scores = []
        
        for _ in range(n_episodes):
            obs = target.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                with torch.no_grad():
                    action = model(obs['observation'])
                    
                obs, reward, done, _ = target.step(action)
                episode_reward += reward
            
            target_scores.append(episode_reward)
        
        # Compute transfer metrics
        source_performance = np.mean(source_scores)
        target_performance = np.mean(target_scores)
        
        # Normalized transfer (accounting for environment difficulty)
        baseline_performance = 0.0  # Random agent baseline
        transfer_efficiency = (target_performance - baseline_performance) / (source_performance - baseline_performance + 1e-6)
        
        return {
            'source_performance': source_performance,
            'target_performance': target_performance, 
            'transfer_efficiency': transfer_efficiency,
            'source_env': source_env,
            'target_env': target_env
        }
    
    def create_meta_environment(self) -> 'MetaEnvironment':
        """Create meta-environment that switches between tasks"""
        return MetaEnvironment(self)


class MetaEnvironment(GeneralEnvironment):
    """
    Meta-environment that dynamically switches between different environments.
    Tests rapid adaptation and meta-learning capabilities.
    """
    
    def __init__(self, env_suite: EnvironmentSuite, switch_frequency: int = 50):
        self.env_suite = env_suite
        self.switch_frequency = switch_frequency
        self.step_count = 0
        self.episode_count = 0
        
        # Start with random environment
        self._switch_environment()
        
    def _switch_environment(self):
        """Switch to new random environment"""
        env_type = np.random.choice(list(self.env_suite.environments.keys()))
        difficulty = 0.3 + (self.episode_count / 100.0) * 0.5  # Increase difficulty over time
        difficulty = min(0.9, difficulty)
        
        self.current_env = self.env_suite.create_environment(env_type, difficulty)
        self.current_env_type = env_type
        self.episode_count += 1
        
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset current environment"""
        self.step_count = 0
        obs = self.current_env.reset()
        
        # Add meta-information
        obs['meta_info'] = torch.tensor([
            self.episode_count,
            self.step_count,
            list(self.env_suite.environments.keys()).index(self.current_env_type),
            self.current_env.difficulty
        ])
        
        return obs
    
    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """Step in current environment, potentially switching"""
        obs, reward, done, info = self.current_env.step(action)
        self.step_count += 1
        
        # Check if should switch
        if done or self.step_count >= self.switch_frequency:
            self._switch_environment()
            obs = self.reset()
            info['environment_switched'] = True
            info['new_environment'] = self.current_env_type
        
        # Add meta-information
        obs['meta_info'] = torch.tensor([
            self.episode_count,
            self.step_count,
            list(self.env_suite.environments.keys()).index(self.current_env_type),
            self.current_env.difficulty
        ])
        
        return obs, reward, done, info
    
    def get_state_dim(self) -> int:
        # Maximum across all environments + meta info
        max_dim = max(env().get_state_dim() for env in self.env_suite.environments.values())
        return max_dim + 4  # +4 for meta info
    
    def get_action_dim(self) -> int:
        # Maximum across all environments
        return max(env().get_action_dim() for env in self.env_suite.environments.values())