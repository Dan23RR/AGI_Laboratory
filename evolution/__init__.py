"""
Evolution System for AGI
========================

Contains the genetic algorithm infrastructure for evolving AGI architectures.
"""

from .general_evolution_lab_v3 import GeneralEvolutionLabV3
from .extended_genome import ExtendedGenome
from .mind_factory_v2 import MindFactoryV2, MindConfig

__all__ = [
    'GeneralEvolutionLabV3',
    'ExtendedGenome', 
    'MindFactoryV2',
    'MindConfig'
]