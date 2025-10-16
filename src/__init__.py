"""
HEFT Scheduling Framework

A modular and extensible framework for designing, implementing, and evaluating
scheduling algorithms for directed acyclic graph (DAG) based workloads in
distributed computing systems.

Usage:
    from src.utils import quick_sanity_check
    
    # Run a quick sanity check
    results = quick_sanity_check(num_tasks=10, num_processors=3)
"""

__version__ = "1.0.0"
__author__ = "HEFT Scheduling Framework Team"

from .core import WorkflowDAG, SystemModel, ScheduleResult
from .algorithms import (
    SchedulingAlgorithm,
    HEFTAlgorithm,
    QLHEFTLargeState,
    QLHEFTSmallState
)
from .utils import (
    DAGGenerator,
    Visualizer,
    SanityChecker,
    quick_sanity_check
)

__all__ = [
    # Core
    'WorkflowDAG',
    'SystemModel',
    'ScheduleResult',
    
    # Algorithms
    'SchedulingAlgorithm',
    'HEFTAlgorithm',
    'QLHEFTLargeState',
    'QLHEFTSmallState',
    
    # Utilities
    'DAGGenerator',
    'Visualizer',
    'SanityChecker',
    'quick_sanity_check',
]
