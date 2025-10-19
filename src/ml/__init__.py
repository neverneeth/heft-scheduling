"""
Machine Learning module for HEFT scheduling framework.

This module provides ML-enhanced scheduling algorithms including:
- Dataset generation for imitation learning
- Regression-based task prioritization
- Feature extraction from DAGs and tasks
"""

from .dataset_generator import ExpertDatasetGenerator
from .feature_extractor import TaskFeatureExtractor
from .regression_heft import RegressionHEFT

__all__ = [
    'ExpertDatasetGenerator',
    'TaskFeatureExtractor', 
    'RegressionHEFT'
]