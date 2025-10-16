"""
Base classes and interfaces for scheduling algorithms.

This module defines the abstract base class that all scheduling algorithms
must implement, ensuring a consistent interface across different strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult


class SchedulingAlgorithm(ABC):
    """
    Abstract base class for all scheduling algorithms.
    
    All scheduling algorithms in the framework must inherit from this class
    and implement the schedule() method. This ensures a consistent interface
    for algorithm execution and comparison.
    
    Attributes:
        name (str): Human-readable name of the algorithm
        config (Dict[str, Any]): Algorithm-specific configuration parameters
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize a scheduling algorithm.
        
        Args:
            name: Name identifier for the algorithm
            config: Optional dictionary of algorithm-specific parameters
        """
        self.name = name
        self.config = config if config is not None else {}
    
    @abstractmethod
    def schedule(self, dag: WorkflowDAG) -> ScheduleResult:
        """
        Schedule the workflow DAG onto available processors.
        
        This is the main method that must be implemented by all scheduling
        algorithms. It takes a workflow DAG and produces a schedule that
        assigns tasks to processors with specific timing.
        
        Args:
            dag: The workflow DAG to schedule
            
        Returns:
            ScheduleResult containing the complete schedule and performance metrics
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this algorithm.
        
        Returns:
            Algorithm name
        """
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of this algorithm.
        
        Returns:
            Dictionary of configuration parameters
        """
        return self.config.copy()
    
    def set_config(self, key: str, value: Any):
        """
        Set a configuration parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self.config[key] = value
    
    def __repr__(self) -> str:
        """String representation of the algorithm."""
        return f"{self.__class__.__name__}(name='{self.name}')"
