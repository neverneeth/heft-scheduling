"""
SystemModel: Represents the computing environment and its characteristics.

This module defines the system model including processor specifications
and communication infrastructure.
"""

from typing import List, Optional


class SystemModel:
    """
    Represents the computing system with multiple processors.
    
    This class encapsulates system-level parameters such as the number
    of processors and their characteristics. It can be extended to model
    heterogeneous systems with varying processor speeds and communication
    bandwidth.
    
    Attributes:
        num_processors (int): Number of processors in the system
        processor_names (List[str]): Names/identifiers for each processor
    """
    
    def __init__(self, num_processors: int, processor_names: Optional[List[str]] = None):
        """
        Initialize a SystemModel.
        
        Args:
            num_processors: Number of processors in the system
            processor_names: Optional custom names for processors
        """
        self.num_processors = num_processors
        
        if processor_names is None:
            self.processor_names = [f"P{i}" for i in range(num_processors)]
        else:
            if len(processor_names) != num_processors:
                raise ValueError(
                    f"Number of processor names ({len(processor_names)}) "
                    f"does not match num_processors ({num_processors})"
                )
            self.processor_names = processor_names
    
    def get_processor_name(self, processor_id: int) -> str:
        """
        Get the name of a processor by its ID.
        
        Args:
            processor_id: Processor index (0-based)
            
        Returns:
            Processor name
        """
        return self.processor_names[processor_id]
    
    def __repr__(self) -> str:
        """String representation of the SystemModel."""
        return f"SystemModel(processors={self.num_processors})"
