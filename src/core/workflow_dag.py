"""
WorkflowDAG: Directed Acyclic Graph representation for task workflows.

This module provides a comprehensive DAG implementation that encapsulates
task dependencies, computation costs, and communication costs for workflow
scheduling problems.
"""

import networkx as nx
import re
from typing import List, Dict, Tuple, Set, Optional
import copy


class WorkflowDAG:
    """
    Represents a workflow as a Directed Acyclic Graph (DAG).
    
    Each task in the workflow has:
    - Computation costs on different processors
    - Communication costs for data transfer between dependent tasks
    - Dependencies (predecessor and successor relationships)
    
    Attributes:
        graph (nx.DiGraph): NetworkX directed graph representing task dependencies
        task_list (List[str]): Ordered list of task identifiers
        task_index (Dict[str, int]): Mapping from task ID to its index
        computation_costs (List[List[float]]): Matrix W[i][j] = cost of task i on processor j
        communication_costs (Dict[Tuple[str, str], float]): Edge weights for inter-task communication
        num_tasks (int): Total number of tasks in the workflow
        num_processors (int): Number of available processors
    """
    
    def __init__(
        self,
        edges: List[Tuple[str, str]],
        computation_costs: List[List[float]],
        communication_costs: Dict[Tuple[str, str], float]
    ):
        """
        Initialize a WorkflowDAG.
        
        Args:
            edges: List of (source, destination) tuples representing task dependencies
            computation_costs: 2D array where W[i][j] = execution time of task i on processor j
            communication_costs: Dictionary mapping (source, dest) to communication time
            
        Raises:
            ValueError: If the graph contains cycles or computation costs are inconsistent
        """
        self.graph = nx.DiGraph(edges)
        
        # Validate DAG property
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The provided graph contains cycles and is not a valid DAG")
        
    # Build task mapping with natural sorting (fix for task ordering)
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', s)]
        self.task_list = sorted(self.graph.nodes(), key=natural_sort_key)
        self.task_index = {task: idx for idx, task in enumerate(self.task_list)}
        
        # Store costs
        self.computation_costs = computation_costs
        self.communication_costs = communication_costs
        
        # Metadata
        self.num_tasks = len(self.task_list)
        self.num_processors = len(computation_costs[0]) if computation_costs else 0
        self.processor_list = [f'p{i}' for i in range(self.num_processors)]
        
        # Validate computation costs dimensions
        if len(computation_costs) != self.num_tasks:
            raise ValueError(
                f"Computation costs has {len(computation_costs)} rows "
                f"but there are {self.num_tasks} tasks"
            )
        
        # Add layer information for visualization
        self._compute_layers()
    
    def _compute_layers(self):
        """Compute topological layers for visualization purposes."""
        for layer, nodes in enumerate(nx.topological_generations(self.graph)):
            for node in nodes:
                self.graph.nodes[node]["layer"] = layer
    
    def get_predecessors(self, task: str) -> List[str]:
        """
        Get all immediate predecessors of a task.
        
        Args:
            task: Task identifier
            
        Returns:
            List of predecessor task identifiers
        """
        return list(self.graph.predecessors(task))
    
    def get_successors(self, task: str) -> List[str]:
        """
        Get all immediate successors of a task.
        
        Args:
            task: Task identifier
            
        Returns:
            List of successor task identifiers
        """
        return list(self.graph.successors(task))
    
    def get_entry_tasks(self) -> List[str]:
        """
        Get all entry tasks (tasks with no predecessors).
        
        Returns:
            List of entry task identifiers
        """
        return [task for task in self.graph.nodes if self.graph.in_degree(task) == 0]
    
    def get_exit_tasks(self) -> List[str]:
        """
        Get all exit tasks (tasks with no successors).
        
        Returns:
            List of exit task identifiers
        """
        return [task for task in self.graph.nodes if self.graph.out_degree(task) == 0]
    
    def get_viable_tasks(self, scheduled: Set[str]) -> List[str]:
        """
        Get tasks that are ready to be scheduled (all predecessors are scheduled).
        
        Args:
            scheduled: Set of already scheduled task identifiers
            
        Returns:
            List of tasks whose predecessors are all scheduled
        """
        return [
            task for task in self.graph.nodes
            if task not in scheduled and set(self.get_predecessors(task)).issubset(scheduled)
        ]
    
    def get_computation_cost(self, task: str, processor: int) -> float:
        """
        Get the execution time of a task on a specific processor.
        
        Args:
            task: Task identifier
            processor: Processor index
            
        Returns:
            Execution time
        """
        task_idx = self.task_index[task]
        if isinstance(processor, str) and processor.startswith('p'):
            # Format: 'p0', 'p1', etc.
            processor_idx = int(processor[1:])
        elif isinstance(processor, int):
            # Format: 0, 1, 2, etc.
            processor_idx = processor
        else:
            raise ValueError(f"Invalid processor format: {processor}. " 
                            f"Expected integer or string like 'p0'")
        return self.computation_costs[task_idx][processor_idx]
    
    def get_communication_cost(self, source: str, dest: str) -> float:
        """
        Get the communication cost between two tasks.
        
        Args:
            source: Source task identifier
            dest: Destination task identifier
            
        Returns:
            Communication cost (0 if no edge exists)
        """
        return self.communication_costs.get((source, dest), 0)
    
    def get_avg_computation_cost(self, task: str) -> float:
        """
        Get the average computation cost of a task across all processors.
        
        Args:
            task: Task identifier
            
        Returns:
            Average execution time
        """
        task_idx = self.task_index[task]
        return sum(self.computation_costs[task_idx]) / len(self.computation_costs[task_idx])
    
    def get_topological_order(self) -> List[str]:
        """
        Get a topological ordering of tasks.
        
        Returns:
            List of tasks in topological order
        """
        return list(nx.topological_sort(self.graph))
    
    def copy(self) -> 'WorkflowDAG':
        """
        Create a deep copy of this WorkflowDAG.
        
        Returns:
            A new WorkflowDAG instance with copied data
        """
        edges = list(self.graph.edges())
        computation_costs = copy.deepcopy(self.computation_costs)
        communication_costs = copy.deepcopy(self.communication_costs)
        return WorkflowDAG(edges, computation_costs, communication_costs)
    
    def __repr__(self) -> str:
        """String representation of the WorkflowDAG."""
        return (
            f"WorkflowDAG(tasks={self.num_tasks}, "
            f"processors={self.num_processors}, "
            f"edges={len(self.graph.edges())})"
        )
