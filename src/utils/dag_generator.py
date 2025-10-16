"""
DAG Generator: Utilities for creating random workflow DAGs.

This module provides functions to generate random DAGs with various
topological structures and cost characteristics for testing and
experimentation.
"""

import random
import networkx as nx
from typing import List, Tuple, Dict, Optional
from src.core.workflow_dag import WorkflowDAG


class DAGGenerator:
    """
    Generator for random Directed Acyclic Graphs (DAGs) representing workflows.
    
    This class provides various methods to create synthetic workflow DAGs
    with configurable characteristics for algorithm testing and evaluation.
    """
    
    @staticmethod
    def generate_random_dag(
        num_tasks: int,
        num_processors: int,
        edge_probability: float = 0.3,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> WorkflowDAG:
        """
        Generate a random DAG with specified parameters.
        
        This method creates a random DAG by:
        1. Generating tasks in topological layers
        2. Adding edges between layers based on probability
        3. Assigning random computation and communication costs
        
        Args:
            num_tasks: Number of tasks in the workflow
            num_processors: Number of processors in the target system
            edge_probability: Probability of creating an edge between tasks in adjacent layers
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            A randomly generated WorkflowDAG
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_tasks < 1:
            raise ValueError("Number of tasks must be at least 1")
        if num_processors < 1:
            raise ValueError("Number of processors must be at least 1")
        if not 0 <= edge_probability <= 1:
            raise ValueError("Edge probability must be between 0 and 1")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Create task identifiers
        task_list = [f"T{i+1}" for i in range(num_tasks)]
        
        # Assign tasks to layers for topological structure
        num_layers = max(2, num_tasks // 3)  # At least 2 layers
        layers = [[] for _ in range(num_layers)]
        
        for i, task in enumerate(task_list):
            layer_idx = (i * num_layers) // num_tasks
            layers[layer_idx].append(task)
        
        # Generate edges between layers
        edges = []
        for layer_idx in range(num_layers - 1):
            current_layer = layers[layer_idx]
            next_layer = layers[layer_idx + 1]
            
            # Ensure connectivity: each task in next layer has at least one predecessor
            for next_task in next_layer:
                if current_layer:
                    predecessor = random.choice(current_layer)
                    edges.append((predecessor, next_task))
            
            # Add additional random edges based on probability
            for current_task in current_layer:
                for next_task in next_layer:
                    if (current_task, next_task) not in edges:
                        if random.random() < edge_probability:
                            edges.append((current_task, next_task))
        
        # Ensure at least one edge exists
        if not edges and num_tasks > 1:
            edges.append((task_list[0], task_list[1]))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(num_tasks):
            task_costs = [
                random.uniform(*computation_cost_range)
                for _ in range(num_processors)
            ]
            computation_costs.append(task_costs)
        
        # Generate communication costs for each edge
        communication_costs = {}
        for source, dest in edges:
            cost = random.uniform(*communication_cost_range)
            communication_costs[(source, dest)] = cost
        
        return WorkflowDAG(edges, computation_costs, communication_costs)
    
    @staticmethod
    def generate_layered_dag(
        num_layers: int,
        tasks_per_layer: int,
        num_processors: int,
        edge_density: float = 0.5,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> WorkflowDAG:
        """
        Generate a layered DAG with uniform layer structure.
        
        Creates a DAG where tasks are organized into distinct layers,
        with edges only going from one layer to the next. This structure
        is common in many workflow applications.
        
        Args:
            num_layers: Number of layers in the DAG
            tasks_per_layer: Number of tasks in each layer
            num_processors: Number of processors in the target system
            edge_density: Proportion of possible inter-layer edges to create
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            A layered WorkflowDAG
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        num_tasks = num_layers * tasks_per_layer
        task_list = [f"T{i+1}" for i in range(num_tasks)]
        
        # Organize tasks into layers
        layers = []
        for layer_idx in range(num_layers):
            start_idx = layer_idx * tasks_per_layer
            end_idx = start_idx + tasks_per_layer
            layers.append(task_list[start_idx:end_idx])
        
        # Generate edges between consecutive layers
        edges = []
        for layer_idx in range(num_layers - 1):
            current_layer = layers[layer_idx]
            next_layer = layers[layer_idx + 1]
            
            # Create edges based on edge density
            for current_task in current_layer:
                for next_task in next_layer:
                    if random.random() < edge_density:
                        edges.append((current_task, next_task))
            
            # Ensure each task in next layer has at least one incoming edge
            for next_task in next_layer:
                has_incoming = any(edge[1] == next_task for edge in edges)
                if not has_incoming:
                    predecessor = random.choice(current_layer)
                    edges.append((predecessor, next_task))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(num_tasks):
            task_costs = [
                random.uniform(*computation_cost_range)
                for _ in range(num_processors)
            ]
            computation_costs.append(task_costs)
        
        # Generate communication costs
        communication_costs = {}
        for source, dest in edges:
            cost = random.uniform(*communication_cost_range)
            communication_costs[(source, dest)] = cost
        
        return WorkflowDAG(edges, computation_costs, communication_costs)
    
    @staticmethod
    def generate_fork_join_dag(
        num_initial_tasks: int,
        num_parallel_tasks: int,
        num_final_tasks: int,
        num_processors: int,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> WorkflowDAG:
        """
        Generate a fork-join style DAG.
        
        Creates a DAG with initial tasks that fork into parallel tasks,
        which then join into final tasks. This pattern is common in
        MapReduce-style workflows.
        
        Args:
            num_initial_tasks: Number of initial (fork) tasks
            num_parallel_tasks: Number of parallel middle tasks
            num_final_tasks: Number of final (join) tasks
            num_processors: Number of processors in the target system
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            A fork-join WorkflowDAG
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        num_tasks = num_initial_tasks + num_parallel_tasks + num_final_tasks
        task_list = [f"T{i+1}" for i in range(num_tasks)]
        
        # Divide tasks into three groups
        initial_tasks = task_list[:num_initial_tasks]
        parallel_tasks = task_list[num_initial_tasks:num_initial_tasks + num_parallel_tasks]
        final_tasks = task_list[num_initial_tasks + num_parallel_tasks:]
        
        # Generate edges: initial -> parallel -> final
        edges = []
        
        # Fork: initial tasks to parallel tasks
        for parallel_task in parallel_tasks:
            source = random.choice(initial_tasks)
            edges.append((source, parallel_task))
        
        # Join: parallel tasks to final tasks
        for final_task in final_tasks:
            # Each final task depends on multiple parallel tasks
            num_dependencies = random.randint(1, min(3, num_parallel_tasks))
            predecessors = random.sample(parallel_tasks, num_dependencies)
            for predecessor in predecessors:
                edges.append((predecessor, final_task))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(num_tasks):
            task_costs = [
                random.uniform(*computation_cost_range)
                for _ in range(num_processors)
            ]
            computation_costs.append(task_costs)
        
        # Generate communication costs
        communication_costs = {}
        for source, dest in edges:
            cost = random.uniform(*communication_cost_range)
            communication_costs[(source, dest)] = cost
        
        return WorkflowDAG(edges, computation_costs, communication_costs)
