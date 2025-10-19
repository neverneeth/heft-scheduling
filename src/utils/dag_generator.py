"""
DAG Generator: Utilities for creating random workflow DAGs.

This module provides functions to generate random DAGs with various
topological structures and cost characteristics for testing and
experimentation.

For HEFT and QLHEFT scheduling algorithms.
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
    ) -> Tuple[List[Tuple[str, str]], List[List[float]], Dict[Tuple[str, str], float]]:
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
            Tuple of (edges, computation_costs, communication_costs)
            
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
            
        # Ensure all tasks are connected (have at least one edge)
        connected_tasks = set()
        for src, dst in edges:
            connected_tasks.add(src)
            connected_tasks.add(dst)
        
        # Add any isolated tasks by connecting them to existing tasks
        isolated_tasks = set(task_list) - connected_tasks
        for task in isolated_tasks:
            # Find which layer this task belongs to
            for layer_idx, layer_tasks in enumerate(layers):
                if task in layer_tasks:
                    task_layer = layer_idx
                    break
            
            if task_layer == 0 and len(layers) > 1:  
                # Connect to a task in the next layer
                if layers[1]:
                    edges.append((task, random.choice(layers[1])))
            elif task_layer == len(layers) - 1 and task_layer > 0:
                # Connect from a task in the previous layer
                if layers[task_layer - 1]:
                    edges.append((random.choice(layers[task_layer - 1]), task))
            else:
                # Choose randomly to connect from previous or to next layer
                if random.choice([True, False]) and task_layer > 0 and layers[task_layer - 1]:
                    # Connect from previous layer
                    edges.append((random.choice(layers[task_layer - 1]), task))
                elif task_layer < len(layers) - 1 and layers[task_layer + 1]:
                    # Connect to next layer
                    edges.append((task, random.choice(layers[task_layer + 1])))
                else:
                    # Fall back: connect to or from any other task
                    other_task = random.choice([t for t in task_list if t != task])
                    if random.choice([True, False]):
                        edges.append((task, other_task))
                    else:
                        edges.append((other_task, task))
        
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
        
        # Return the tuple instead of constructing WorkflowDAG directly
        # This avoids potential issues with how WorkflowDAG handles the input
        return edges, computation_costs, communication_costs
    
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
        
        # Generate computation costs - use actual task count from edges
        all_tasks = set()
        for source, dest in edges:
            all_tasks.add(source)
            all_tasks.add(dest)
        actual_num_tasks = len(all_tasks)
        
        computation_costs = []
        for _ in range(actual_num_tasks):
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
    
    @staticmethod
    def generate_gaussian_elimination_dag(
        chi: int,
        num_processors: int,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> Tuple[List[Tuple[str, str]], List[List[float]], Dict[Tuple[str, str], float]]:
        """
        Generate a Gaussian Elimination DAG.
        
        Args:
            chi: Number of equations (parameter controlling DAG size)
            num_processors: Number of processors in the target system
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            Tuple of (edges, computation_costs, communication_costs)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate expected task count (χ² + χ - 2) / 2
        expected_task_count = (chi**2 + chi - 2) // 2
        task_nodes = [f"T{i+1}" for i in range(expected_task_count)]
        
        # Create layers structure
        layer = 0
        layers = []
        current_n = chi
        node_idx = 0
        
        while current_n > 1:
            # Add pivot node
            layers.append([task_nodes[node_idx]])
            node_idx += 1
            layer += 1
            
            # Add elimination nodes
            elim_nodes = []
            for _ in range(current_n-1):
                elim_nodes.append(task_nodes[node_idx])
                node_idx += 1
            layers.append(elim_nodes)
            layer += 1
            current_n -= 1
        
        # Generate edges (direct task-to-task)
        edges = []
        for l in range(len(layers)):
            if l == 0:
                continue
            if l % 2 == 1:
                for node in layers[l]:
                    # All elimination nodes depend on previous pivot
                    edges.append((layers[l-1][0], node))
                    
                # Diagonal dependencies to next elimination layer
                if l+2 < len(layers):
                    for j in range(1, len(layers[l])):
                        edges.append((layers[l][j], layers[l+2][j-1]))
                        
                # Connect first elimination to next pivot
                if l+1 < len(layers):
                    edges.append((layers[l][0], layers[l+1][0]))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(expected_task_count):
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
        
        return edges, computation_costs, communication_costs
    
    @staticmethod
    def generate_epigenomics_dag(
        gamma: int,
        num_processors: int,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> Tuple[List[Tuple[str, str]], List[List[float]], Dict[Tuple[str, str], float]]:
        """
        Generate an Epigenomics workflow DAG.
        
        Args:
            gamma: Number of parallel branches
            num_processors: Number of processors in the target system
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            Tuple of (edges, computation_costs, communication_costs)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate number of tasks: 4γ + 4
        n_tasks = 4 * gamma + 4
        
        # Create edges first to determine which tasks actually exist
        edges = []
        
        # Initial task (T1) connects to first γ tasks (T2 to T(γ+1))
        for i in range(1, gamma+1):
            edges.append((f"T1", f"T{i+1}"))
        
        # Connect parallel branches through all stages
        for i in range(1, gamma+1):
            for j in range(3):
                from_idx = i + j*gamma + 1
                to_idx = i + (j+1)*gamma + 1
                if to_idx <= n_tasks:
                    edges.append((f"T{from_idx}", f"T{to_idx}"))
        
        # Connect last stage to join node
        join_node_idx = 4*gamma + 2  # T(4γ+2)
        for i in range(1, gamma+1):
            last_stage_idx = i + 3*gamma + 1
            if last_stage_idx <= n_tasks and join_node_idx <= n_tasks:
                edges.append((f"T{last_stage_idx}", f"T{join_node_idx}"))
        
        # Connect final pipeline
        for j in range(join_node_idx, n_tasks):
            edges.append((f"T{j}", f"T{j+1}"))
        
        # Get unique task nodes from edges
        all_tasks = set()
        for src, dst in edges:
            all_tasks.add(src)
            all_tasks.add(dst)
        
        # Sort tasks naturally and create final task list
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split(r'(\d+)', s)]
        
        task_nodes = sorted(list(all_tasks), key=natural_sort_key)
        actual_n_tasks = len(task_nodes)
        
        # Generate computation costs
        computation_costs = []
        for _ in range(actual_n_tasks):
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
        
        return edges, computation_costs, communication_costs
    
    @staticmethod
    def generate_laplace_dag(
        phi: int,
        num_processors: int,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> Tuple[List[Tuple[str, str]], List[List[float]], Dict[Tuple[str, str], float]]:
        """
        Generate a Laplace solver DAG.
        
        Args:
            phi: Matrix size parameter
            num_processors: Number of processors in the target system
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            Tuple of (edges, computation_costs, communication_costs)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate number of tasks: ϕ²
        n_tasks = phi**2
        task_nodes = [f"T{i+1}" for i in range(n_tasks)]
        
        # Create a simpler tree-like structure for Laplace
        edges = []
        
        # Create layers
        nodes_per_layer = []
        layer_starts = []
        current_node = 0
        
        # First half: expanding layers
        for level in range(1, phi):
            layer_starts.append(current_node)
            nodes_per_layer.append(level)
            current_node += level
            
        # Second half: contracting layers  
        for level in range(phi, 0, -1):
            layer_starts.append(current_node)
            nodes_per_layer.append(level)
            current_node += level
        
        # Connect layers
        for layer_idx in range(len(layer_starts) - 1):
            current_layer_start = layer_starts[layer_idx]
            current_layer_size = nodes_per_layer[layer_idx]
            next_layer_start = layer_starts[layer_idx + 1]
            next_layer_size = nodes_per_layer[layer_idx + 1]
            
            # Connect each task in current layer to tasks in next layer
            for i in range(current_layer_size):
                current_task_idx = current_layer_start + i
                if current_task_idx < n_tasks:
                    # Connect to 1-2 tasks in next layer
                    for j in range(min(2, next_layer_size)):
                        next_task_idx = next_layer_start + (i + j) % next_layer_size
                        if next_task_idx < n_tasks and current_task_idx < next_task_idx:
                            edges.append((task_nodes[current_task_idx], task_nodes[next_task_idx]))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(n_tasks):
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
        
        return edges, computation_costs, communication_costs
    
    @staticmethod
    def generate_stencil_dag(
        xi: int,
        num_processors: int,
        computation_cost_range: Tuple[float, float] = (5, 50),
        communication_cost_range: Tuple[float, float] = (1, 20),
        random_seed: Optional[int] = None
    ) -> Tuple[List[Tuple[str, str]], List[List[float]], Dict[Tuple[str, str], float]]:
        """
        Generate a Stencil computation DAG.
        
        Args:
            xi: Size parameter (xi x xi grid)
            num_processors: Number of processors in the target system
            computation_cost_range: (min, max) range for task execution times
            communication_cost_range: (min, max) range for data transfer times
            random_seed: Seed for reproducibility
            
        Returns:
            Tuple of (edges, computation_costs, communication_costs)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Calculate number of tasks: xi * xi
        n_tasks = xi * xi
        node_count = 0
        task_nodes = [f"T{i+1}" for i in range(n_tasks)]
        
        # Create layers
        layers = []
        for i in range(xi):
            layer = []
            for j in range(xi):
                layer.append(task_nodes[node_count])
                node_count += 1
            layers.append(layer)

        # Add dependencies (direct edges)
        edges = []
        for l in range(1, xi):
            for i in range(xi):
                for j in range(max(0, i-1), min(i+2, xi)):
                    edges.append((layers[l-1][j], layers[l][i]))
        
        # Generate computation costs
        computation_costs = []
        for _ in range(n_tasks):
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
        
        return edges, computation_costs, communication_costs
