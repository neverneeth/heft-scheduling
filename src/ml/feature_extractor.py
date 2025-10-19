"""
Task Feature Extractor: Extracts features from DAG tasks for machine learning.

This module extracts comprehensive features from tasks within a workflow DAG
to enable regression-based task prioritization.
"""

from typing import Dict, List, Set, Any
import numpy as np
import networkx as nx
from src.core.workflow_dag import WorkflowDAG


class TaskFeatureExtractor:
    """
    Extracts machine learning features from DAG tasks.
    
    Features are organized into categories:
    1. Task-centric: Properties of the task itself
    2. Predecessor-based: Properties related to predecessor tasks
    3. Successor-based: Properties related to successor tasks  
    4. Graph-level: Global DAG properties
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_task_features(self, dag: WorkflowDAG, task: str) -> Dict[str, float]:
        """
        Extract all features for a given task in a DAG.
        
        Args:
            dag: The workflow DAG
            task: Task identifier
            
        Returns:
            Dictionary of feature name -> feature value
        """
        features = {}
        
        # Task-centric features
        features.update(self._extract_task_centric_features(dag, task))
        
        # Predecessor-based features
        features.update(self._extract_predecessor_features(dag, task))
        
        # Successor-based features
        features.update(self._extract_successor_features(dag, task))
        
        # Graph-level features
        features.update(self._extract_graph_level_features(dag, task))
        
        return features
    
    def _extract_task_centric_features(self, dag: WorkflowDAG, task: str) -> Dict[str, float]:
        """Extract features related to the task itself."""
        features = {}
        
        # Computation costs
        comp_costs = [dag.get_computation_cost(task, i) 
                     for i in range(dag.num_processors)]
        
        features['comp_cost_min'] = min(comp_costs)
        features['comp_cost_max'] = max(comp_costs)
        features['comp_cost_mean'] = np.mean(comp_costs)
        features['comp_cost_std'] = np.std(comp_costs)
        features['comp_cost_range'] = max(comp_costs) - min(comp_costs)
        
        # Computation cost ratios
        if features['comp_cost_mean'] > 0:
            features['comp_cost_cv'] = features['comp_cost_std'] / features['comp_cost_mean']
        else:
            features['comp_cost_cv'] = 0
        
        # Best processor ratio
        best_processor_cost = min(comp_costs)
        worst_processor_cost = max(comp_costs)
        
        if worst_processor_cost > 0:
            features['processor_efficiency_ratio'] = best_processor_cost / worst_processor_cost
        else:
            features['processor_efficiency_ratio'] = 1.0
        
        return features
    
    def _extract_predecessor_features(self, dag: WorkflowDAG, task: str) -> Dict[str, float]:
        """Extract features related to predecessor tasks."""
        features = {}
        
        predecessors = dag.get_predecessors(task)
        
        # Basic predecessor counts
        features['num_predecessors'] = len(predecessors)
        features['is_entry_task'] = float(len(predecessors) == 0)
        
        if len(predecessors) == 0:
            # Entry task - set default values
            features.update({
                'pred_comp_cost_mean': 0.0,
                'pred_comp_cost_max': 0.0,
                'pred_comp_cost_sum': 0.0,
                'pred_comm_cost_mean': 0.0,
                'pred_comm_cost_max': 0.0,
                'pred_comm_cost_sum': 0.0,
                'pred_comm_comp_ratio_mean': 0.0,
                'pred_depth_mean': 0.0,
                'pred_depth_max': 0.0,
                'pred_out_degree_mean': 0.0,
                'pred_out_degree_max': 0.0
            })
            return features
        
        # Predecessor computation costs
        pred_comp_costs = []
        for pred in predecessors:
            pred_costs = [dag.get_computation_cost(pred, i) 
                         for i in range(dag.num_processors)]
            pred_comp_costs.append(np.mean(pred_costs))
        
        features['pred_comp_cost_mean'] = np.mean(pred_comp_costs)
        features['pred_comp_cost_max'] = max(pred_comp_costs)
        features['pred_comp_cost_sum'] = sum(pred_comp_costs)
        
        # Predecessor communication costs
        pred_comm_costs = []
        for pred in predecessors:
            comm_cost = dag.get_communication_cost(pred, task)
            pred_comm_costs.append(comm_cost)
        
        features['pred_comm_cost_mean'] = np.mean(pred_comm_costs)
        features['pred_comm_cost_max'] = max(pred_comm_costs)
        features['pred_comm_cost_sum'] = sum(pred_comm_costs)
        
        # Communication to computation ratios
        comm_comp_ratios = []
        for i, pred in enumerate(predecessors):
            pred_comp_mean = pred_comp_costs[i]
            comm_cost = pred_comm_costs[i]
            if pred_comp_mean > 0:
                comm_comp_ratios.append(comm_cost / pred_comp_mean)
            else:
                comm_comp_ratios.append(0)
        
        features['pred_comm_comp_ratio_mean'] = np.mean(comm_comp_ratios)
        
        # Predecessor depths in DAG
        pred_depths = [self._get_task_depth(dag, pred) for pred in predecessors]
        features['pred_depth_mean'] = np.mean(pred_depths)
        features['pred_depth_max'] = max(pred_depths)
        
        # Predecessor out-degrees
        pred_out_degrees = [dag.graph.out_degree(pred) for pred in predecessors]
        features['pred_out_degree_mean'] = np.mean(pred_out_degrees)
        features['pred_out_degree_max'] = max(pred_out_degrees)
        
        return features
    
    def _extract_successor_features(self, dag: WorkflowDAG, task: str) -> Dict[str, float]:
        """Extract features related to successor tasks."""
        features = {}
        
        successors = dag.get_successors(task)
        
        # Basic successor counts
        features['num_successors'] = len(successors)
        features['is_exit_task'] = float(len(successors) == 0)
        
        if len(successors) == 0:
            # Exit task - set default values
            features.update({
                'succ_comp_cost_mean': 0.0,
                'succ_comp_cost_max': 0.0,
                'succ_comp_cost_sum': 0.0,
                'succ_comm_cost_mean': 0.0,
                'succ_comm_cost_max': 0.0,
                'succ_comm_cost_sum': 0.0,
                'succ_comm_comp_ratio_mean': 0.0,
                'succ_height_mean': 0.0,
                'succ_height_max': 0.0,
                'succ_in_degree_mean': 0.0,
                'succ_in_degree_max': 0.0
            })
            return features
        
        # Successor computation costs
        succ_comp_costs = []
        for succ in successors:
            succ_costs = [dag.get_computation_cost(succ, i) 
                         for i in range(dag.num_processors)]
            succ_comp_costs.append(np.mean(succ_costs))
        
        features['succ_comp_cost_mean'] = np.mean(succ_comp_costs)
        features['succ_comp_cost_max'] = max(succ_comp_costs)
        features['succ_comp_cost_sum'] = sum(succ_comp_costs)
        
        # Successor communication costs
        succ_comm_costs = []
        for succ in successors:
            comm_cost = dag.get_communication_cost(task, succ)
            succ_comm_costs.append(comm_cost)
        
        features['succ_comm_cost_mean'] = np.mean(succ_comm_costs)
        features['succ_comm_cost_max'] = max(succ_comm_costs)
        features['succ_comm_cost_sum'] = sum(succ_comm_costs)
        
        # Communication to computation ratios
        comm_comp_ratios = []
        for i, succ in enumerate(successors):
            succ_comp_mean = succ_comp_costs[i]
            comm_cost = succ_comm_costs[i]
            if succ_comp_mean > 0:
                comm_comp_ratios.append(comm_cost / succ_comp_mean)
            else:
                comm_comp_ratios.append(0)
        
        features['succ_comm_comp_ratio_mean'] = np.mean(comm_comp_ratios)
        
        # Successor heights in DAG (distance to exit tasks)
        succ_heights = [self._get_task_height(dag, succ) for succ in successors]
        features['succ_height_mean'] = np.mean(succ_heights)
        features['succ_height_max'] = max(succ_heights)
        
        # Successor in-degrees
        succ_in_degrees = [dag.graph.in_degree(succ) for succ in successors]
        features['succ_in_degree_mean'] = np.mean(succ_in_degrees)
        features['succ_in_degree_max'] = max(succ_in_degrees)
        
        return features
    
    def _extract_graph_level_features(self, dag: WorkflowDAG, task: str) -> Dict[str, float]:
        """Extract features related to the overall DAG structure."""
        features = {}
        
        # Basic DAG properties
        features['dag_num_tasks'] = dag.num_tasks
        features['dag_num_processors'] = dag.num_processors
        features['dag_num_edges'] = dag.graph.number_of_edges()
        features['dag_edge_density'] = dag.graph.number_of_edges() / (dag.num_tasks * (dag.num_tasks - 1))
        
        # Task position in DAG
        features['task_depth'] = self._get_task_depth(dag, task)
        features['task_height'] = self._get_task_height(dag, task)
        
        # Normalized position
        max_depth = self._get_max_depth(dag)
        max_height = self._get_max_height(dag)
        
        if max_depth > 0:
            features['task_depth_normalized'] = features['task_depth'] / max_depth
        else:
            features['task_depth_normalized'] = 0
            
        if max_height > 0:
            features['task_height_normalized'] = features['task_height'] / max_height
        else:
            features['task_height_normalized'] = 0
        
        # Critical path features
        features['dag_critical_path_length'] = self._get_critical_path_length(dag)
        features['task_on_critical_path'] = float(self._is_on_critical_path(dag, task))
        
        # Centrality measures
        features['task_betweenness_centrality'] = nx.betweenness_centrality(dag.graph)[task]
        features['task_closeness_centrality'] = nx.closeness_centrality(dag.graph)[task]
        
        # Task degree features
        features['task_in_degree'] = dag.graph.in_degree(task)
        features['task_out_degree'] = dag.graph.out_degree(task)
        features['task_total_degree'] = features['task_in_degree'] + features['task_out_degree']
        
        # Relative degree features
        max_in_degree = max(dag.graph.in_degree(n) for n in dag.graph.nodes())
        max_out_degree = max(dag.graph.out_degree(n) for n in dag.graph.nodes())
        
        if max_in_degree > 0:
            features['task_in_degree_normalized'] = features['task_in_degree'] / max_in_degree
        else:
            features['task_in_degree_normalized'] = 0
            
        if max_out_degree > 0:
            features['task_out_degree_normalized'] = features['task_out_degree'] / max_out_degree
        else:
            features['task_out_degree_normalized'] = 0
        
        # Communication vs computation ratio for entire DAG
        total_comp_cost = sum(
            min(dag.get_computation_cost(t, i) for i in range(dag.num_processors))
            for t in dag.task_list
        )
        
        total_comm_cost = sum(
            dag.get_communication_cost(u, v)
            for u, v in dag.graph.edges()
        )
        
        if total_comp_cost > 0:
            features['dag_comm_comp_ratio'] = total_comm_cost / total_comp_cost
        else:
            features['dag_comm_comp_ratio'] = 0
        
        return features
    
    def _get_task_depth(self, dag: WorkflowDAG, task: str) -> int:
        """Get the depth of a task (distance from entry tasks)."""
        if task not in dag.graph:
            return 0
            
        # Find entry tasks (tasks with no predecessors)
        entry_tasks = [t for t in dag.task_list if dag.graph.in_degree(t) == 0]
        
        if task in entry_tasks:
            return 0
        
        # Use BFS to find shortest path from any entry task
        min_depth = float('inf')
        
        for entry in entry_tasks:
            try:
                depth = nx.shortest_path_length(dag.graph, entry, task)
                min_depth = min(min_depth, depth)
            except nx.NetworkXNoPath:
                continue
        
        return min_depth if min_depth != float('inf') else 0
    
    def _get_task_height(self, dag: WorkflowDAG, task: str) -> int:
        """Get the height of a task (distance to exit tasks)."""
        if task not in dag.graph:
            return 0
            
        # Find exit tasks (tasks with no successors)
        exit_tasks = [t for t in dag.task_list if dag.graph.out_degree(t) == 0]
        
        if task in exit_tasks:
            return 0
        
        # Use BFS to find shortest path to any exit task
        min_height = float('inf')
        
        for exit_task in exit_tasks:
            try:
                height = nx.shortest_path_length(dag.graph, task, exit_task)
                min_height = min(min_height, height)
            except nx.NetworkXNoPath:
                continue
        
        return min_height if min_height != float('inf') else 0
    
    def _get_max_depth(self, dag: WorkflowDAG) -> int:
        """Get the maximum depth in the DAG."""
        entry_tasks = [t for t in dag.task_list if dag.graph.in_degree(t) == 0]
        
        if not entry_tasks:
            return 0
        
        max_depth = 0
        for task in dag.task_list:
            depth = self._get_task_depth(dag, task)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _get_max_height(self, dag: WorkflowDAG) -> int:
        """Get the maximum height in the DAG."""
        exit_tasks = [t for t in dag.task_list if dag.graph.out_degree(t) == 0]
        
        if not exit_tasks:
            return 0
        
        max_height = 0
        for task in dag.task_list:
            height = self._get_task_height(dag, task)
            max_height = max(max_height, height)
        
        return max_height
    
    def _get_critical_path_length(self, dag: WorkflowDAG) -> float:
        """Calculate the critical path length."""
        # Use the upward rank calculation which gives us the critical path
        upward_ranks = {}
        
        # Process in reverse topological order
        topological_order = dag.get_topological_order()
        
        for task in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(task)
            successors = dag.get_successors(task)
            
            if not successors:
                upward_ranks[task] = w_avg
            else:
                max_successor_cost = max(
                    dag.get_communication_cost(task, succ) + upward_ranks[succ]
                    for succ in successors
                )
                upward_ranks[task] = w_avg + max_successor_cost
        
        # Critical path length is the maximum upward rank
        return max(upward_ranks.values()) if upward_ranks else 0
    
    def _is_on_critical_path(self, dag: WorkflowDAG, task: str) -> bool:
        """Check if a task is on the critical path."""
        # This is a simplified check - a more accurate implementation
        # would require tracking the actual critical path
        
        # For now, consider tasks with high upward ranks as critical
        upward_ranks = {}
        topological_order = dag.get_topological_order()
        
        for t in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(t)
            successors = dag.get_successors(t)
            
            if not successors:
                upward_ranks[t] = w_avg
            else:
                max_successor_cost = max(
                    dag.get_communication_cost(t, succ) + upward_ranks[succ]
                    for succ in successors
                )
                upward_ranks[t] = w_avg + max_successor_cost
        
        # Consider a task critical if its upward rank is close to the maximum
        max_rank = max(upward_ranks.values())
        task_rank = upward_ranks[task]
        
        # Threshold: within 10% of maximum rank
        threshold = 0.9 * max_rank
        return task_rank >= threshold
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be extracted."""
        # This would return all feature names - useful for debugging
        return [
            # Task-centric features
            'comp_cost_min', 'comp_cost_max', 'comp_cost_mean', 'comp_cost_std',
            'comp_cost_range', 'comp_cost_cv', 'processor_efficiency_ratio',
            
            # Predecessor features
            'num_predecessors', 'is_entry_task', 'pred_comp_cost_mean',
            'pred_comp_cost_max', 'pred_comp_cost_sum', 'pred_comm_cost_mean',
            'pred_comm_cost_max', 'pred_comm_cost_sum', 'pred_comm_comp_ratio_mean',
            'pred_depth_mean', 'pred_depth_max', 'pred_out_degree_mean', 'pred_out_degree_max',
            
            # Successor features
            'num_successors', 'is_exit_task', 'succ_comp_cost_mean',
            'succ_comp_cost_max', 'succ_comp_cost_sum', 'succ_comm_cost_mean',
            'succ_comm_cost_max', 'succ_comm_cost_sum', 'succ_comm_comp_ratio_mean',
            'succ_height_mean', 'succ_height_max', 'succ_in_degree_mean', 'succ_in_degree_max',
            
            # Graph-level features
            'dag_num_tasks', 'dag_num_processors', 'dag_num_edges', 'dag_edge_density',
            'task_depth', 'task_height', 'task_depth_normalized', 'task_height_normalized',
            'dag_critical_path_length', 'task_on_critical_path', 'task_betweenness_centrality',
            'task_closeness_centrality', 'task_in_degree', 'task_out_degree', 'task_total_degree',
            'task_in_degree_normalized', 'task_out_degree_normalized', 'dag_comm_comp_ratio'
        ]


def main():
    """Example usage of the feature extractor."""
    from src.utils.dag_generator import DAGGenerator
    
    # Generate a test DAG
    edges, costs, comm = DAGGenerator.generate_random_dag(
        num_tasks=10, 
        num_processors=3,
        edge_probability=0.3
    )
    dag = WorkflowDAG(edges, costs, comm)
    
    # Extract features
    extractor = TaskFeatureExtractor()
    
    print("Extracting features for all tasks:")
    for task in dag.task_list:
        features = extractor.extract_task_features(dag, task)
        print(f"\nTask {task}: {len(features)} features")
        
        # Show a few example features
        example_features = {k: v for k, v in list(features.items())[:5]}
        for feature, value in example_features.items():
            print(f"  {feature}: {value:.3f}")


if __name__ == "__main__":
    main()