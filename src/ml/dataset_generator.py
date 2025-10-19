"""
Expert Dataset Generator: Creates training data for imitation learning.

This module implements the "Heuristic Oracle" approach where we generate
expert schedules by running HEFT once and QL-HEFT multiple times,
keeping the best schedule as ground truth.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.workflow_dag import WorkflowDAG
from src.algorithms.heft import HEFTAlgorithm
from src.algorithms.qlheft import QLHEFTLargeState, QLHEFTSmallState
from src.utils.dag_generator import DAGGenerator


class ExpertDatasetGenerator:
    """
    Generates expert datasets for imitation learning by comparing
    HEFT and QL-HEFT algorithms on diverse DAG instances.
    """
    
    def __init__(self, output_dir: str = "datasets"):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize algorithms
        self.heft = HEFTAlgorithm()
        self.ql_large = QLHEFTLargeState(num_episodes=5000)
        self.ql_small = QLHEFTSmallState(num_episodes=5000)
        
    def generate_dataset_v1(
        self,
        num_dags: int = 1000,
        dag_configs: List[Dict] = None,
        ql_runs_per_dag: int = 10,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Generate Dataset V1: Expert schedules from best HEFT/QL-HEFT runs.
        
        Args:
            num_dags: Number of DAGs to generate
            dag_configs: List of DAG configuration parameters
            ql_runs_per_dag: Number of QL-HEFT runs per DAG
            save_intermediate: Save progress periodically
            
        Returns:
            DataFrame with expert schedules and metadata
        """
        print(f"🚀 Generating Dataset V1 with {num_dags} DAGs")
        print(f"📊 Running HEFT once + QL-HEFT {ql_runs_per_dag} times per DAG")
        
        if dag_configs is None:
            dag_configs = self._get_default_dag_configs()
        
        dataset = []
        dag_metadata = []
        
        for dag_id in tqdm(range(num_dags), desc="Generating expert data"):
            try:
                # Select random configuration
                config = np.random.choice(dag_configs)
                
                # Generate DAG
                dag, dag_info = self._generate_dag_with_config(config, dag_id)
                
                # Find expert schedule
                expert_result, algorithm_results = self._find_expert_schedule(
                    dag, ql_runs_per_dag
                )
                
                # Extract features and labels
                dag_data = self._extract_dag_features(
                    dag, expert_result, algorithm_results, dag_info
                )
                
                dataset.extend(dag_data)
                dag_metadata.append({
                    'dag_id': dag_id,
                    'config': config,
                    'expert_makespan': expert_result.makespan,
                    'expert_algorithm': expert_result.algorithm_name,
                    'dag_info': dag_info
                })
                
                # Save intermediate results
                if save_intermediate and (dag_id + 1) % 100 == 0:
                    self._save_intermediate_results(
                        dataset, dag_metadata, dag_id + 1
                    )
                    
            except Exception as e:
                print(f"❌ Error processing DAG {dag_id}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Save final dataset
        self._save_final_dataset(df, dag_metadata, "dataset_v1")
        
        print(f"✅ Generated Dataset V1: {len(df)} task samples from {len(dag_metadata)} DAGs")
        print(f"📁 Saved to: {self.output_dir}/dataset_v1.csv")
        
        return df
    
    def _get_default_dag_configs(self) -> List[Dict]:
        """Get default DAG generation configurations for diverse dataset."""
        configs = []
        
        # Random DAGs with varying parameters
        for num_tasks in [10, 15, 20, 25, 30, 40, 50]:
            for num_processors in [3, 4, 5, 6]:
                for edge_prob in [0.2, 0.3, 0.4]:
                    configs.append({
                        'type': 'random',
                        'num_tasks': num_tasks,
                        'num_processors': num_processors,
                        'edge_probability': edge_prob,
                        'computation_cost_range': (10, 50),
                        'communication_cost_range': (5, 20)
                    })
        
        # Layered DAGs
        for num_layers in [3, 4, 5]:
            for tasks_per_layer in [3, 4, 5, 6]:
                for num_processors in [3, 4, 5]:
                    configs.append({
                        'type': 'layered',
                        'num_layers': num_layers,
                        'tasks_per_layer': tasks_per_layer,
                        'num_processors': num_processors,
                        'edge_density': 0.5,
                        'computation_cost_range': (10, 50),
                        'communication_cost_range': (5, 20)
                    })
        
        # Fork-Join DAGs
        for initial in [2, 3]:
            for parallel in [6, 8, 10]:
                for final in [2, 3]:
                    for num_processors in [3, 4, 5]:
                        configs.append({
                            'type': 'fork_join',
                            'num_initial_tasks': initial,
                            'num_parallel_tasks': parallel,
                            'num_final_tasks': final,
                            'num_processors': num_processors,
                            'computation_cost_range': (10, 50),
                            'communication_cost_range': (5, 20)
                        })
        
        # Benchmark DAGs
        for chi in [4, 5, 6]:
            for num_processors in [3, 4, 5]:
                configs.append({
                    'type': 'gaussian_elimination',
                    'chi': chi,
                    'num_processors': num_processors,
                    'computation_cost_range': (10, 50),
                    'communication_cost_range': (5, 20)
                })
        
        for gamma in [2, 3, 4]:
            for num_processors in [3, 4, 5]:
                configs.append({
                    'type': 'epigenomics',
                    'gamma': gamma,
                    'num_processors': num_processors,
                    'computation_cost_range': (10, 50),
                    'communication_cost_range': (5, 20)
                })
        
        return configs
    
    def _generate_dag_with_config(self, config: Dict, dag_id: int) -> Tuple[WorkflowDAG, Dict]:
        """Generate a DAG based on configuration."""
        config = config.copy()
        dag_type = config.pop('type')
        
        # Add random seed for reproducibility
        config['random_seed'] = dag_id * 42
        
        if dag_type == 'random':
            edges, costs, comm = DAGGenerator.generate_random_dag(**config)
        elif dag_type == 'layered':
            dag = DAGGenerator.generate_layered_dag(**config)
            edges = list(dag.graph.edges())
            costs = dag.computation_costs
            comm = dag.communication_costs
        elif dag_type == 'fork_join':
            dag = DAGGenerator.generate_fork_join_dag(**config)
            edges = list(dag.graph.edges())
            costs = dag.computation_costs
            comm = dag.communication_costs
        elif dag_type == 'gaussian_elimination':
            edges, costs, comm = DAGGenerator.generate_gaussian_elimination_dag(**config)
        elif dag_type == 'epigenomics':
            edges, costs, comm = DAGGenerator.generate_epigenomics_dag(**config)
        elif dag_type == 'laplace':
            edges, costs, comm = DAGGenerator.generate_laplace_dag(**config)
        elif dag_type == 'stencil':
            edges, costs, comm = DAGGenerator.generate_stencil_dag(**config)
        else:
            raise ValueError(f"Unknown DAG type: {dag_type}")
        
        # Create WorkflowDAG
        if dag_type in ['layered', 'fork_join']:
            dag_obj = dag
        else:
            dag_obj = WorkflowDAG(edges, costs, comm)
        
        # Collect metadata
        dag_info = {
            'dag_id': dag_id,
            'type': dag_type,
            'num_tasks': dag_obj.num_tasks,
            'num_processors': dag_obj.num_processors,
            'num_edges': len(dag_obj.graph.edges()),
            'config': config
        }
        
        return dag_obj, dag_info
    
    def _find_expert_schedule(self, dag: WorkflowDAG, ql_runs: int) -> Tuple[Any, Dict]:
        """
        Find the expert schedule by running HEFT once and QL-HEFT multiple times.
        
        Returns:
            Best result and dictionary of all algorithm results
        """
        results = {}
        
        # Run HEFT once
        start_time = time.time()
        heft_result = self.heft.schedule(dag)
        heft_time = time.time() - start_time
        
        results['heft'] = {
            'result': heft_result,
            'makespan': heft_result.makespan,
            'time': heft_time
        }
        
        best_result = heft_result
        best_makespan = heft_result.makespan
        
        # Run QL-HEFT Large State multiple times
        ql_large_results = []
        for run in range(ql_runs):
            start_time = time.time()
            ql_result = self.ql_large.schedule(dag)
            ql_time = time.time() - start_time
            
            ql_large_results.append({
                'result': ql_result,
                'makespan': ql_result.makespan,
                'time': ql_time
            })
            
            if ql_result.makespan < best_makespan:
                best_result = ql_result
                best_makespan = ql_result.makespan
        
        results['ql_large'] = ql_large_results
        
        # Run QL-HEFT Small State multiple times
        ql_small_results = []
        for run in range(ql_runs):
            start_time = time.time()
            ql_result = self.ql_small.schedule(dag)
            ql_time = time.time() - start_time
            
            ql_small_results.append({
                'result': ql_result,
                'makespan': ql_result.makespan,
                'time': ql_time
            })
            
            if ql_result.makespan < best_makespan:
                best_result = ql_result
                best_makespan = ql_result.makespan
        
        results['ql_small'] = ql_small_results
        
        return best_result, results
    
    def _extract_dag_features(
        self, 
        dag: WorkflowDAG, 
        expert_result: Any, 
        algorithm_results: Dict,
        dag_info: Dict
    ) -> List[Dict]:
        """
        Extract features for each task in the DAG.
        
        Returns:
            List of feature dictionaries, one per task
        """
        from .feature_extractor import TaskFeatureExtractor
        
        extractor = TaskFeatureExtractor()
        task_features = []
        
        # Get expert task order (this is our ground truth)
        expert_task_order = self._extract_task_order(expert_result)
        
        # Calculate upward ranks (target labels)
        upward_ranks = self._calculate_upward_ranks(dag)
        
        for task in dag.task_list:
            features = extractor.extract_task_features(dag, task)
            
            # Add expert information
            features.update({
                'dag_id': dag_info['dag_id'],
                'dag_type': dag_info['type'],
                'task_id': task,
                'expert_priority': expert_task_order.get(task, len(dag.task_list)),
                'upward_rank': upward_ranks[task],  # This is our regression target
                'expert_makespan': expert_result.makespan,
                'expert_algorithm': expert_result.algorithm_name
            })
            
            task_features.append(features)
        
        return task_features
    
    def _extract_task_order(self, result) -> Dict[str, int]:
        """Extract task scheduling order from result."""
        task_order = {}
        
        # Sort tasks by their start time in the schedule
        scheduled_tasks = [(task, info['start_time']) 
                         for task, info in result.task_schedule.items()]
        scheduled_tasks.sort(key=lambda x: x[1])
        
        # Assign priority based on scheduling order (earlier = higher priority)
        for priority, (task, _) in enumerate(scheduled_tasks):
            task_order[task] = priority
        
        return task_order
    
    def _calculate_upward_ranks(self, dag: WorkflowDAG) -> Dict[str, float]:
        """Calculate upward rank for each task (our regression target)."""
        rank = {}
        
        # Process tasks in reverse topological order
        topological_order = dag.get_topological_order()
        
        for task in reversed(topological_order):
            w_avg = dag.get_avg_computation_cost(task)
            successors = dag.get_successors(task)
            
            if not successors:
                # Exit task: rank is just the average computation cost
                rank[task] = w_avg
            else:
                # Rank is avg computation cost + max{comm_cost + successor_rank}
                max_successor_cost = max(
                    dag.get_communication_cost(task, succ) + rank[succ]
                    for succ in successors
                )
                rank[task] = w_avg + max_successor_cost
        
        return rank
    
    def _save_intermediate_results(self, dataset: List, metadata: List, count: int):
        """Save intermediate results."""
        df = pd.DataFrame(dataset)
        intermediate_path = os.path.join(self.output_dir, f"dataset_v1_intermediate_{count}.csv")
        df.to_csv(intermediate_path, index=False)
        
        metadata_path = os.path.join(self.output_dir, f"metadata_v1_intermediate_{count}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_final_dataset(self, df: pd.DataFrame, metadata: List, name: str):
        """Save final dataset and metadata."""
        # Save dataset
        dataset_path = os.path.join(self.output_dir, f"{name}.csv")
        df.to_csv(dataset_path, index=False)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary statistics
        summary = {
            'total_tasks': len(df),
            'total_dags': len(metadata),
            'dag_types': df['dag_type'].value_counts().to_dict(),
            'expert_algorithms': df['expert_algorithm'].value_counts().to_dict(),
            'makespan_stats': {
                'mean': df['expert_makespan'].mean(),
                'std': df['expert_makespan'].std(),
                'min': df['expert_makespan'].min(),
                'max': df['expert_makespan'].max()
            },
            'upward_rank_stats': {
                'mean': df['upward_rank'].mean(),
                'std': df['upward_rank'].std(),
                'min': df['upward_rank'].min(),
                'max': df['upward_rank'].max()
            }
        }
        
        summary_path = os.path.join(self.output_dir, f"{name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📈 Dataset Summary:")
        print(f"   Total tasks: {summary['total_tasks']}")
        print(f"   Total DAGs: {summary['total_dags']}")
        print(f"   Expert algorithms: {summary['expert_algorithms']}")
        print(f"   Makespan range: {summary['makespan_stats']['min']:.2f} - {summary['makespan_stats']['max']:.2f}")


def main():
    """Example usage of the dataset generator."""
    generator = ExpertDatasetGenerator("datasets")
    
    # Generate a smaller dataset for testing
    df = generator.generate_dataset_v1(
        num_dags=50,  # Start with smaller number for testing
        ql_runs_per_dag=5
    )
    
    print("Dataset generation complete!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()