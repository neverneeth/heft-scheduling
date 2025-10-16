"""
Sanity Checker: Comprehensive testing and validation module.

This module provides utilities to quickly test and validate scheduling
algorithms by generating random DAGs, running multiple algorithms,
and visualizing the results for comparison.
"""

from typing import List, Optional, Dict, Any
import time
from src.core.workflow_dag import WorkflowDAG
from src.core.schedule_result import ScheduleResult
from src.algorithms.base import SchedulingAlgorithm
from src.algorithms.heft import HEFTAlgorithm
from src.algorithms.qlheft import QLHEFTLargeState, QLHEFTSmallState
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer


class SanityChecker:
    """
    Comprehensive sanity checker for scheduling algorithms.
    
    This class provides utilities to:
    1. Generate random DAGs
    2. Run multiple scheduling algorithms
    3. Visualize DAG structure
    4. Display Gantt charts for each algorithm
    5. Compare algorithm performance
    """
    
    def __init__(self):
        """Initialize the SanityChecker."""
        self.results: List[ScheduleResult] = []
        self.dag: Optional[WorkflowDAG] = None
        self.execution_times: Dict[str, float] = {}
    
    def run_sanity_check(
        self,
        num_tasks: int = 9,
        num_processors: int = 3,
        algorithms: Optional[List[SchedulingAlgorithm]] = None,
        dag_type: str = "random",
        random_seed: Optional[int] = None,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run a comprehensive sanity check.
        
        This method:
        1. Generates a random DAG with specified parameters
        2. Visualizes the DAG structure
        3. Runs all specified algorithms
        4. Generates Gantt charts for each algorithm
        5. Compares algorithm performance
        
        Args:
            num_tasks: Number of tasks in the workflow
            num_processors: Number of processors in the system
            algorithms: List of algorithms to test (defaults to HEFT and QL-HEFT variants)
            dag_type: Type of DAG to generate ("random", "layered", "fork_join")
            random_seed: Seed for reproducible results
            visualize: Whether to show visualizations
            
        Returns:
            Dictionary containing summary statistics and results
        """
        print("=" * 80)
        print("SCHEDULING FRAMEWORK SANITY CHECK")
        print("=" * 80)
        
        # Step 1: Generate DAG
        print(f"\n[1/5] Generating {dag_type} DAG...")
        print(f"      - Tasks: {num_tasks}")
        print(f"      - Processors: {num_processors}")
        print(f"      - Random Seed: {random_seed}")
        
        self.dag = self._generate_dag(
            dag_type=dag_type,
            num_tasks=num_tasks,
            num_processors=num_processors,
            random_seed=random_seed
        )
        
        print(f"      ✓ Generated DAG: {self.dag}")
        
        # Step 2: Visualize DAG
        if visualize:
            print("\n[2/5] Visualizing DAG structure...")
            Visualizer.visualize_dag(
                self.dag,
                title=f"Workflow DAG ({num_tasks} tasks, {num_processors} processors)"
            )
            print("      ✓ DAG visualization complete")
        else:
            print("\n[2/5] Skipping DAG visualization")
        
        # Step 3: Set up algorithms
        if algorithms is None:
            print("\n[3/5] Initializing default algorithms...")
            algorithms = [
                HEFTAlgorithm(),
                QLHEFTLargeState(num_episodes=5000),
                QLHEFTSmallState(num_episodes=10000, convergence_threshold=0.1)
            ]
        else:
            print(f"\n[3/5] Using {len(algorithms)} custom algorithms...")
        
        for algo in algorithms:
            print(f"      - {algo.name}")
        
        # Step 4: Run algorithms
        print("\n[4/5] Running scheduling algorithms...")
        self.results = []
        self.execution_times = {}
        
        for i, algo in enumerate(algorithms, 1):
            print(f"\n      [{i}/{len(algorithms)}] Running {algo.name}...")
            
            start_time = time.time()
            result = algo.schedule(self.dag)
            end_time = time.time()
            
            execution_time = end_time - start_time
            self.execution_times[algo.name] = execution_time
            self.results.append(result)
            
            print(f"           Makespan: {result.makespan:.2f}")
            print(f"           Avg Utilization: {result.get_average_utilization():.2f}%")
            print(f"           Execution Time: {execution_time:.4f}s")
        
        # Step 5: Visualize results
        if visualize:
            print("\n[5/5] Generating Gantt charts and comparisons...")
            
            for result in self.results:
                Visualizer.visualize_gantt_chart(
                    result,
                    title=f"Gantt Chart - {result.algorithm_name}"
                )
            
            # Compare algorithms
            if len(self.results) > 1:
                Visualizer.compare_algorithms(self.results)
            
            # Show convergence for QL-HEFT algorithms
            for result in self.results:
                if 'convergence_history' in result.metadata:
                    Visualizer.visualize_convergence(
                        result.metadata['convergence_history'],
                        title=f"Q-Learning Convergence - {result.algorithm_name}"
                    )
            
            print("      ✓ All visualizations complete")
        else:
            print("\n[5/5] Skipping visualizations")
        
        # Generate summary
        summary = self._generate_summary()
        
        print("\n" + "=" * 80)
        print("SANITY CHECK COMPLETE")
        print("=" * 80)
        print(summary['text_summary'])
        
        return summary
    
    def _generate_dag(
        self,
        dag_type: str,
        num_tasks: int,
        num_processors: int,
        random_seed: Optional[int]
    ) -> WorkflowDAG:
        """Generate a DAG based on the specified type."""
        if dag_type == "random":
            return DAGGenerator.generate_random_dag(
                num_tasks=num_tasks,
                num_processors=num_processors,
                random_seed=random_seed
            )
        elif dag_type == "layered":
            num_layers = max(2, num_tasks // 3)
            tasks_per_layer = num_tasks // num_layers
            return DAGGenerator.generate_layered_dag(
                num_layers=num_layers,
                tasks_per_layer=tasks_per_layer,
                num_processors=num_processors,
                random_seed=random_seed
            )
        elif dag_type == "fork_join":
            num_initial = max(1, num_tasks // 4)
            num_parallel = max(1, num_tasks // 2)
            num_final = num_tasks - num_initial - num_parallel
            return DAGGenerator.generate_fork_join_dag(
                num_initial_tasks=num_initial,
                num_parallel_tasks=num_parallel,
                num_final_tasks=num_final,
                num_processors=num_processors,
                random_seed=random_seed
            )
        else:
            raise ValueError(f"Unknown DAG type: {dag_type}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the sanity check results."""
        if not self.results:
            return {'text_summary': "No results available", 'data': {}}
        
        # Find best and worst makespans
        best_result = min(self.results, key=lambda r: r.makespan)
        worst_result = max(self.results, key=lambda r: r.makespan)
        
        # Calculate statistics
        makespans = [r.makespan for r in self.results]
        avg_makespan = sum(makespans) / len(makespans)
        
        utilizations = [r.get_average_utilization() for r in self.results]
        avg_utilization = sum(utilizations) / len(utilizations)
        
        # Create text summary
        lines = [
            "\nRESULTS SUMMARY",
            "-" * 80,
            f"\nDAG Configuration:",
            f"  Tasks: {self.dag.num_tasks}",
            f"  Processors: {self.dag.num_processors}",
            f"  Edges: {len(self.dag.graph.edges())}",
            f"\nAlgorithm Performance:",
        ]
        
        for result in self.results:
            exec_time = self.execution_times.get(result.algorithm_name, 0)
            lines.append(
                f"  {result.algorithm_name:25s} | "
                f"Makespan: {result.makespan:8.2f} | "
                f"Util: {result.get_average_utilization():5.2f}% | "
                f"Time: {exec_time:.4f}s"
            )
        
        lines.extend([
            f"\nStatistics:",
            f"  Best Makespan: {best_result.makespan:.2f} ({best_result.algorithm_name})",
            f"  Worst Makespan: {worst_result.makespan:.2f} ({worst_result.algorithm_name})",
            f"  Average Makespan: {avg_makespan:.2f}",
            f"  Average Utilization: {avg_utilization:.2f}%",
        ])
        
        if len(self.results) > 1:
            improvement = ((worst_result.makespan - best_result.makespan) / worst_result.makespan) * 100
            lines.append(f"  Improvement (best vs worst): {improvement:.2f}%")
        
        text_summary = "\n".join(lines)
        
        # Create data dictionary
        data = {
            'dag_info': {
                'num_tasks': self.dag.num_tasks,
                'num_processors': self.dag.num_processors,
                'num_edges': len(self.dag.graph.edges())
            },
            'results': [
                {
                    'algorithm': r.algorithm_name,
                    'makespan': r.makespan,
                    'utilization': r.get_average_utilization(),
                    'execution_time': self.execution_times.get(r.algorithm_name, 0)
                }
                for r in self.results
            ],
            'best_algorithm': best_result.algorithm_name,
            'best_makespan': best_result.makespan,
        }
        
        return {
            'text_summary': text_summary,
            'data': data
        }
    
    def get_results(self) -> List[ScheduleResult]:
        """
        Get the list of scheduling results.
        
        Returns:
            List of ScheduleResult objects
        """
        return self.results
    
    def get_dag(self) -> Optional[WorkflowDAG]:
        """
        Get the generated DAG.
        
        Returns:
            WorkflowDAG object or None if not yet generated
        """
        return self.dag


def quick_sanity_check(
    num_tasks: int = 9,
    num_processors: int = 3,
    random_seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Convenience function for quick sanity checking.
    
    Args:
        num_tasks: Number of tasks in the workflow
        num_processors: Number of processors
        random_seed: Random seed for reproducibility
        
    Returns:
        Summary dictionary
    """
    checker = SanityChecker()
    return checker.run_sanity_check(
        num_tasks=num_tasks,
        num_processors=num_processors,
        random_seed=random_seed
    )
