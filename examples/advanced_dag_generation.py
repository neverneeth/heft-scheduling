"""
Advanced DAG Generation Example

This script demonstrates various DAG generation options and
how to customize DAG characteristics.
"""

import sys
sys.path.append('..')

from src.utils import DAGGenerator, Visualizer
from src.algorithms import HEFTAlgorithm


def main():
    """Demonstrate different DAG generation methods."""
    print("Advanced DAG Generation Examples\n")
    print("=" * 80)
    
    # Example 1: Random DAG
    print("\n[1] Generating Random DAG...")
    random_dag = DAGGenerator.generate_random_dag(
        num_tasks=12,
        num_processors=4,
        edge_probability=0.4,
        computation_cost_range=(10, 50),
        communication_cost_range=(5, 20),
        random_seed=42
    )
    print(f"    Generated: {random_dag}")
    Visualizer.visualize_dag(random_dag, title="Random DAG")
    
    # Example 2: Layered DAG
    print("\n[2] Generating Layered DAG...")
    layered_dag = DAGGenerator.generate_layered_dag(
        num_layers=4,
        tasks_per_layer=3,
        num_processors=3,
        edge_density=0.6,
        random_seed=42
    )
    print(f"    Generated: {layered_dag}")
    Visualizer.visualize_dag(layered_dag, title="Layered DAG")
    
    # Example 3: Fork-Join DAG
    print("\n[3] Generating Fork-Join DAG...")
    fork_join_dag = DAGGenerator.generate_fork_join_dag(
        num_initial_tasks=2,
        num_parallel_tasks=8,
        num_final_tasks=2,
        num_processors=4,
        random_seed=42
    )
    print(f"    Generated: {fork_join_dag}")
    Visualizer.visualize_dag(fork_join_dag, title="Fork-Join DAG")
    
    # Example 4: Schedule one of them
    print("\n[4] Scheduling the Fork-Join DAG with HEFT...")
    heft = HEFTAlgorithm()
    result = heft.schedule(fork_join_dag)
    print(f"    Makespan: {result.makespan:.2f}")
    print(f"    Avg Utilization: {result.get_average_utilization():.2f}%")
    Visualizer.visualize_gantt_chart(result)
    
    print("\n✓ All DAG generation examples complete!")


if __name__ == "__main__":
    main()
