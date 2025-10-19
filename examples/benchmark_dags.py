"""
Benchmark DAG Examples

This script demonstrates generating and visualizing standard benchmark DAGs:
- Gaussian Elimination
- Epigenomics  
- Laplace
- Stencil

These are common in workflow scheduling research and are converted from
message-passing models to direct-edge DAGs compatible with HEFT.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer
from src.algorithms.heft import HEFTAlgorithm

def main():
    """Generate and visualize benchmark DAGs."""
    
    # Parameters
    num_processors = 3
    
    # 1. Gaussian Elimination
    print("\n" + "=" * 80)
    print("GAUSSIAN ELIMINATION DAG")
    print("=" * 80)
    chi = 5  # Parameter controlling size
    E, W, c = DAGGenerator.generate_gaussian_elimination_dag(
        chi=chi,
        num_processors=num_processors,
        computation_cost_range=(10, 30),
        communication_cost_range=(5, 15),
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Tasks: {dag.num_tasks}, Edges: {len(dag.graph.edges())}")
    Visualizer.visualize_dag(dag, title=f"Gaussian Elimination DAG (χ={chi})")
    
    # Run HEFT on the DAG
    heft = HEFTAlgorithm()
    result = heft.schedule(dag)
    print(f"Makespan: {result.makespan:.2f}")
    Visualizer.visualize_gantt_chart(result, title=f"Gaussian Elimination Schedule")
    
    # 2. Epigenomics
    print("\n" + "=" * 80)
    print("EPIGENOMICS DAG")
    print("=" * 80)
    gamma = 3  # Parallel branches
    E, W, c = DAGGenerator.generate_epigenomics_dag(
        gamma=gamma,
        num_processors=num_processors,
        computation_cost_range=(10, 30),
        communication_cost_range=(5, 15),
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Tasks: {dag.num_tasks}, Edges: {len(dag.graph.edges())}")
    Visualizer.visualize_dag(dag, title=f"Epigenomics DAG (γ={gamma})")
    
    # Run HEFT on the DAG
    result = heft.schedule(dag)
    print(f"Makespan: {result.makespan:.2f}")
    Visualizer.visualize_gantt_chart(result, title=f"Epigenomics Schedule")
    
    # 3. Laplace
    print("\n" + "=" * 80)
    print("LAPLACE DAG")
    print("=" * 80)
    phi = 4  # Matrix size
    E, W, c = DAGGenerator.generate_laplace_dag(
        phi=phi,
        num_processors=num_processors,
        computation_cost_range=(10, 30),
        communication_cost_range=(5, 15),
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Tasks: {dag.num_tasks}, Edges: {len(dag.graph.edges())}")
    Visualizer.visualize_dag(dag, title=f"Laplace DAG (φ={phi})")
    
    # Run HEFT on the DAG
    result = heft.schedule(dag)
    print(f"Makespan: {result.makespan:.2f}")
    Visualizer.visualize_gantt_chart(result, title=f"Laplace Schedule")
    
    # 4. Stencil
    print("\n" + "=" * 80)
    print("STENCIL DAG")
    print("=" * 80)
    xi = 3  # Grid size
    E, W, c = DAGGenerator.generate_stencil_dag(
        xi=xi,
        num_processors=num_processors,
        computation_cost_range=(10, 30),
        communication_cost_range=(5, 15),
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Tasks: {dag.num_tasks}, Edges: {len(dag.graph.edges())}")
    Visualizer.visualize_dag(dag, title=f"Stencil DAG (ξ={xi})")
    
    # Run HEFT on the DAG
    result = heft.schedule(dag)
    print(f"Makespan: {result.makespan:.2f}")
    Visualizer.visualize_gantt_chart(result, title=f"Stencil Schedule")
    
    print("\n" + "=" * 80)
    print("BENCHMARK DAG GENERATION COMPLETE")
    print("=" * 80)
    print("All standard benchmark DAGs have been generated and scheduled.")
    print("Check the generated plots for visual comparison of DAG structures and schedules.")

if __name__ == "__main__":
    main()