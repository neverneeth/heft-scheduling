"""
Framework Test Script

This script runs a comprehensive test of the HEFT Scheduling Framework
to verify all components are working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import WorkflowDAG, SystemModel, ScheduleResult
from src.algorithms import HEFTAlgorithm, QLHEFTLargeState, QLHEFTSmallState
from src.utils import DAGGenerator, Visualizer, SanityChecker, quick_sanity_check


def test_core_components():
    """Test core data structures."""
    print("\n" + "=" * 80)
    print("TESTING CORE COMPONENTS")
    print("=" * 80)
    
    # Test DAG creation
    print("\n[1] Testing WorkflowDAG...")
    edges = [('T1', 'T2'), ('T1', 'T3'), ('T2', 'T4'), ('T3', 'T4')]
    computation_costs = [[10, 8, 6], [12, 10, 8], [8, 6, 5], [15, 12, 10]]
    communication_costs = {('T1', 'T2'): 5, ('T1', 'T3'): 4, 
                          ('T2', 'T4'): 6, ('T3', 'T4'): 7}
    
    dag = WorkflowDAG(edges, computation_costs, communication_costs)
    print(f"    ✓ Created: {dag}")
    print(f"    ✓ Entry tasks: {dag.get_entry_tasks()}")
    print(f"    ✓ Exit tasks: {dag.get_exit_tasks()}")
    
    # Test SystemModel
    print("\n[2] Testing SystemModel...")
    system = SystemModel(num_processors=3)
    print(f"    ✓ Created: {system}")
    print(f"    ✓ Processor names: {system.processor_names}")
    
    print("\n✓ Core components test passed!")
    return dag


def test_dag_generators():
    """Test DAG generation."""
    print("\n" + "=" * 80)
    print("TESTING DAG GENERATORS")
    print("=" * 80)
    
    print("\n[1] Testing random DAG generation...")
    dag1 = DAGGenerator.generate_random_dag(
        num_tasks=8,
        num_processors=3,
        random_seed=42
    )
    print(f"    ✓ Generated: {dag1}")
    
    print("\n[2] Testing layered DAG generation...")
    dag2 = DAGGenerator.generate_layered_dag(
        num_layers=3,
        tasks_per_layer=3,
        num_processors=3,
        random_seed=42
    )
    print(f"    ✓ Generated: {dag2}")
    
    print("\n[3] Testing fork-join DAG generation...")
    dag3 = DAGGenerator.generate_fork_join_dag(
        num_initial_tasks=2,
        num_parallel_tasks=4,
        num_final_tasks=2,
        num_processors=3,
        random_seed=42
    )
    print(f"    ✓ Generated: {dag3}")
    
    print("\n✓ DAG generators test passed!")
    return dag1


def test_algorithms(dag):
    """Test scheduling algorithms."""
    print("\n" + "=" * 80)
    print("TESTING SCHEDULING ALGORITHMS")
    print("=" * 80)
    
    results = []
    
    # Test HEFT
    print("\n[1] Testing HEFT algorithm...")
    heft = HEFTAlgorithm()
    result1 = heft.schedule(dag)
    results.append(result1)
    print(f"    ✓ Makespan: {result1.makespan:.2f}")
    print(f"    ✓ Avg Utilization: {result1.get_average_utilization():.2f}%")
    
    # Test QL-HEFT Large State
    print("\n[2] Testing QL-HEFT Large State algorithm...")
    ql_large = QLHEFTLargeState(num_episodes=2000)
    result2 = ql_large.schedule(dag)
    results.append(result2)
    print(f"    ✓ Makespan: {result2.makespan:.2f}")
    print(f"    ✓ Q-table size: {result2.metadata['q_table_size']}")
    
    # Test QL-HEFT Small State
    print("\n[3] Testing QL-HEFT Small State algorithm...")
    ql_small = QLHEFTSmallState(
        num_episodes=5000,
        convergence_threshold=0.1
    )
    result3 = ql_small.schedule(dag)
    results.append(result3)
    print(f"    ✓ Makespan: {result3.makespan:.2f}")
    print(f"    ✓ Episodes run: {result3.metadata['episodes_run']}")
    
    print("\n✓ Algorithms test passed!")
    return results


def test_visualization(dag, results):
    """Test visualization tools."""
    print("\n" + "=" * 80)
    print("TESTING VISUALIZATION")
    print("=" * 80)
    
    print("\n[1] Testing DAG visualization...")
    fig1 = Visualizer.visualize_dag(dag, show=False)
    print("    ✓ DAG visualization created")
    
    print("\n[2] Testing Gantt chart visualization...")
    fig2 = Visualizer.visualize_gantt_chart(results[0], show=False)
    print("    ✓ Gantt chart created")
    
    print("\n[3] Testing algorithm comparison...")
    fig3 = Visualizer.compare_algorithms(results, show=False)
    print("    ✓ Comparison chart created")
    
    print("\n[4] Testing convergence visualization...")
    if 'convergence_history' in results[2].metadata:
        fig4 = Visualizer.visualize_convergence(
            results[2].metadata['convergence_history'],
            show=False
        )
        print("    ✓ Convergence plot created")
    
    print("\n✓ Visualization test passed!")


def test_sanity_checker():
    """Test sanity checker."""
    print("\n" + "=" * 80)
    print("TESTING SANITY CHECKER")
    print("=" * 80)
    
    print("\n[1] Running quick sanity check...")
    # Turn off visualization for automated testing
    checker = SanityChecker()
    summary = checker.run_sanity_check(
        num_tasks=6,
        num_processors=3,
        random_seed=42,
        visualize=False  # Disable plots for testing
    )
    
    print("\n✓ Sanity checker test passed!")
    return summary


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HEFT SCHEDULING FRAMEWORK TEST" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Run tests
        dag = test_core_components()
        dag = test_dag_generators()  # Use generated DAG
        results = test_algorithms(dag)
        test_visualization(dag, results)
        summary = test_sanity_checker()
        
        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe framework is working correctly and ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python examples/simple_sanity_check.py' for a visual demo")
        print("  2. Check FRAMEWORK_README.md for detailed documentation")
        print("  3. Explore examples/ directory for more usage examples")
        print("\n")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED! ✗")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
