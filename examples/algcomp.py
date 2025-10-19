"""
Aims to compare both algorithms on a set of dags and report results. Perform experiments over 100 iterations to obtain 
statistically significant results. 

Algorithms being checked:
- HEFT 
- QL-HEFT (Large State)
- QL-HEFT (Small State)
"""
import sys

sys.path.append('..')
from src.core import WorkflowDAG
from src.algorithms import HEFTAlgorithm, QLHEFTLargeState, QLHEFTSmallState
from src.utils import Visualizer, SanityChecker
from src.utils.dag_generator import DAGGenerator

num_tasks_list = [10, 15, 20]
edge_probs = [0.2, 0.3]
num_processors_list = [4, 5, 6]


for (num_tasks, edge_prob, num_procs) in zip (num_tasks_list, edge_probs, num_processors_list):
    for i in range(2):
        print(f"\n{'='*80}\nGENERATING DAG {i+1}/30\n{'='*80}")
        dag_generator = DAGGenerator()
        E, W, c = dag_generator.generate_random_dag(
            num_tasks=num_tasks,
            num_processors=num_procs,
            edge_probability=edge_prob,
            computation_cost_range=(10, 50),
            communication_cost_range=(10, 30),
            random_seed=i
        )
        dag = WorkflowDAG(E, W, c)
        Visualizer.visualize_dag(dag, title="Example DAG for Algorithm Comparison")

        algorithms = [
                        HEFTAlgorithm(),
                        QLHEFTLargeState(num_episodes=5000),
                        QLHEFTSmallState(num_episodes=5000, convergence_threshold=0.1)
                    ]

        for i, algo in enumerate(algorithms):
            print("\n" + "=" * 80)
            print(f"RUNNING ALGORITHM {i+1}/{len(algorithms)}: {algo.name}")
            print("=" * 80)
            
            result = algo.schedule(dag)
            
            print(f"\nResults for {algo.name}:")
            print(f"  Makespan: {result.makespan:.2f}")
            print(f"  Average Processor Utilization: {result.get_average_utilization():.2f}%")

            Visualizer.visualize_gantt_chart(result, title=f"Gantt Chart - {algo.name}")