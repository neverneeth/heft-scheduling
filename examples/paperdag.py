import sys

sys.path.append('..')
from src.core import WorkflowDAG
from src.algorithms import HEFTAlgorithm, QLHEFTLargeState, QLHEFTSmallState
from src.utils import Visualizer, SanityChecker
from src.utils.schedule_validator import validate_schedule
# E[i] = edge from E[i][0] to E[i][1]

E = [('T1', 'T2'), ('T1', 'T3'), ('T1', 'T4'),
     ('T1', 'T5'), ('T1', 'T6'), ('T2', 'T8'),
     ('T2', 'T9'), ('T3', 'T7'), ('T4', 'T8'),
     ('T4', 'T9'), ('T5', 'T9'), ('T6', 'T8'),
     ('T7', 'T10'), ('T8', 'T10'), ('T9', 'T10')]

# w[i][j] = cost of computation of ith task on jth processor

W = [[14, 16, 9], [13, 19, 18], [11, 13, 19],
     [13, 8, 17], [12, 13, 10], [13, 16, 9],
     [7, 15, 11], [5, 11, 14], [18, 12, 20],
     [21, 7, 16]]

# sparse matrix representation
# c[i][2] = average cost of communication between c[i][0]th task to c[i][1]th task between distinct processors

c = {("T1", "T2"): 18, ("T1", "T3"): 12, ("T1", "T4"): 9,
     ("T1", "T5"): 11, ("T1", "T6"): 14, ("T2", "T8"): 19,
     ("T2", "T9"): 16, ("T3", "T7"): 23, ("T4", "T8"): 27,
     ("T4", "T9"): 23, ("T5", "T9"): 13, ("T6", "T8"): 15,
     ("T7", "T10"): 17, ("T8", "T10"): 11, ("T9", "T10"): 13}

dag = WorkflowDAG(E, W, c)
Visualizer.visualize_dag(dag, title="Example DAG for Algorithm Comparison")

algorithms = [
                HEFTAlgorithm(),
                QLHEFTLargeState(num_episodes=5000),
                QLHEFTSmallState(num_episodes=10000, convergence_threshold=0.1)
            ]

for i, algo in enumerate(algorithms):
    print("\n" + "=" * 80)
    print(f"RUNNING ALGORITHM {i+1}/{len(algorithms)}: {algo.name}")
    print("=" * 80)
    
    result = algo.schedule(dag)
    is_valid = validate_schedule(dag, result, verbose=True)
    print(f"\nResults for {algo.name}:")
    print(f"  Makespan: {result.makespan:.2f}")
    print(f"  Average Processor Utilization: {result.get_average_utilization():.2f}%")

    Visualizer.visualize_gantt_chart(result, title=f"Gantt Chart - {algo.name}")