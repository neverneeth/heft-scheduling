"""
Debug feature extraction issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml import TaskFeatureExtractor
from src.utils.dag_generator import DAGGenerator
from src.core.workflow_dag import WorkflowDAG

# Generate a simple test DAG
edges, costs, comm = DAGGenerator.generate_random_dag(
    num_tasks=5,
    num_processors=3,
    edge_probability=0.4,
    random_seed=42
)

print("Generated DAG:")
print(f"Edges: {edges}")
print(f"Costs type: {type(costs)}")
print(f"Costs sample: {list(costs.items())[:2] if hasattr(costs, 'items') else costs}")
print(f"Comm type: {type(comm)}")

dag = WorkflowDAG(edges, costs, comm)
print(f"DAG created with {dag.num_tasks} tasks")
print(f"Task list: {dag.task_list}")

# Try to extract features
extractor = TaskFeatureExtractor()
first_task = dag.task_list[0]
print(f"Extracting features for task: {first_task}")

try:
    features = extractor.extract_task_features(dag, first_task)
    print("Features extracted successfully!")
    print(f"Number of features: {len(features)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()