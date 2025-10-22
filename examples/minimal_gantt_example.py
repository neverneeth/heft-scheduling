"""
Minimal Gantt Chart Example: The easiest way to visualize ML-HEFT results.

This shows the absolute minimum code needed to:
1. Load a trained model
2. Schedule a DAG  
3. Create a Gantt chart
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT
from src.algorithms.heft import HEFTAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer
from src.utils.schedule_validator import validate_schedule

# Step 1: Load your trained model
ml_heft = RegressionHEFT()
ml_heft.load_model("models/smart_xgboost_model.joblib")

# Step 2: Create a DAG to schedule
dag_generator = DAGGenerator()
E, W, c = dag_generator.generate_random_dag(num_tasks=6, num_processors=3, random_seed=42)
dag = WorkflowDAG(E, W, c)

# Step 3: Schedule with ML-HEFT
ml_result = ml_heft.schedule(dag)
print(f"ML-HEFT makespan: {ml_result.makespan:.2f}")

# Step 4: Create Gantt chart
visualizer = Visualizer()
visualizer.visualize_gantt_chart(
    ml_result,
    title=f"My ML-HEFT Schedule (Makespan: {ml_result.makespan:.2f})",
    save_path="my_gantt_chart.png",
    show=False  # Set to True if you want to display it
)

is_valid = validate_schedule(dag, ml_result, verbose=True)

print("✅ Gantt chart saved as 'my_gantt_chart.png'")
print("💡 That's it! Only 4 steps to visualize your ML-HEFT results!")