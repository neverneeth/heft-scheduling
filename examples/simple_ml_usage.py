"""
Simple example: How to use a pre-trained model efficiently.

This shows the most convenient way to get predictions without retraining.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator

def predict_makespan_simple(dag, model_path="models/quick_model.joblib"):
    """
    Get makespan prediction for a DAG using pre-trained model.
    
    Args:
        dag: WorkflowDAG object
        model_path: Path to the trained model file
        
    Returns:
        float: Predicted makespan
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("🔧 You need to train a model first. Run smart_mltester.py")
        return None
    
    # Load and use model (very fast!)
    ml_heft = RegressionHEFT()
    ml_heft.load_model(model_path)
    
    # Get prediction
    result = ml_heft.schedule(dag)
    return result.makespan

def main():
    """Simple demo of using pre-trained model."""
    print("🚀 Simple ML-HEFT Usage Demo")
    print("="*40)
    
    # Create a test DAG
    dag_generator = DAGGenerator()
    E, W, c = dag_generator.generate_random_dag(
        num_tasks=12,
        num_processors=4,
        edge_probability=0.3,
        random_seed=12345
    )
    dag = WorkflowDAG(E, W, c)
    
    print(f"Generated DAG: {dag.num_tasks} tasks, {dag.num_processors} processors")
    
    # Get prediction (loads model automatically)
    predicted_makespan = predict_makespan_simple(dag)
    
    if predicted_makespan:
        print(f"Predicted makespan: {predicted_makespan:.2f}")
        print("\n💡 This was instant - no training required!")
    else:
        print("Run smart_mltester.py first to train a model.")

if __name__ == "__main__":
    main()