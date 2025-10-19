"""
Simple Gantt Visualization Demo: Shows how to visualize ML-HEFT results.

This script demonstrates the simplest way to create Gantt chart visualizations
for your ML-HEFT scheduling results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT, ExpertDatasetGenerator
from src.algorithms.heft import HEFTAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer

def simple_visualization_demo():
    """Create Gantt charts with minimal training."""
    print("🎨 Simple Gantt Visualization Demo")
    print("="*50)
    
    # Check if we have an existing model
    existing_models = [
        "models/smart_xgboost_model.joblib",
        "models/trained_xgboost.joblib", 
        "models/quick_demo_model.joblib",
        "../ml_demo_output/models/xgboost_heft_model.joblib",
        "../ml_quick_demo/models/xgboost_model.joblib"
    ]
    
    ml_heft = None
    for model_path in existing_models:
        if os.path.exists(model_path):
            print(f"📁 Found existing model: {model_path}")
            try:
                ml_heft = RegressionHEFT()
                ml_heft.load_model(model_path)
                print("✅ Model loaded successfully!")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue
    
    # If no model found, train a very small one
    if ml_heft is None:
        print("🤖 No existing model found. Training a minimal model...")
        ml_heft = RegressionHEFT(model_type="xgboost")
        
        # Generate minimal training data (very small)
        dataset_gen = ExpertDatasetGenerator()
        training_data = dataset_gen.generate_dataset_v1(
            num_dags=5,  # Very small for speed
            ql_runs_per_dag=2  # Minimal Q-learning runs
        )
        
        # Train with minimal hyperparameter tuning
        ml_heft.train(training_data, tune_hyperparameters=False)
        print("✅ Quick model trained!")
    
    # Generate a test DAG
    print("📊 Generating test DAG...")
    dag_generator = DAGGenerator()
    E, W, c = dag_generator.generate_random_dag(
        num_tasks=8,  # Small DAG for clear visualization
        num_processors=3,
        edge_probability=0.4,
        random_seed=123
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Generated DAG: {dag.num_tasks} tasks, {dag.num_processors} processors")
    
    # Schedule with both algorithms
    print("⚡ Scheduling...")
    ml_result = ml_heft.schedule(dag)
    heft_original = HEFTAlgorithm()
    heft_result = heft_original.schedule(dag)
    
    print(f"  ML-HEFT makespan:      {ml_result.makespan:.2f}")
    print(f"  Original HEFT makespan: {heft_result.makespan:.2f}")
    
    # Create visualizations
    print(f"🎨 Creating Gantt charts...")
    visualizer = Visualizer()
    output_dir = "simple_gantt_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ML-HEFT Gantt chart
        ml_path = os.path.join(output_dir, "ml_heft_gantt.png")
        visualizer.visualize_gantt_chart(
            ml_result, 
            title=f"ML-HEFT Schedule (Makespan: {ml_result.makespan:.2f})",
            save_path=ml_path,
            show=False
        )
        print(f"✅ ML-HEFT chart: {ml_path}")
        
        # Original HEFT Gantt chart  
        heft_path = os.path.join(output_dir, "original_heft_gantt.png")
        visualizer.visualize_gantt_chart(
            heft_result, 
            title=f"Original HEFT Schedule (Makespan: {heft_result.makespan:.2f})",
            save_path=heft_path,
            show=False
        )
        print(f"✅ Original HEFT chart: {heft_path}")
        
        # DAG structure
        dag_path = os.path.join(output_dir, "dag_structure.png")
        visualizer.visualize_dag(
            dag, 
            title=f"DAG Structure ({dag.num_tasks} tasks)",
            save_path=dag_path,
            show=False
        )
        print(f"✅ DAG structure: {dag_path}")
        
        print(f"\n📂 All visualizations saved in: {output_dir}/")
        print("💡 Open the PNG files to see your Gantt charts!")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {str(e)}")
        print("💡 Make sure matplotlib is installed: pip install matplotlib")
        return False

if __name__ == "__main__":
    success = simple_visualization_demo()
    if success:
        print("\n🎉 Visualization demo completed!")
        print("📊 Your Gantt charts are ready!")
    else:
        print("\n❌ Demo failed. Check error messages above.")