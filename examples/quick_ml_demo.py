"""
Quick ML Pipeline Demo: Simplified version to demonstrate the ML functionality.

This runs a smaller version of the ML pipeline to show the Dataset V1 and Model V1 
working correctly without the full computational overhead.
"""

import os
import time
from typing import Dict, List
import pandas as pd

from src.ml import ExpertDatasetGenerator, RegressionHEFT
from src.algorithms.heft import HEFTAlgorithm
from src.utils.dag_generator import DAGGenerator
from src.core.workflow_dag import WorkflowDAG


def main():
    """Run a quick ML pipeline demo."""
    print("🚀 Quick ML Pipeline Demo for HEFT Scheduling")
    print("="*60)
    
    output_dir = "ml_quick_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate small expert dataset
    print("\n📊 Step 1: Generating Expert Dataset (Dataset V1)")
    
    dataset_generator = ExpertDatasetGenerator(
        output_dir=os.path.join(output_dir, "datasets")
    )
    
    # Generate smaller dataset for demo
    dataset_df = dataset_generator.generate_dataset_v1(
        num_dags=20,  # Small number for quick demo
        ql_runs_per_dag=2,  # Fewer runs for speed
        save_intermediate=False
    )
    
    dataset_path = os.path.join(output_dir, "datasets", "dataset_v1.csv")
    print(f"✅ Dataset generated: {len(dataset_df)} samples")
    
    # Step 2: Train a simple model
    print("\n🤖 Step 2: Training XGBoost Model (Model V1)")
    
    reg_heft = RegressionHEFT(model_type="xgboost")
    
    train_results = reg_heft.train(
        dataset_path=dataset_path,
        test_size=0.2,
        tune_hyperparameters=False,  # Skip tuning for speed
        save_model_path=os.path.join(output_dir, "models", "xgboost_model.joblib")
    )
    
    print(f"✅ Model trained: R² = {train_results['test_metrics']['r2']:.3f}")
    
    # Step 3: Test on a single DAG
    print("\n🎯 Step 3: Testing on Sample DAG")
    
    # Generate a simple test DAG
    edges, costs, comm = DAGGenerator.generate_random_dag(
        num_tasks=10,
        num_processors=3,
        edge_probability=0.3,
        random_seed=123
    )
    test_dag = WorkflowDAG(edges, costs, comm)
    
    # Compare algorithms
    heft_original = HEFTAlgorithm()
    
    # Test original HEFT
    start_time = time.time()
    heft_result = heft_original.schedule(test_dag)
    heft_time = time.time() - start_time
    
    # Test regression HEFT
    start_time = time.time()
    ml_result = reg_heft.schedule(test_dag)
    ml_time = time.time() - start_time
    
    print(f"📈 Results:")
    print(f"  Original HEFT: {heft_result.makespan:.2f} (in {heft_time*1000:.2f}ms)")
    print(f"  Regression HEFT: {ml_result.makespan:.2f} (in {ml_time*1000:.2f}ms)")
    
    improvement = (heft_result.makespan - ml_result.makespan) / heft_result.makespan * 100
    speed_ratio = heft_time / ml_time if ml_time > 0 else float('inf')
    
    if improvement > 0:
        print(f"  🎉 Improvement: {improvement:.2f}% better makespan")
    else:
        print(f"  📊 Performance: {abs(improvement):.2f}% difference")
    
    if ml_time <= 0 or heft_time <= 0:
        print(f"  ⚡ Speed: execution time too fast to measure accurately")
    elif speed_ratio > 1:
        print(f"  🚀 Speed: {speed_ratio:.1f}x faster")
    else:
        print(f"  ⏱️ Speed: {1/speed_ratio:.1f}x slower")
    
    # Step 4: Show feature importance
    print("\n📈 Step 4: Feature Importance Analysis")
    try:
        importance = reg_heft.get_feature_importance(top_n=10)
        print("  Top 10 Most Important Features:")
        for i, (feature, value) in enumerate(importance.items(), 1):
            print(f"    {i:2d}. {feature}: {value:.4f}")
    except Exception as e:
        print(f"  ⚠️ Feature importance not available: {e}")
    
    print(f"\n✅ Quick Demo Complete!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  - Generated {len(dataset_df)} training samples")
    print(f"  - Trained XGBoost model with R² = {train_results['test_metrics']['r2']:.3f}")
    print(f"  - Tested on 10-task DAG with {improvement:+.2f}% performance difference")
    print(f"  - ML-enhanced HEFT demonstrates regression-based task prioritization")


if __name__ == "__main__":
    main()