"""
Quick Gantt Chart Demo: Simple example of visualizing ML-HEFT schedules.

This script demonstrates how to:
1. Load a trained ML-HEFT model
2. Schedule a DAG with both ML-HEFT and original HEFT
3. Create beautiful Gantt chart visualizations
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.ml import RegressionHEFT, ExpertDatasetGenerator
from src.algorithms.heft import HEFTAlgorithm
from src.core.workflow_dag import WorkflowDAG
from src.utils.dag_generator import DAGGenerator
from src.utils.visualizer import Visualizer
import time

def create_gantt_visualization():
    """Create and save Gantt chart visualizations."""
    print("🎨 Quick Gantt Chart Demo")
    print("="*50)
    
    # Step 1: Get or create a trained model
    model_path = "models/quick_demo_model.joblib"
    
    if os.path.exists(model_path):
        print(f"📁 Loading existing model...")
        ml_heft = RegressionHEFT()
        ml_heft.load_model(model_path)
    else:
        print(f"🤖 Training new model...")
        ml_heft = RegressionHEFT(model_type="xgboost")
        
        # Quick training
        dataset_gen = ExpertDatasetGenerator()
        training_data = dataset_gen.generate_dataset_v1(num_dags=15)
        ml_heft.train(training_data, save_model_path=model_path)
    
    # Step 2: Generate a test DAG
    print("📊 Generating test DAG...")
    dag_generator = DAGGenerator()
    E, W, c = dag_generator.generate_random_dag(
        num_tasks=12,
        num_processors=4,
        edge_probability=0.35,
        random_seed=42
    )
    dag = WorkflowDAG(E, W, c)
    print(f"Generated DAG: {dag.num_tasks} tasks, {dag.num_processors} processors")
    
    # Step 3: Schedule with both algorithms
    print("⚡ Scheduling with both algorithms...")
    
    # ML-HEFT scheduling
    ml_start = time.time()
    ml_result = ml_heft.schedule(dag)
    ml_time = time.time() - ml_start
    
    # Original HEFT scheduling
    heft_original = HEFTAlgorithm()
    heft_start = time.time()
    heft_result = heft_original.schedule(dag)
    heft_time = time.time() - heft_start
    
    # Step 4: Display results
    print(f"\n📈 Scheduling Results:")
    print(f"  ML-HEFT makespan:      {ml_result.makespan:.2f} (in {ml_time*1000:.1f}ms)")
    print(f"  Original HEFT makespan: {heft_result.makespan:.2f} (in {heft_time*1000:.1f}ms)")
    
    difference = abs(ml_result.makespan - heft_result.makespan)
    percent_diff = (difference / heft_result.makespan) * 100
    print(f"  Absolute difference:    {difference:.2f}")
    print(f"  Percentage difference:  {percent_diff:.2f}%")
    
    if ml_result.makespan < heft_result.makespan:
        print("  🎉 ML-HEFT performed better!")
    elif ml_result.makespan > heft_result.makespan:
        print("  📈 Original HEFT performed better")
    else:
        print("  🤝 Both algorithms achieved the same makespan")
    
    # Step 5: Create visualizations
    print(f"\n🎨 Creating Gantt chart visualizations...")
    visualizer = Visualizer()
    
    # Create output directory
    output_dir = "gantt_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Individual Gantt charts
        ml_chart_path = os.path.join(output_dir, "ml_heft_schedule.png")
        visualizer.visualize_gantt_chart(
            dag, 
            ml_result, 
            title=f"ML-HEFT Schedule (Makespan: {ml_result.makespan:.2f})",
            save_path=ml_chart_path
        )
        print(f"  ✅ ML-HEFT chart: {ml_chart_path}")
        
        heft_chart_path = os.path.join(output_dir, "original_heft_schedule.png")
        visualizer.visualize_gantt_chart(
            dag, 
            heft_result, 
            title=f"Original HEFT Schedule (Makespan: {heft_result.makespan:.2f})",
            save_path=heft_chart_path
        )
        print(f"  ✅ Original HEFT chart: {heft_chart_path}")
        
        # Algorithm comparison
        comparison_path = os.path.join(output_dir, "algorithm_comparison.png")
        visualizer.compare_algorithms(
            dag, 
            {"ML-HEFT": ml_result, "Original HEFT": heft_result},
            save_path=comparison_path
        )
        print(f"  ✅ Comparison chart: {comparison_path}")
        
        # DAG structure
        dag_path = os.path.join(output_dir, "dag_structure.png")
        visualizer.visualize_dag(
            dag, 
            title=f"DAG Structure ({dag.num_tasks} tasks)",
            save_path=dag_path
        )
        print(f"  ✅ DAG structure: {dag_path}")
        
        print(f"\n📂 All visualizations saved in: {output_dir}/")
        print("💡 Open the PNG files to see your Gantt charts!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        print("💡 Make sure matplotlib is installed: pip install matplotlib")
        return False

def main():
    """Main function."""
    success = create_gantt_visualization()
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print("📊 Your Gantt charts are ready to view!")
    else:
        print(f"\n❌ Demo encountered errors.")
        print("💡 Check that all dependencies are installed.")

if __name__ == "__main__":
    main()